#!/usr/bin/env python3
"""Triangulate raw Sapiens multi-view keypoints into 3D.

This script reads:
- per-camera Sapiens 2D pose JSONs under raw_data/outputs/<model>/<camera>/*.json
- crop metadata under raw_data/results_cropped/<camera>/**/crop_inference_meta.jsonl
- camera calibration from a Nerfstudio transforms.json or EasyVolcap intri/extri cameras

It remaps the crop-space keypoints back into the calibration image resolution, then
triangulates each keypoint with weighted linear + non-linear least squares. Views
with higher keypoint scores contribute more; low-confidence views contribute less.
Frames can be triangulated with between 2 and 4 cameras, using as many views as are
available for that frame/keypoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

try:
    from scipy.optimize import least_squares as scipy_least_squares
except ModuleNotFoundError:
    scipy_least_squares = None


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = Path(__file__).resolve().parent
ZONE_IDENTIFIER_SUFFIX = ":Zone.Identifier"
INVALID = -1e6


@dataclass(frozen=True)
class CameraInput:
    pose_camera: str
    calib_camera: str
    pose_dir: Path
    meta_paths: Sequence[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=RAW_DATA_ROOT / "outputs" / "0.6b",
        help="Root containing Sapiens pose JSON folders, one per camera.",
    )
    parser.add_argument(
        "--cropped-root",
        type=Path,
        default=RAW_DATA_ROOT / "results_cropped",
        help="Root containing crop_inference_meta.jsonl files, one or more per camera.",
    )
    parser.add_argument(
        "--camera-path",
        type=Path,
        default=RAW_DATA_ROOT / "transforms_room.json",
        help="Camera calibration as Nerfstudio transforms.json, a camera directory, or intri/extri yml path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=RAW_DATA_ROOT / "triangulated_poses_3d",
        help="Directory where per-frame 3D pose JSONs will be written.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="",
        help="Comma-separated raw pose camera names to use. Defaults to all cameras under --pose-root.",
    )
    parser.add_argument(
        "--camera-label-map",
        type=str,
        default="",
        help=(
            "Optional JSON object or path to a JSON file mapping raw pose camera names "
            "to calibration camera labels, e.g. '{\"test2_c1\":\"00\"}'."
        ),
    )
    parser.add_argument(
        "--scene-label",
        type=str,
        default=None,
        help="Output scene label. Defaults to a label inferred from the selected cameras.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional inclusive lower bound on source frame ids.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional inclusive upper bound on source frame ids.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of processed frames after filtering.",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help="Minimum number of valid camera observations required per keypoint/frame.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=4,
        help="Maximum number of camera observations used per keypoint/frame.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Optional hard keypoint-score threshold before weighted triangulation.",
    )
    parser.add_argument(
        "--write-ply",
        action="store_true",
        help="Also write valid 3D keypoints for each frame as a simple ASCII PLY point cloud.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output scene directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report the triangulation plan without writing files.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def parse_camera_label_map(value: str) -> dict[str, str]:
    if not value:
        return {}
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--camera-label-map must decode to a JSON object.")
    return {str(key): str(val) for key, val in payload.items()}


def find_meta_paths(camera_dir: Path) -> list[Path]:
    if not camera_dir.is_dir():
        return []
    candidates = sorted(
        path
        for path in camera_dir.rglob("crop_inference_meta.jsonl")
        if path.is_file() and ZONE_IDENTIFIER_SUFFIX not in path.name
    )
    if not candidates:
        return []
    min_depth = min(len(path.relative_to(camera_dir).parts) for path in candidates)
    return [path for path in candidates if len(path.relative_to(camera_dir).parts) == min_depth]


def load_meta_index(meta_paths: Sequence[Path]) -> Dict[int, dict]:
    meta_index: Dict[int, dict] = {}
    for meta_path in meta_paths:
        with meta_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                frame = int(payload["frame"])
                if frame in meta_index:
                    existing_payload = {
                        key: value
                        for key, value in meta_index[frame].items()
                        if not key.startswith("_")
                    }
                    if existing_payload != payload:
                        raise ValueError(
                            f"Duplicate frame {frame} with conflicting crop metadata in {meta_path}."
                        )
                    continue

                record = dict(payload)
                record["_meta_path"] = str(meta_path)
                record["_meta_line"] = line_no
                meta_index[frame] = record
    return meta_index


def frame_set_from_paths(paths: Iterable[Path]) -> set[int]:
    frames = set()
    for path in paths:
        stem = path.stem
        if stem.isdigit():
            frames.add(int(stem))
    return frames


def list_real_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix.lower() and ZONE_IDENTIFIER_SUFFIX not in path.name
    )


def infer_scene_label(camera_names: Sequence[str]) -> str:
    if len(camera_names) == 1:
        camera_name = camera_names[0]
        trimmed = re.sub(r"([_-]?c\d*)$", "", camera_name).rstrip("_-")
        return trimmed or camera_name

    prefix = os.path.commonprefix(list(camera_names)).rstrip("_-")
    if prefix:
        prefix = re.sub(r"([_-]?c\d*)$", "", prefix).rstrip("_-")
    return prefix or "scene"


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_calibration_cameras(camera_path: Path) -> dict[str, dict[str, np.ndarray | tuple[int, int]]]:
    if camera_path.suffix.lower() == ".json":
        data = json.loads(camera_path.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        cameras: dict[str, dict[str, np.ndarray | tuple[int, int]]] = {}
        for frame in frames:
            label = str(frame["camera_label"])
            if label in cameras:
                continue
            K = np.array(
                [
                    [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                    [0.0, float(frame["fl_y"]), float(frame["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            c2w = c2w.copy()
            c2w[:3, 1:3] *= -1.0  # nerfstudio/OpenGL -> OpenCV
            w2c = np.linalg.inv(c2w)
            width = int(frame["w"])
            height = int(frame["h"])
            cameras[label] = {
                "K": K,
                "w2c": w2c,
                "image_size": (width, height),
            }
        if not cameras:
            raise ValueError(f"No camera frames found in {camera_path}.")
        return cameras

    from easyvolcap.utils.easy_utils import read_camera

    cams = read_camera(str(camera_path))
    cameras: dict[str, dict[str, np.ndarray | tuple[int, int]]] = {}
    for label, cam in cams.items():
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = np.asarray(cam["RT"], dtype=np.float64)
        cameras[str(label)] = {
            "K": np.asarray(cam["K"], dtype=np.float64).reshape(3, 3),
            "w2c": w2c,
            "image_size": (int(cam["W"]), int(cam["H"])),
        }
    if not cameras:
        raise ValueError(f"No cameras found in {camera_path}.")
    return cameras


def resolve_calibration_label(
    pose_camera: str,
    calibration_labels: Sequence[str],
    camera_label_map: dict[str, str],
) -> str:
    mapped = camera_label_map.get(pose_camera)
    if mapped is not None:
        return mapped
    if pose_camera in calibration_labels:
        return pose_camera

    target = normalize_label(pose_camera)
    matches = [label for label in calibration_labels if normalize_label(label) == target]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Could not match raw pose camera '{pose_camera}' to any calibration label. "
        f"Available calibration labels: {sorted(calibration_labels)}. "
        "Use --camera-label-map if the names differ."
    )


def discover_cameras(
    pose_root: Path,
    cropped_root: Path,
    requested_cameras: Sequence[str],
    calibration_labels: Sequence[str],
    camera_label_map: dict[str, str],
) -> list[CameraInput]:
    source_names = (
        list(requested_cameras)
        if requested_cameras
        else sorted(path.name for path in pose_root.iterdir() if path.is_dir())
    )
    if not source_names:
        raise ValueError(f"No cameras found under {pose_root}.")

    cameras: list[CameraInput] = []
    for source_name in source_names:
        pose_dir = pose_root / source_name
        cropped_dir = cropped_root / source_name
        meta_paths = find_meta_paths(cropped_dir)

        missing = []
        if not pose_dir.is_dir():
            missing.append(str(pose_dir))
        if not meta_paths:
            missing.append(f"{cropped_dir}/**/crop_inference_meta.jsonl")
        if missing:
            raise FileNotFoundError(
                f"Camera {source_name} is missing required inputs: {', '.join(missing)}"
            )

        calib_camera = resolve_calibration_label(source_name, calibration_labels, camera_label_map)
        cameras.append(
            CameraInput(
                pose_camera=source_name,
                calib_camera=calib_camera,
                pose_dir=pose_dir,
                meta_paths=meta_paths,
            )
        )
    return cameras


def select_frames(frames: Iterable[int], start_frame: int | None, end_frame: int | None, max_frames: int | None) -> list[int]:
    selected = sorted(set(frames))
    if start_frame is not None:
        selected = [frame for frame in selected if frame >= start_frame]
    if end_frame is not None:
        selected = [frame for frame in selected if frame <= end_frame]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def choose_primary_instance(instances: Sequence[dict]) -> int:
    if not instances:
        raise ValueError("Pose JSON has no instances.")
    scores = []
    for instance in instances:
        values = instance.get("keypoint_scores", [])
        scores.append(float(np.mean(values)) if values else -1.0)
    return int(np.argmax(scores))


def load_primary_pose(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    instances = data.get("instance_info", [])
    primary_index = choose_primary_instance(instances)
    instance = instances[primary_index]
    keypoints = np.asarray(instance["keypoints"], dtype=np.float64)
    scores = instance.get("keypoint_scores")
    if scores is None:
        scores_array = np.ones((len(keypoints),), dtype=np.float64)
    else:
        scores_array = np.asarray(scores, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"{path}: expected keypoints shape (k, 2), got {keypoints.shape}.")
    if scores_array.shape != (len(keypoints),):
        raise ValueError(f"{path}: expected keypoint_scores shape ({len(keypoints)},), got {scores_array.shape}.")
    return keypoints, scores_array, primary_index


def remap_keypoints_to_image_resolution(
    keypoints: np.ndarray,
    meta: dict,
    target_size: tuple[int, int],
) -> np.ndarray:
    pad_x, pad_y = meta["pad"]
    scale = float(meta["scale"])
    x1, y1, _, _ = meta["box_original"]
    full_w, full_h = meta["full_frame_size"]
    target_w, target_h = target_size

    remapped = keypoints.astype(np.float64).copy()
    remapped[:, 0] = ((remapped[:, 0] - pad_x) / scale + x1) * (target_w / full_w)
    remapped[:, 1] = ((remapped[:, 1] - pad_y) / scale + y1) * (target_h / full_h)
    remapped[:, 0] = np.clip(remapped[:, 0], 0.0, target_w - 1.0)
    remapped[:, 1] = np.clip(remapped[:, 1], 0.0, target_h - 1.0)
    return remapped


def project_one_point(kp3d: np.ndarray, Ks: np.ndarray, Ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(kp3d)) or np.any(kp3d <= INVALID / 2.0):
        return np.full((Ks.shape[0], 2), INVALID, dtype=np.float64), np.full((Ks.shape[0],), INVALID, dtype=np.float64)

    kp3d_h = np.append(kp3d, 1.0)
    P = Ks @ Ts[:, :3]
    kp2d_h = P @ kp3d_h
    depth = kp2d_h[:, 2]
    kp2d = kp2d_h[:, :2] / (depth[:, None] + 1e-9)
    return kp2d, depth


def finite_difference_jacobian(
    params: np.ndarray,
    residual_fn,
    eps: float = 1e-4,
) -> np.ndarray:
    base = residual_fn(params)
    jacobian = np.zeros((base.size, params.size), dtype=np.float64)
    for index in range(params.size):
        step = max(eps, abs(params[index]) * eps)
        delta = np.zeros_like(params)
        delta[index] = step
        plus = residual_fn(params + delta)
        minus = residual_fn(params - delta)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def refine_point_with_lm(
    initial: np.ndarray,
    residual_fn,
    max_iterations: int = 25,
    lambda_init: float = 1e-3,
    step_tolerance: float = 1e-8,
    cost_tolerance: float = 1e-10,
) -> np.ndarray:
    params = initial.astype(np.float64).copy()
    residuals = residual_fn(params)
    cost = 0.5 * float(residuals @ residuals)
    lambda_value = float(lambda_init)

    for _ in range(max_iterations):
        jacobian = finite_difference_jacobian(params, residual_fn)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        if np.linalg.norm(gradient, ord=np.inf) < step_tolerance:
            break

        diagonal = np.maximum(np.diag(hessian), 1.0)
        accepted = False
        local_lambda = lambda_value
        for _ in range(8):
            system = hessian + np.diag(diagonal * local_lambda)
            try:
                delta = -np.linalg.solve(system, gradient)
            except np.linalg.LinAlgError:
                local_lambda *= 10.0
                continue

            if np.linalg.norm(delta) < step_tolerance:
                accepted = True
                break

            candidate = params + delta
            candidate_residuals = residual_fn(candidate)
            candidate_cost = 0.5 * float(candidate_residuals @ candidate_residuals)
            if candidate_cost + cost_tolerance < cost:
                params = candidate
                residuals = candidate_residuals
                cost = candidate_cost
                lambda_value = max(local_lambda * 0.3, 1e-12)
                accepted = True
                break
            local_lambda *= 10.0

        if not accepted:
            break

    return params


def aggregate_3d_score(observation_scores: np.ndarray, reproj_error: float) -> float:
    if observation_scores.size == 0 or not math.isfinite(reproj_error):
        return 0.0
    mean_score = float(np.average(observation_scores, weights=np.clip(observation_scores, 1e-6, None)))
    reproj_factor = math.exp(-reproj_error / 20.0)
    return float(mean_score * reproj_factor)


def triangulate_one_point_weighted(
    Ks: np.ndarray,
    Ts: np.ndarray,
    kp2d: np.ndarray,
    kp2d_score: np.ndarray,
    min_views: int,
    max_views: int | None,
    score_thr: float,
) -> tuple[np.ndarray | None, float | None, int, float]:
    valid = np.isfinite(kp2d).all(axis=1) & np.isfinite(kp2d_score)
    valid &= kp2d_score > max(score_thr, 0.0)
    valid &= kp2d[:, 0] >= 0.0
    valid &= kp2d[:, 1] >= 0.0

    if not np.any(valid):
        return None, None, 0, 0.0

    indices = np.flatnonzero(valid)
    scores = kp2d_score[indices]
    if max_views is not None and len(indices) > max_views:
        order = np.argsort(-scores)[:max_views]
        indices = indices[order]
        scores = scores[order]

    if len(indices) < min_views:
        return None, None, int(len(indices)), 0.0

    Ks_sel = Ks[indices]
    Ts_sel = Ts[indices]
    kp2d_sel = kp2d[indices]
    score_sel = np.clip(scores.astype(np.float64), 1e-6, None)

    A = []
    weights = []
    for (u, v), K, T, s in zip(kp2d_sel, Ks_sel, Ts_sel, score_sel):
        P = K @ T[:3]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
        weights.extend([s, s])
    A = np.stack(A, axis=0)
    W = np.diag(np.sqrt(np.asarray(weights, dtype=np.float64)))
    Aw = W @ A
    _, _, Vt = np.linalg.svd(Aw)
    kp3d_h = Vt[-1]
    kp3d_lin = kp3d_h[:3] / (kp3d_h[3] + 1e-9)

    coord_w = np.repeat(np.sqrt(score_sel), 2)

    def residual(candidate: np.ndarray) -> np.ndarray:
        pred, _ = project_one_point(candidate, Ks_sel, Ts_sel)
        return (pred.reshape(-1) - kp2d_sel.reshape(-1)) * coord_w

    if scipy_least_squares is not None:
        result = scipy_least_squares(
            residual,
            kp3d_lin,
            method="trf",
            loss="huber",
            f_scale=1.0,
            max_nfev=50,
        )
        kp3d = result.x.astype(np.float64)
    else:
        kp3d = refine_point_with_lm(kp3d_lin, residual)

    kp2d_hat, depth = project_one_point(kp3d, Ks_sel, Ts_sel)
    positive_depth = depth > 1e-6
    if int(np.sum(positive_depth)) < min_views:
        return None, None, int(len(indices)), 0.0

    err_px = np.linalg.norm(kp2d_hat - kp2d_sel, axis=1)
    reproj = float(np.average(err_px, weights=score_sel))
    kp3d_score = aggregate_3d_score(score_sel, reproj)
    return kp3d, reproj, int(len(indices)), kp3d_score


def triangulate_keypoints_weighted(
    Ks: np.ndarray,
    Ts: np.ndarray,
    kp2d: np.ndarray,
    kp2d_score: np.ndarray,
    min_views: int,
    max_views: int | None,
    score_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k, _ = kp2d.shape
    if kp2d_score.shape != (n, k):
        raise ValueError(f"kp2d_score must have shape {(n, k)}, got {kp2d_score.shape}.")

    kp3d = np.full((k, 3), INVALID, dtype=np.float64)
    reproj = np.full((k,), INVALID, dtype=np.float64)
    num_views = np.zeros((k,), dtype=np.int32)
    kp3d_score = np.zeros((k,), dtype=np.float64)

    for index in range(k):
        point3d, point_reproj, point_views, point_score = triangulate_one_point_weighted(
            Ks=Ks,
            Ts=Ts,
            kp2d=kp2d[:, index],
            kp2d_score=kp2d_score[:, index],
            min_views=min_views,
            max_views=max_views,
            score_thr=score_thr,
        )
        num_views[index] = point_views
        kp3d_score[index] = point_score
        if point3d is not None:
            kp3d[index] = point3d
        if point_reproj is not None:
            reproj[index] = point_reproj

    return kp3d, kp3d_score, reproj, num_views


def write_ply(path: Path, points: np.ndarray) -> None:
    valid = np.isfinite(points).all(axis=1) & ~(points <= INVALID / 2.0).any(axis=1)
    points = points[valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]} {point[1]} {point[2]}\n")


def main() -> None:
    args = parse_args()
    if args.min_views < 2:
        raise ValueError("--min-views must be at least 2.")
    if args.max_views is not None and args.max_views < args.min_views:
        raise ValueError("--max-views must be >= --min-views.")

    camera_label_map = parse_camera_label_map(args.camera_label_map)
    calibration = load_calibration_cameras(args.camera_path)

    requested_cameras = [part.strip() for part in args.cameras.split(",") if part.strip()]
    cameras = discover_cameras(
        pose_root=args.pose_root,
        cropped_root=args.cropped_root,
        requested_cameras=requested_cameras,
        calibration_labels=list(calibration.keys()),
        camera_label_map=camera_label_map,
    )
    scene_label = args.scene_label or infer_scene_label([camera.pose_camera for camera in cameras])

    meta_by_camera: dict[str, dict[int, dict]] = {}
    pose_files_by_camera: dict[str, dict[int, Path]] = {}
    frame_width = None
    all_frames: set[int] = set()

    for camera in cameras:
        meta_index = load_meta_index(camera.meta_paths)
        pose_files = {
            int(path.stem): path
            for path in list_real_files(camera.pose_dir, ".json")
            if path.stem.isdigit()
        }
        if not pose_files:
            raise FileNotFoundError(f"{camera.pose_dir} has no pose JSON files.")
        meta_by_camera[camera.pose_camera] = meta_index
        pose_files_by_camera[camera.pose_camera] = pose_files
        all_frames.update(set(meta_index) & set(pose_files))
        if frame_width is None:
            frame_width = len(next(iter(pose_files.values())).stem)

    if frame_width is None:
        raise ValueError("Could not infer frame label width from pose JSON files.")

    selected_frames = select_frames(all_frames, args.start_frame, args.end_frame, args.max_frames)
    if not selected_frames:
        raise ValueError("No frames remain after applying frame filters.")

    summary = {
        "scene_label": scene_label,
        "camera_path": str(args.camera_path),
        "selected_pose_cameras": [camera.pose_camera for camera in cameras],
        "camera_mapping": {camera.pose_camera: camera.calib_camera for camera in cameras},
        "min_views": int(args.min_views),
        "max_views": None if args.max_views is None else int(args.max_views),
        "score_thr": float(args.score_thr),
        "selected_frames": selected_frames,
        "frames_written": [],
        "frames_skipped": [],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    scene_dir = args.out_root / scene_label
    ensure_clean_dir(scene_dir, args.overwrite)
    pcd_dir = scene_dir / "pcd" if args.write_ply else None

    for frame in selected_frames:
        observations = []
        available_pose_cameras = []
        for camera in cameras:
            pose_path = pose_files_by_camera[camera.pose_camera].get(frame)
            meta = meta_by_camera[camera.pose_camera].get(frame)
            if pose_path is None or meta is None:
                continue

            keypoints, scores, primary_index = load_primary_pose(pose_path)
            calib = calibration[camera.calib_camera]
            remapped_keypoints = remap_keypoints_to_image_resolution(
                keypoints=keypoints,
                meta=meta,
                target_size=calib["image_size"],
            )
            observations.append(
                {
                    "pose_camera": camera.pose_camera,
                    "calib_camera": camera.calib_camera,
                    "K": calib["K"],
                    "w2c": calib["w2c"],
                    "keypoints": remapped_keypoints,
                    "scores": scores,
                    "pose_path": str(pose_path),
                    "meta_path": meta["_meta_path"],
                    "primary_instance_index": int(primary_index),
                }
            )
            available_pose_cameras.append(camera.pose_camera)

        if len(observations) < args.min_views:
            summary["frames_skipped"].append(
                {
                    "frame": frame,
                    "reason": f"only {len(observations)} cameras available; need at least {args.min_views}",
                    "available_pose_cameras": available_pose_cameras,
                }
            )
            continue

        num_keypoints_set = {len(obs["keypoints"]) for obs in observations}
        if len(num_keypoints_set) != 1:
            raise ValueError(
                f"Frame {frame} has inconsistent keypoint counts across cameras: {sorted(num_keypoints_set)}."
            )

        Ks = np.stack([obs["K"] for obs in observations], axis=0).astype(np.float64)
        Ts = np.stack([obs["w2c"] for obs in observations], axis=0).astype(np.float64)
        kp2d = np.stack([obs["keypoints"] for obs in observations], axis=0).astype(np.float64)
        kp2d_score = np.stack([obs["scores"] for obs in observations], axis=0).astype(np.float64)

        kp3d, kp3d_score, reproj, num_views = triangulate_keypoints_weighted(
            Ks=Ks,
            Ts=Ts,
            kp2d=kp2d,
            kp2d_score=kp2d_score,
            min_views=args.min_views,
            max_views=args.max_views,
            score_thr=args.score_thr,
        )

        frame_name = f"{frame:0{frame_width}d}"
        out_json_path = scene_dir / f"{frame_name}.json"
        payload = {
            "scene_label": scene_label,
            "source_frame": int(frame),
            "frame_label": frame_name,
            "num_keypoints": int(kp3d.shape[0]),
            "invalid_value": float(INVALID),
            "camera_names": [obs["pose_camera"] for obs in observations],
            "camera_calibration_labels": [obs["calib_camera"] for obs in observations],
            "pose_paths": [obs["pose_path"] for obs in observations],
            "meta_paths": [obs["meta_path"] for obs in observations],
            "instance_info": [
                {
                    "keypoints": kp3d.tolist(),
                    "keypoint_scores": kp3d_score.tolist(),
                    "keypoint_reproj": reproj.tolist(),
                    "keypoint_num_views": num_views.tolist(),
                }
            ],
        }
        out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if pcd_dir is not None:
            write_ply(pcd_dir / f"{frame_name}.ply", kp3d)

        summary["frames_written"].append(
            {
                "frame": int(frame),
                "frame_label": frame_name,
                "num_cameras": len(observations),
                "camera_names": [obs["pose_camera"] for obs in observations],
                "num_valid_keypoints": int(np.sum(num_views >= args.min_views)),
            }
        )

    (scene_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote triangulated 3D poses to {scene_dir}")
    print(f"Frames written: {len(summary['frames_written'])}")
    print(f"Frames skipped: {len(summary['frames_skipped'])}")


if __name__ == "__main__":
    main()
