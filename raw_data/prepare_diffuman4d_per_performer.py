#!/usr/bin/env python3
"""Prepare per-performer temporal-window Diffuman4D dataset scaffolds.

This script converts performer metadata plus calibrated cameras into a
windowed dataset layout that is useful for later cropping / copying image,
mask, and skeleton assets into per-window canonical scenes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAPIENS_DEMO_ROOT = ROOT / "scripts" / "preprocess" / "sapiens" / "lite" / "demo"
if str(SAPIENS_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAPIENS_DEMO_ROOT))

from classes_and_palettes import (  # noqa: E402
    COCO_KPTS_COLORS,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
)

INVALID = -1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("performer_metadata.json"),
        help="Performer metadata JSON from extract_performer_bboxes.py.",
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        default=Path("raw_data/transforms_room.json"),
        help="Global Nerfstudio-style camera calibration JSON.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path("diffuman4d_per_performer"),
        help="Output directory for per-performer temporal windows.",
    )
    parser.add_argument(
        "--camera-labels",
        type=str,
        default=None,
        help="Comma-separated ordered source camera labels to use, e.g. test2_c1,test2_c2,test2_c3,test2_c4.",
    )
    parser.add_argument(
        "--canonical-size",
        type=float,
        default=2.0,
        help="Largest performer-window bbox dimension after canonical scaling.",
    )
    parser.add_argument(
        "--source-scene-dir",
        type=Path,
        default=Path("diffuman4d_prepared/test2"),
        help="Prepared global Diffuman4D scene to subset for images/fmasks/skeletons.",
    )
    parser.add_argument(
        "--pose3d-scene-dir",
        type=Path,
        default=None,
        help="Triangulated 3D pose scene directory. Defaults to raw_data/triangulated_poses_3d/<scene>.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Temporal window size in frames.",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=0,
        help="Overlap between adjacent windows in frames.",
    )
    parser.add_argument(
        "--min-window-frames",
        type=int,
        default=8,
        help="Drop trailing windows shorter than this many frames.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.1,
        help="Near plane written into EasyVolcap extrinsics.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=100.0,
        help="Far plane written into EasyVolcap extrinsics.",
    )
    parser.add_argument(
        "--virtual-ring-views",
        type=int,
        default=0,
        help="Number of virtual target cameras to place on a ring around each window. Use 0 to disable.",
    )
    parser.add_argument(
        "--virtual-ring-radius-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the average real-camera ring radius when placing virtual cameras.",
    )
    parser.add_argument(
        "--virtual-ring-lookat-z",
        type=float,
        default=0.0,
        help="Local-space Z coordinate that virtual cameras look at.",
    )
    parser.add_argument(
        "--virtual-ring-focal-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the average real-camera focal length for virtual cameras.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold used when drawing virtual skeleton maps.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=4,
        help="Keypoint radius in virtual skeleton maps.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness in virtual skeleton maps.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="Skeleton map background color for generated virtual views.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Only write metadata/cameras/configs and do not copy images/fmasks/skeletons.",
    )
    return parser.parse_args()


@dataclass
class WindowCamera:
    source_label: str
    spa_label: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    c2w_opengl: np.ndarray
    rotation_opencv: np.ndarray
    tvec_opencv: np.ndarray
    rvec_opencv: np.ndarray


def pose_spec(num_keypoints: int):
    if num_keypoints == 17:
        return COCO_KPTS_COLORS, COCO_SKELETON_INFO
    if num_keypoints == 133:
        return COCO_WHOLEBODY_KPTS_COLORS, COCO_WHOLEBODY_SKELETON_INFO
    if num_keypoints == 308:
        return GOLIATH_KPTS_COLORS, GOLIATH_SKELETON_INFO
    raise ValueError(f"Unsupported num_keypoints={num_keypoints}.")


def draw_skeleton_map(
    image_shape: tuple[int, int],
    keypoints: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    kpt_thr: float,
    radius: int,
    thickness: int,
    background: str,
) -> np.ndarray:
    height, width = image_shape
    bg_value = 0 if background == "black" else 255
    canvas = np.full((height, width, 3), bg_value, dtype=np.uint8)
    kpt_colors, skeleton_info = pose_spec(num_keypoints)

    for _, link_info in skeleton_info.items():
        pt1_idx, pt2_idx = link_info["link"]
        if scores[pt1_idx] <= kpt_thr or scores[pt2_idx] <= kpt_thr:
            continue
        color = tuple(int(channel) for channel in link_info["color"][::-1])
        pt1 = tuple(int(v) for v in keypoints[pt1_idx])
        pt2 = tuple(int(v) for v in keypoints[pt2_idx])
        cv2.line(canvas, pt1, pt2, color, thickness=thickness)

    for idx, point in enumerate(keypoints):
        if scores[idx] <= kpt_thr:
            continue
        color = kpt_colors[idx]
        if color is None:
            continue
        color_bgr = tuple(int(channel) for channel in color[::-1])
        center = tuple(int(v) for v in point)
        cv2.circle(canvas, center, int(radius), color_bgr, -1)

    return canvas


def project_one_point(kp3d: np.ndarray, Ks: np.ndarray, Ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project one 3D point into multiple cameras."""
    if (kp3d == INVALID).any():
        invalid_2d = np.full((Ks.shape[0], 2), INVALID, dtype=np.float64)
        invalid_depth = np.full((Ks.shape[0],), INVALID, dtype=np.float64)
        return invalid_2d, invalid_depth

    kp3d_h = np.append(kp3d, 1.0)
    projection = Ks @ Ts[:, :3]
    kp2d_h = projection @ kp3d_h
    depth = kp2d_h[:, 2]
    kp2d = kp2d_h[:, :2] / (depth[:, None] + 1e-9)
    return kp2d, depth


def project_points(
    kp3d: np.ndarray,
    Ks: np.ndarray,
    Ts: np.ndarray,
    kp3d_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    projections = [project_one_point(point, Ks, Ts) for point in kp3d]
    kp2d = np.asarray([proj[0] for proj in projections], dtype=np.float64).transpose(1, 0, 2)
    kp2d_depth = np.asarray([proj[1] for proj in projections], dtype=np.float64).transpose(1, 0)

    if kp3d_score is None:
        return kp2d, kp2d_depth, None

    kp2d_score = kp3d_score[None, :].repeat(kp2d.shape[0], axis=0)
    if kp3d.shape[0] >= 91:
        nose, left_eye, right_eye = kp3d[:3]
        eye_mid = (left_eye + right_eye) / 2.0
        face_normal = np.cross(right_eye - left_eye, nose - eye_mid)
        face_norm = np.linalg.norm(face_normal)
        if face_norm > 1e-8:
            face_normal /= face_norm
            cam_normal = Ts[:, 2, :3]
            face_cam_cos = -np.dot(cam_normal, face_normal)
            face_cam_score = face_cam_cos * 0.5 + 0.5
            kp2d_score[:, :3] *= face_cam_score[:, None]
            kp2d_score[:, 23:91] *= face_cam_score[:, None]

    return kp2d, kp2d_depth, kp2d_score


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_float(value: float) -> str:
    return f"{float(value):.10f}"


def format_opencv_matrix(name: str, matrix: np.ndarray) -> str:
    rows, cols = matrix.shape
    flat = ", ".join(format_float(value) for value in matrix.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        f"  rows: {rows}\n"
        f"  cols: {cols}\n"
        f"  dt: d\n"
        f"  data: [{flat}]\n"
    )


def write_easyvolcap_intrinsics(path: Path, cameras: list[WindowCamera]) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for camera in cameras:
        lines.append(f'  - "{camera.spa_label}"')
    for camera in cameras:
        label = camera.spa_label
        lines.append(format_opencv_matrix(f"K_{label}", camera.K))
        lines.append(f"H_{label}: {format_float(camera.height)}")
        lines.append(f"W_{label}: {format_float(camera.width)}")
        lines.append(format_opencv_matrix(f"D_{label}", camera.D.reshape(5, 1)))
        lines.append(format_opencv_matrix(f"ccm_{label}", np.eye(3, dtype=np.float64)))
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_easyvolcap_extrinsics(
    path: Path,
    cameras: list[WindowCamera],
    near: float,
    far: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for camera in cameras:
        lines.append(f'  - "{camera.spa_label}"')
    bounds_array = np.asarray([bounds_min, bounds_max], dtype=np.float64).reshape(2, 3)
    for camera in cameras:
        label = camera.spa_label
        lines.append(format_opencv_matrix(f"R_{label}", camera.rvec_opencv.reshape(3, 1)))
        lines.append(format_opencv_matrix(f"Rot_{label}", camera.rotation_opencv))
        lines.append(format_opencv_matrix(f"T_{label}", camera.tvec_opencv.reshape(3, 1)))
        lines.append(f"t_{label}: {format_float(0.0)}")
        lines.append(f"n_{label}: {format_float(near)}")
        lines.append(f"f_{label}: {format_float(far)}")
        lines.append(format_opencv_matrix(f"bounds_{label}", bounds_array))
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_scene_norm(cameras: list[WindowCamera]) -> dict[str, Any]:
    centers = np.stack([camera.c2w_opengl[:3, 3] for camera in cameras], axis=0)
    min_bound = centers.min(axis=0)
    max_bound = centers.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    scale_denom = float(np.linalg.norm(max_bound - min_bound))
    scale = 1.0 / scale_denom if scale_denom > 1e-8 else 1.0
    return {
        "center": center.astype(np.float64).tolist(),
        "scale": float(scale),
        "formula": "normalized_camera_center = (camera_center - center) * scale",
    }


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def build_lookat_opengl_c2w(
    position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    if world_up is None:
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    back = normalize_vector(position - target)
    right = np.cross(world_up, back)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(world_up, back)
    right = normalize_vector(right)
    up = normalize_vector(np.cross(back, right))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = back
    c2w[:3, 3] = position
    return c2w


def choose_ring_offset(real_angles: np.ndarray, num_views: int) -> float:
    if num_views <= 0:
        return 0.0
    step = (2.0 * math.pi) / num_views
    best_offset = 0.0
    best_margin = -1.0
    for fraction in [0.0, 0.25, 0.5, 0.75]:
        offset = step * fraction
        ring_angles = offset + np.arange(num_views, dtype=np.float64) * step
        margin = min(
            min(abs(math.atan2(math.sin(angle - real), math.cos(angle - real))) for real in real_angles)
            for angle in ring_angles
        )
        if margin > best_margin:
            best_margin = margin
            best_offset = offset
    return best_offset


def load_performer_metadata(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    performers = {
        performer_id: data
        for performer_id, data in raw.items()
        if isinstance(data, dict) and "frame_bboxes" in data
    }
    if not performers:
        raise ValueError(f"No performer entries with frame_bboxes found in {path}")
    return performers


def parse_camera_labels(camera_labels_arg: str | None, transforms_frames: list[dict[str, Any]]) -> list[str]:
    if camera_labels_arg:
        labels = [label.strip() for label in camera_labels_arg.split(",") if label.strip()]
        if not labels:
            raise ValueError("--camera-labels was provided but no valid labels were parsed.")
        return labels

    seen = []
    for frame in transforms_frames:
        label = str(frame["camera_label"])
        if label not in seen:
            seen.append(label)
    if not seen:
        raise ValueError("No camera labels found in transforms JSON.")
    return seen


def load_static_camera_frames(transforms_path: Path, ordered_labels: list[str]) -> dict[str, dict[str, Any]]:
    with transforms_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} does not contain a non-empty frames list.")

    by_label: dict[str, dict[str, Any]] = {}
    for frame in frames:
        label = str(frame["camera_label"])
        if label in ordered_labels and label not in by_label:
            by_label[label] = frame

    missing = [label for label in ordered_labels if label not in by_label]
    if missing:
        raise ValueError(f"Missing camera labels in {transforms_path}: {missing}")

    return by_label


def sliding_windows(frames: list[int], window_size: int, overlap: int, min_window_frames: int) -> list[list[int]]:
    if window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if overlap < 0:
        raise ValueError("--window-overlap must be >= 0")
    if overlap >= window_size:
        raise ValueError("--window-overlap must be smaller than --window-size")

    stride = window_size - overlap
    windows: list[list[int]] = []
    for start in range(0, len(frames), stride):
        window = frames[start : start + window_size]
        if len(window) < min_window_frames:
            break
        windows.append(window)
    return windows


def bbox_from_metadata(frame_bboxes: dict[str, dict[str, Any]], frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for frame in frames:
        bbox = frame_bboxes[str(frame)]
        mins.append(np.asarray(bbox["min"], dtype=np.float64))
        maxs.append(np.asarray(bbox["max"], dtype=np.float64))
    mins_array = np.stack(mins, axis=0)
    maxs_array = np.stack(maxs, axis=0)
    return mins_array.min(axis=0), maxs_array.max(axis=0)


def build_local_camera(frame: dict[str, Any], center_world: np.ndarray, scale: float, spa_label: str) -> WindowCamera:
    c2w_world = np.asarray(frame["transform_matrix"], dtype=np.float64)
    local_center = (c2w_world[:3, 3] - center_world) * scale

    c2w_local = np.eye(4, dtype=np.float64)
    c2w_local[:3, :3] = c2w_world[:3, :3]
    c2w_local[:3, 3] = local_center

    c2w_cv = c2w_local.copy()
    c2w_cv[:3, 1:3] *= -1.0
    w2c_cv = np.linalg.inv(c2w_cv)
    rotation = w2c_cv[:3, :3]
    tvec = w2c_cv[:3, 3]
    rvec = cv2.Rodrigues(rotation)[0].reshape(3)

    return WindowCamera(
        source_label=str(frame["camera_label"]),
        spa_label=spa_label,
        width=int(frame["w"]),
        height=int(frame["h"]),
        K=np.asarray(
            [
                [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                [0.0, float(frame["fl_y"]), float(frame["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        D=np.asarray(
            [
                float(frame.get("k1", 0.0)),
                float(frame.get("k2", 0.0)),
                float(frame.get("p1", 0.0)),
                float(frame.get("p2", 0.0)),
                float(frame.get("k3", 0.0)),
            ],
            dtype=np.float64,
        ),
        c2w_opengl=c2w_local,
        rotation_opencv=rotation,
        tvec_opencv=tvec,
        rvec_opencv=rvec,
    )


def build_virtual_ring_cameras(
    real_cameras: list[WindowCamera],
    num_virtual_views: int,
    radius_scale: float,
    lookat_z: float,
    focal_scale: float,
) -> list[WindowCamera]:
    if num_virtual_views <= 0:
        return []

    positions = np.stack([camera.c2w_opengl[:3, 3] for camera in real_cameras], axis=0)
    horizontal_radius = np.linalg.norm(positions[:, :2], axis=1)
    radius = float(horizontal_radius.mean()) * float(radius_scale)
    radius = max(radius, 1e-3)
    camera_z = float(np.median(positions[:, 2]))
    target = np.asarray([0.0, 0.0, float(lookat_z)], dtype=np.float64)

    real_angles = np.arctan2(positions[:, 1], positions[:, 0])
    offset = choose_ring_offset(real_angles, num_virtual_views)

    fx = float(np.mean([camera.K[0, 0] for camera in real_cameras])) * float(focal_scale)
    fy = float(np.mean([camera.K[1, 1] for camera in real_cameras])) * float(focal_scale)
    cx = float(np.mean([camera.K[0, 2] for camera in real_cameras]))
    cy = float(np.mean([camera.K[1, 2] for camera in real_cameras]))
    width = int(round(np.mean([camera.width for camera in real_cameras])))
    height = int(round(np.mean([camera.height for camera in real_cameras])))
    distortion = np.mean(np.stack([camera.D for camera in real_cameras], axis=0), axis=0)

    virtual_cameras = []
    start_index = len(real_cameras)
    for view_index in range(num_virtual_views):
        angle = offset + (2.0 * math.pi * view_index / num_virtual_views)
        position = np.asarray(
            [radius * math.cos(angle), radius * math.sin(angle), camera_z],
            dtype=np.float64,
        )
        c2w_opengl = build_lookat_opengl_c2w(position, target)
        c2w_cv = c2w_opengl.copy()
        c2w_cv[:3, 1:3] *= -1.0
        w2c_cv = np.linalg.inv(c2w_cv)
        rotation = w2c_cv[:3, :3]
        tvec = w2c_cv[:3, 3]
        rvec = cv2.Rodrigues(rotation)[0].reshape(3)
        virtual_cameras.append(
            WindowCamera(
                source_label=f"virtual_ring_{view_index:03d}",
                spa_label=f"{start_index + view_index:02d}",
                width=width,
                height=height,
                K=np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
                D=np.asarray(distortion, dtype=np.float64),
                c2w_opengl=c2w_opengl,
                rotation_opencv=rotation,
                tvec_opencv=tvec,
                rvec_opencv=rvec,
            )
        )
    return virtual_cameras


def build_window_camera_transforms_json(cameras: list[WindowCamera]) -> dict[str, Any]:
    frames = []
    for camera in cameras:
        frames.append(
            {
                "camera_model": "OPENCV",
                "camera_label": camera.spa_label,
                "source_camera_label": camera.source_label,
                "w": camera.width,
                "h": camera.height,
                "fl_x": float(camera.K[0, 0]),
                "fl_y": float(camera.K[1, 1]),
                "cx": float(camera.K[0, 2]),
                "cy": float(camera.K[1, 2]),
                "k1": float(camera.D[0]),
                "k2": float(camera.D[1]),
                "p1": float(camera.D[2]),
                "p2": float(camera.D[3]),
                "k3": float(camera.D[4]),
                "transform_matrix": camera.c2w_opengl.tolist(),
                "file_path": f"images/{camera.spa_label}/000000.png",
                "frame": 0,
            }
        )
    return {"frames": frames}


def build_window_scene_transforms_json(cameras: list[WindowCamera], num_frames: int) -> dict[str, Any]:
    frames = []
    for camera in cameras:
        for frame_index in range(num_frames):
            tem_label = f"{frame_index:06d}"
            frames.append(
                {
                    "camera_model": "OPENCV",
                    "camera_label": camera.spa_label,
                    "source_camera_label": camera.source_label,
                    "w": camera.width,
                    "h": camera.height,
                    "fl_x": float(camera.K[0, 0]),
                    "fl_y": float(camera.K[1, 1]),
                    "cx": float(camera.K[0, 2]),
                    "cy": float(camera.K[1, 2]),
                    "k1": float(camera.D[0]),
                    "k2": float(camera.D[1]),
                    "p1": float(camera.D[2]),
                    "p2": float(camera.D[3]),
                    "k3": float(camera.D[4]),
                    "transform_matrix": camera.c2w_opengl.tolist(),
                    "file_path": f"images/{camera.spa_label}/{tem_label}.png",
                    "frame": frame_index,
                }
            )
    return {"frames": frames}


def relpath_str(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source asset: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        dst.unlink()
    ensure_parent(dst)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_input_spa_indices(num_cameras: int) -> list[int]:
    if num_cameras < 2:
        return []
    preferred_target = 1 if num_cameras > 1 else 0
    return [index for index in range(num_cameras) if index != preferred_target]


def localize_point(point_world: np.ndarray, center_world: np.ndarray, scale: float) -> np.ndarray:
    return (point_world - center_world) * scale


def resolve_pose3d_scene_dir(args: argparse.Namespace) -> Path:
    if args.pose3d_scene_dir is not None:
        return args.pose3d_scene_dir
    return RAW_DATA_ROOT / "triangulated_poses_3d" / args.source_scene_dir.name


def resolve_pose3d_frame_path(scene_dir: Path, source_frame: int) -> Path:
    candidates = [
        scene_dir / f"{source_frame:06d}.json",
        scene_dir / f"{source_frame:04d}.json",
        scene_dir / f"{source_frame}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find 3D pose JSON for source frame {source_frame} under {scene_dir}")


def localize_keypoints3d(keypoints_world: np.ndarray, center_world: np.ndarray, scale: float) -> np.ndarray:
    keypoints_local = np.asarray(keypoints_world, dtype=np.float64).copy()
    valid = np.isfinite(keypoints_local).all(axis=1) & ~(keypoints_local <= INVALID * 0.1).any(axis=1)
    keypoints_local[valid] = (keypoints_local[valid] - center_world) * scale
    keypoints_local[~valid] = INVALID
    return keypoints_local


def load_pose3d_cache(scene_dir: Path, source_frames: list[int]) -> dict[int, dict[str, Any]]:
    cache = {}
    for source_frame in source_frames:
        pose_path = resolve_pose3d_frame_path(scene_dir, source_frame)
        cache[source_frame] = json.loads(pose_path.read_text(encoding="utf-8"))
    return cache


def write_virtual_skeleton_assets(
    virtual_cameras: list[WindowCamera],
    window_frames: list[int],
    skeletons_dir: Path,
    poses2d_dir: Path,
    pose3d_cache: dict[int, dict[str, Any]],
    center_world: np.ndarray,
    scale: float,
    kpt_thr: float,
    radius: int,
    thickness: int,
    background: str,
    overwrite: bool,
) -> int:
    if not virtual_cameras:
        return 0

    generated = 0
    Ks = np.stack([camera.K for camera in virtual_cameras], axis=0)
    Ts = []
    for camera in virtual_cameras:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = camera.rotation_opencv
        w2c[:3, 3] = camera.tvec_opencv
        Ts.append(w2c)
    Ts = np.stack(Ts, axis=0)

    for local_frame_index, source_frame in enumerate(window_frames):
        pose = pose3d_cache[source_frame]
        instances = pose.get("instance_info", [])
        if not instances:
            continue
        instance = instances[0]
        keypoints_world = np.asarray(instance.get("keypoints", []), dtype=np.float64)
        scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float64)
        if keypoints_world.ndim != 2 or keypoints_world.shape[1] < 3:
            continue
        if scores.shape[0] != keypoints_world.shape[0]:
            scores = np.ones((keypoints_world.shape[0],), dtype=np.float64)
        keypoints_local = localize_keypoints3d(keypoints_world[:, :3], center_world, scale)
        kp2d, kp_depth, kp_score = project_points(keypoints_local, Ks, Ts, kp3d_score=scores)
        if kp_score is None:
            kp_score = np.broadcast_to(scores[None, :], kp_depth.shape)

        for camera_index, camera in enumerate(virtual_cameras):
            points_2d = kp2d[camera_index]
            points_depth = kp_depth[camera_index]
            points_score = kp_score[camera_index].copy()

            invalid = ~np.isfinite(points_2d).all(axis=1)
            invalid |= points_depth <= 1e-6
            invalid |= points_2d[:, 0] < 0
            invalid |= points_2d[:, 0] >= camera.width
            invalid |= points_2d[:, 1] < 0
            invalid |= points_2d[:, 1] >= camera.height
            points_score[invalid] = 0.0

            clipped = points_2d.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], 0, camera.width - 1)
            clipped[:, 1] = np.clip(clipped[:, 1], 0, camera.height - 1)

            pose_payload = {
                "instance_info": [
                    {
                        "keypoints": clipped.astype(np.float32).tolist(),
                        "keypoint_depths": points_depth.astype(np.float32).tolist(),
                        "keypoint_scores": points_score.astype(np.float32).tolist(),
                    }
                ]
            }
            pose_path = poses2d_dir / camera.spa_label / f"{local_frame_index:06d}.json"
            write_json(pose_path, pose_payload, overwrite=overwrite)

            skeleton = draw_skeleton_map(
                image_shape=(camera.height, camera.width),
                keypoints=clipped.astype(np.int32),
                scores=points_score.astype(np.float32),
                num_keypoints=keypoints_world.shape[0],
                kpt_thr=kpt_thr,
                radius=radius,
                thickness=thickness,
                background=background,
            )
            skeleton_path = skeletons_dir / camera.spa_label / f"{local_frame_index:06d}.png"
            ensure_parent(skeleton_path)
            if skeleton_path.exists() and not overwrite:
                raise FileExistsError(f"{skeleton_path} already exists. Use --overwrite to replace it.")
            cv2.imwrite(str(skeleton_path), skeleton)
            generated += 1

    return generated


def load_scene_frame_name_map(source_scene_dir: Path) -> dict[int, str]:
    scene_metadata_path = source_scene_dir / "scene_metadata.json"
    if scene_metadata_path.is_file():
        scene_metadata = json.loads(scene_metadata_path.read_text(encoding="utf-8"))
        output_frame_names = scene_metadata.get("output_frame_names", {})
        if output_frame_names:
            return {int(source_frame): str(output_name) for source_frame, output_name in output_frame_names.items()}

    image_root = source_scene_dir / "images"
    first_camera_dir = next((path for path in sorted(image_root.iterdir()) if path.is_dir()), None)
    if first_camera_dir is None:
        raise FileNotFoundError(f"Could not infer frame names because {image_root} has no camera directories.")
    stems = sorted(path.stem for path in first_camera_dir.iterdir() if path.is_file())
    return {index: stem for index, stem in enumerate(stems)}


def index_scene_assets(source_scene_dir: Path, asset_kind: str) -> dict[str, dict[str, Path]]:
    asset_root = source_scene_dir / asset_kind
    if not asset_root.is_dir():
        raise FileNotFoundError(f"Missing asset directory: {asset_root}")

    index: dict[str, dict[str, Path]] = {}
    for camera_dir in sorted(asset_root.iterdir()):
        if not camera_dir.is_dir():
            continue
        files_by_stem = {
            path.stem: path
            for path in sorted(camera_dir.iterdir())
            if path.is_file()
        }
        if not files_by_stem:
            raise FileNotFoundError(f"No files found under {camera_dir}")
        index[camera_dir.name] = files_by_stem

    if not index:
        raise FileNotFoundError(f"No camera asset directories found in {asset_root}")
    return index


def copy_window_assets(
    source_scene_dir: Path,
    camera_mappings: list[dict[str, Any]],
    window_frames: list[int],
    images_dir: Path,
    fmasks_dir: Path,
    skeletons_dir: Path,
    overwrite: bool,
) -> dict[str, int]:
    frame_name_map = load_scene_frame_name_map(source_scene_dir)
    image_index = index_scene_assets(source_scene_dir, "images")
    fmask_index = index_scene_assets(source_scene_dir, "fmasks")
    skeleton_index = index_scene_assets(source_scene_dir, "skeletons")

    copied_counts = {"images": 0, "fmasks": 0, "skeletons": 0}
    asset_dirs = {
        "images": images_dir,
        "fmasks": fmasks_dir,
        "skeletons": skeletons_dir,
    }
    asset_indices = {
        "images": image_index,
        "fmasks": fmask_index,
        "skeletons": skeleton_index,
    }

    for camera_mapping in camera_mappings:
        spa_label = str(camera_mapping["spa_label"])
        for local_frame_index, source_frame in enumerate(window_frames):
            source_frame_name = frame_name_map.get(int(source_frame))
            if source_frame_name is None:
                raise KeyError(
                    f"Source frame {source_frame} is missing from {source_scene_dir}/scene_metadata.json output_frame_names"
                )
            local_frame_name = f"{local_frame_index:06d}"

            for asset_kind in ["images", "fmasks", "skeletons"]:
                by_camera = asset_indices[asset_kind].get(spa_label)
                if by_camera is None:
                    raise KeyError(f"{asset_kind}: missing camera label {spa_label} under {source_scene_dir}")
                src = by_camera.get(source_frame_name)
                if src is None:
                    raise KeyError(
                        f"{asset_kind}: missing source frame {source_frame_name} for camera {spa_label} in {source_scene_dir}"
                    )
                dst = asset_dirs[asset_kind] / spa_label / f"{local_frame_name}{src.suffix.lower()}"
                copy_file(src, dst, overwrite=overwrite)
                copied_counts[asset_kind] += 1

    return copied_counts


def prepare_performer_windows(args: argparse.Namespace) -> None:
    performers = load_performer_metadata(args.metadata)

    with args.transforms.open("r", encoding="utf-8") as handle:
        transforms_data = json.load(handle)
    transforms_frames = transforms_data.get("frames", [])

    ordered_source_camera_labels = parse_camera_labels(args.camera_labels, transforms_frames)
    static_frames = load_static_camera_frames(args.transforms, ordered_source_camera_labels)

    output_root = args.output_base_dir
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = output_root / "configs"
    if configs_dir.exists() and args.overwrite:
        shutil.rmtree(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    summary_performers = []
    total_windows = 0

    for performer_id, performer_data in sorted(performers.items()):
        frame_bboxes = performer_data["frame_bboxes"]
        performer_frames = sorted(int(frame) for frame in frame_bboxes.keys())
        windows = sliding_windows(
            performer_frames,
            window_size=args.window_size,
            overlap=args.window_overlap,
            min_window_frames=args.min_window_frames,
        )

        performer_dir = output_root / performer_id
        if performer_dir.exists() and args.overwrite:
            shutil.rmtree(performer_dir)
        performer_dir.mkdir(parents=True, exist_ok=True)

        performer_windows_summary = []
        for window_index, window_frames in enumerate(windows):
            window_id = f"window_{window_index:03d}"
            window_dir = performer_dir / window_id
            images_dir = window_dir / "images"
            fmasks_dir = window_dir / "fmasks"
            skeletons_dir = window_dir / "skeletons"
            poses2d_dir = window_dir / "poses_2d"
            cameras_dir = window_dir / "cameras"
            surfs_dir = window_dir / "surfs"
            for base_dir in [images_dir, fmasks_dir, skeletons_dir, poses2d_dir, cameras_dir, surfs_dir]:
                base_dir.mkdir(parents=True, exist_ok=True)

            world_min, world_max = bbox_from_metadata(frame_bboxes, window_frames)
            world_center = (world_min + world_max) / 2.0
            world_size = world_max - world_min
            longest_dim = float(np.max(world_size))
            if longest_dim <= 0.0:
                raise ValueError(f"{performer_id}/{window_id} has a zero-size world bbox.")
            scale = float(args.canonical_size / longest_dim)

            local_min = localize_point(world_min, world_center, scale)
            local_max = localize_point(world_max, world_center, scale)
            canonical_half = float(args.canonical_size) / 2.0
            canonical_bounds_min = np.asarray([-canonical_half, -canonical_half, -canonical_half], dtype=np.float64)
            canonical_bounds_max = np.asarray([canonical_half, canonical_half, canonical_half], dtype=np.float64)
            carve_bounds = [
                -canonical_half,
                canonical_half,
                -canonical_half,
                canonical_half,
                -canonical_half,
                canonical_half,
            ]

            real_camera_mappings = []
            real_window_cameras = []
            for camera_index, source_label in enumerate(ordered_source_camera_labels):
                spa_label = f"{camera_index:02d}"
                real_camera_mappings.append(
                    {
                        "spa_label": spa_label,
                        "source_camera_label": source_label,
                        "camera_index": camera_index,
                        "camera_role": "real_input",
                    }
                )
                camera = build_local_camera(static_frames[source_label], world_center, scale, spa_label)
                real_window_cameras.append(camera)
                (images_dir / spa_label).mkdir(parents=True, exist_ok=True)
                (fmasks_dir / spa_label).mkdir(parents=True, exist_ok=True)

            virtual_window_cameras = build_virtual_ring_cameras(
                real_cameras=real_window_cameras,
                num_virtual_views=args.virtual_ring_views,
                radius_scale=args.virtual_ring_radius_scale,
                lookat_z=args.virtual_ring_lookat_z,
                focal_scale=args.virtual_ring_focal_scale,
            )
            virtual_camera_mappings = [
                {
                    "spa_label": camera.spa_label,
                    "source_camera_label": camera.source_label,
                    "camera_index": len(real_window_cameras) + virtual_index,
                    "camera_role": "virtual_target",
                }
                for virtual_index, camera in enumerate(virtual_window_cameras)
            ]

            camera_mappings = real_camera_mappings + virtual_camera_mappings
            window_cameras = real_window_cameras + virtual_window_cameras
            for camera in window_cameras:
                (skeletons_dir / camera.spa_label).mkdir(parents=True, exist_ok=True)
            for camera in virtual_window_cameras:
                (poses2d_dir / camera.spa_label).mkdir(parents=True, exist_ok=True)

            scene_transforms_payload = build_window_scene_transforms_json(window_cameras, len(window_frames))
            camera_transforms_payload = build_window_camera_transforms_json(window_cameras)
            write_json(window_dir / "transforms.json", scene_transforms_payload, overwrite=args.overwrite)
            write_json(cameras_dir / "transforms.json", camera_transforms_payload, overwrite=args.overwrite)
            write_easyvolcap_intrinsics(cameras_dir / "intri.yml", window_cameras)
            write_easyvolcap_extrinsics(
                cameras_dir / "extri.yml",
                window_cameras,
                near=float(args.near),
                far=float(args.far),
                bounds_min=canonical_bounds_min,
                bounds_max=canonical_bounds_max,
            )
            scene_norm_payload = compute_scene_norm(window_cameras)
            write_json(
                cameras_dir / "scene_norm.json",
                scene_norm_payload,
                overwrite=args.overwrite,
            )

            frame_records = []
            for local_frame_index, source_frame in enumerate(window_frames):
                frame_bbox = frame_bboxes[str(source_frame)]
                frame_world_min = np.asarray(frame_bbox["min"], dtype=np.float64)
                frame_world_max = np.asarray(frame_bbox["max"], dtype=np.float64)
                frame_world_center = np.asarray(frame_bbox["center"], dtype=np.float64)

                frame_local_min = localize_point(frame_world_min, world_center, scale)
                frame_local_max = localize_point(frame_world_max, world_center, scale)
                frame_local_center = localize_point(frame_world_center, world_center, scale)

                frame_records.append(
                    {
                        "local_frame_index": local_frame_index,
                        "local_tem_label": f"{local_frame_index:06d}",
                        "source_frame": int(source_frame),
                        "track_local_id": frame_bbox.get("local_id"),
                        "world_bbox": {
                            "min": frame_world_min.tolist(),
                            "max": frame_world_max.tolist(),
                            "center": frame_world_center.tolist(),
                            "size": (frame_world_max - frame_world_min).tolist(),
                        },
                        "local_bbox": {
                            "min": frame_local_min.tolist(),
                            "max": frame_local_max.tolist(),
                            "center": frame_local_center.tolist(),
                            "size": (frame_local_max - frame_local_min).tolist(),
                        },
                    }
                )

            copied_assets = {"images": 0, "fmasks": 0, "skeletons": 0}
            if not args.skip_assets:
                copied_assets = copy_window_assets(
                    source_scene_dir=args.source_scene_dir,
                    camera_mappings=real_camera_mappings,
                    window_frames=window_frames,
                    images_dir=images_dir,
                    fmasks_dir=fmasks_dir,
                    skeletons_dir=skeletons_dir,
                    overwrite=args.overwrite,
                )

            generated_virtual_assets = {"skeletons": 0, "poses2d": 0}
            if virtual_window_cameras and not args.skip_assets:
                pose3d_cache = load_pose3d_cache(resolve_pose3d_scene_dir(args), window_frames)
                generated_count = write_virtual_skeleton_assets(
                    virtual_cameras=virtual_window_cameras,
                    window_frames=window_frames,
                    skeletons_dir=skeletons_dir,
                    poses2d_dir=poses2d_dir,
                    pose3d_cache=pose3d_cache,
                    center_world=world_center,
                    scale=scale,
                    kpt_thr=float(args.kpt_thr),
                    radius=int(args.radius),
                    thickness=int(args.thickness),
                    background=str(args.background),
                    overwrite=args.overwrite,
                )
                generated_virtual_assets = {"skeletons": generated_count, "poses2d": generated_count}

            has_gt_target = not bool(virtual_window_cameras)
            input_spa_indices = (
                list(range(len(real_window_cameras)))
                if virtual_window_cameras
                else default_input_spa_indices(len(window_cameras))
            )
            inference_overrides = {
                "data": "custom_png",
                "exp": "demo_4d_custom_png",
                "data.data_dir": str(window_dir.resolve()),
                "data.scene_label": ".",
                "data.camera_path_pat": "{data_dir}/{scene_label}/cameras/transforms.json",
                "data.has_gt_target": has_gt_target,
                "sampler.spa_label_range": [0, len(window_cameras), 1],
                "sampler.input_spa_labels": input_spa_indices,
                "sampler.tem_label_range": [0, len(window_frames), 1],
                "sampler.window_size": 1,
                "sampler.sliding_stride": 1,
            }
            carve_config = {
                "fmasks_dir": relpath_str(fmasks_dir, output_root),
                "cameras_path": relpath_str(cameras_dir, output_root),
                "out_vhull_dir": relpath_str(surfs_dir, output_root),
                "frame_range": [0, 1, 1],
                "bounds": carve_bounds,
                "voxel_size": 0.05,
                "min_views": min(2, len(real_window_cameras)) if real_window_cameras else 0,
            }
            inference_command = (
                "python inference.py "
                f"data={inference_overrides['data']} "
                f"exp={inference_overrides['exp']} "
                f"data.data_dir={window_dir.resolve()} "
                "data.scene_label=. "
                "'data.camera_path_pat=\"{data_dir}/{scene_label}/cameras/transforms.json\"' "
                f"data.has_gt_target={str(has_gt_target).lower()} "
                f"'sampler.spa_label_range=[0,{len(window_cameras)},1]' "
                f"'sampler.input_spa_labels={input_spa_indices}' "
                f"'sampler.tem_label_range=[0,{len(window_frames)},1]' "
                "sampler.window_size=1 "
                "sampler.sliding_stride=1"
            )
            carve_command = (
                "python scripts/preprocess/carve_visual_hull.py "
                f"--fmasks_dir {fmasks_dir.resolve()} "
                f"--cameras_path {cameras_dir.resolve()} "
                f"--out_vhull_dir {surfs_dir.resolve()} "
                "--frame_range='(0, 1, 1)' "
                f"--bounds='({', '.join(format(value, '.6f') for value in carve_bounds)})' "
                "--voxel_size=0.05 "
                f"--min_views={carve_config['min_views']}"
            )
            ready_for_inference = not args.skip_assets and (
                not virtual_window_cameras or generated_virtual_assets["skeletons"] == len(virtual_window_cameras) * len(window_frames)
            )
            missing_assets = []
            if args.skip_assets:
                missing_assets = ["images", "fmasks", "skeletons"]
                if virtual_window_cameras:
                    missing_assets.append("poses_2d")
            elif virtual_window_cameras and generated_virtual_assets["skeletons"] < len(virtual_window_cameras) * len(window_frames):
                missing_assets = ["virtual_skeletons", "poses_2d"]

            window_metadata = {
                "performer_id": performer_id,
                "window_id": window_id,
                "camera_mappings": camera_mappings,
                "real_camera_count": len(real_window_cameras),
                "virtual_camera_count": len(virtual_window_cameras),
                "source_frame_range": [int(window_frames[0]), int(window_frames[-1])],
                "source_frames": [int(frame) for frame in window_frames],
                "frame_count": len(window_frames),
                "canonical_size": float(args.canonical_size),
                "world_bbox": {
                    "min": world_min.tolist(),
                    "max": world_max.tolist(),
                    "center": world_center.tolist(),
                    "size": world_size.tolist(),
                },
                "local_bbox": {
                    "min": local_min.tolist(),
                    "max": local_max.tolist(),
                    "center": [0.0, 0.0, 0.0],
                    "size": (local_max - local_min).tolist(),
                },
                "canonical_bounds": {
                    "min": canonical_bounds_min.tolist(),
                    "max": canonical_bounds_max.tolist(),
                    "carve_visual_hull_bounds": carve_bounds,
                },
                "world_to_local": {
                    "center_world": world_center.tolist(),
                    "uniform_scale": scale,
                    "formula": "local = (world - center_world) * uniform_scale",
                    "inverse_formula": "world = local / uniform_scale + center_world",
                },
                "camera_scene_norm": scene_norm_payload,
                "frame_records": frame_records,
                "expected_assets": {
                    "images_dir": relpath_str(images_dir, window_dir),
                    "fmasks_dir": relpath_str(fmasks_dir, window_dir),
                    "skeletons_dir": relpath_str(skeletons_dir, window_dir),
                    "poses2d_dir": relpath_str(poses2d_dir, window_dir),
                    "cameras_dir": relpath_str(cameras_dir, window_dir),
                    "real_input_spa_labels": [camera.spa_label for camera in real_window_cameras],
                    "virtual_target_spa_labels": [camera.spa_label for camera in virtual_window_cameras],
                },
                "source_scene_dir": str(args.source_scene_dir.resolve()),
                "copied_asset_counts": copied_assets,
                "generated_virtual_asset_counts": generated_virtual_assets,
                "ready_for_inference": ready_for_inference,
                "missing_assets": missing_assets,
                "carve_visual_hull": carve_config,
                "virtual_ring": {
                    "enabled": bool(virtual_window_cameras),
                    "views": len(virtual_window_cameras),
                    "radius_scale": float(args.virtual_ring_radius_scale),
                    "lookat_z": float(args.virtual_ring_lookat_z),
                    "focal_scale": float(args.virtual_ring_focal_scale),
                },
                "recommended_inference": inference_overrides,
            }
            write_json(window_dir / "metadata.json", window_metadata, overwrite=args.overwrite)
            write_json(
                configs_dir / f"{performer_id}_{window_id}_inference.json",
                {
                    "performer_id": performer_id,
                    "window_id": window_id,
                    "window_dir": relpath_str(window_dir, output_root),
                    "camera_mappings": camera_mappings,
                    "frame_count": len(window_frames),
                    "source_frames": [int(frame) for frame in window_frames],
                    "hydra_overrides": inference_overrides,
                    "command": inference_command,
                    "notes": [
                        "Populate real-camera images/, fmasks/, and skeletons/ before running inference.",
                        "Virtual target cameras rely on generated skeleton maps plus data.has_gt_target=false.",
                        "The camera_path_pat override points Diffuman4D at cameras/transforms.json and uses scene_norm.json from the same directory to normalize camera scale into the model's expected range.",
                    ],
                },
                overwrite=args.overwrite,
            )
            write_json(
                configs_dir / f"{performer_id}_{window_id}_vhull.json",
                {
                    "performer_id": performer_id,
                    "window_id": window_id,
                    "window_dir": relpath_str(window_dir, output_root),
                    "carve_visual_hull": carve_config,
                    "command": carve_command,
                },
                overwrite=args.overwrite,
            )

            performer_windows_summary.append(
                {
                    "window_id": window_id,
                    "path": relpath_str(window_dir, output_root),
                    "frame_count": len(window_frames),
                    "source_frame_range": [int(window_frames[0]), int(window_frames[-1])],
                    "real_camera_count": len(real_window_cameras),
                    "virtual_camera_count": len(virtual_window_cameras),
                    "total_camera_count": len(window_cameras),
                    "world_bbox_size": world_size.tolist(),
                    "local_bbox_size": (local_max - local_min).tolist(),
                    "uniform_scale": scale,
                }
            )
            total_windows += 1

        performer_summary = {
            "performer_id": performer_id,
            "path": relpath_str(performer_dir, output_root),
            "frame_count": int(performer_data["frame_count"]),
            "frame_range": performer_data["frame_range"],
            "global_bbox": performer_data["global_bbox"],
            "window_count": len(performer_windows_summary),
            "windows": performer_windows_summary,
        }
        write_json(performer_dir / "metadata.json", performer_summary, overwrite=args.overwrite)
        summary_performers.append(performer_summary)

    summary = {
        "metadata_path": str(args.metadata),
        "transforms_path": str(args.transforms),
        "output_base_dir": str(output_root),
        "camera_labels": ordered_source_camera_labels,
        "source_scene_dir": str(args.source_scene_dir),
        "camera_mappings": [
            {"spa_label": f"{index:02d}", "source_camera_label": label, "camera_index": index}
            for index, label in enumerate(ordered_source_camera_labels)
        ],
        "canonical_size": float(args.canonical_size),
        "window_size": int(args.window_size),
        "window_overlap": int(args.window_overlap),
        "min_window_frames": int(args.min_window_frames),
        "virtual_ring_views": int(args.virtual_ring_views),
        "virtual_ring_radius_scale": float(args.virtual_ring_radius_scale),
        "virtual_ring_lookat_z": float(args.virtual_ring_lookat_z),
        "virtual_ring_focal_scale": float(args.virtual_ring_focal_scale),
        "total_performers": len(summary_performers),
        "total_windows": total_windows,
        "performers": summary_performers,
    }
    write_json(output_root / "summary.json", summary, overwrite=args.overwrite)

    print(
        f"Prepared {total_windows} temporal windows across {len(summary_performers)} performers "
        f"in {output_root}"
    )


def main() -> None:
    args = parse_args()
    prepare_performer_windows(args)


if __name__ == "__main__":
    main()
