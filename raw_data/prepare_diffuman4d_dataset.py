#!/usr/bin/env python3
"""Convert local Sapiens outputs into a Diffuman4D-style scene package.

This script builds the data tree expected by Diffuman4D from:

- preferred full-frame RGB images in raw_data/results_cropped/<camera>/**/rgb_full
- fallback full-frame RGB images in raw_data/results/<camera>/fgr
- downsampled full-frame alpha masks in raw_data/results/<camera>/pha
- crop metadata in raw_data/results_cropped/<camera>/**/crop_inference_meta.jsonl
- optional full-resolution alpha masks in raw_data/results_cropped/<camera>/**/pha_full
- Sapiens pose predictions in raw_data/outputs/<model>/<camera>/*.json

It remaps keypoints from 1024x1024 crop coordinates back to full-frame image
coordinates, writes per-camera 2D pose files, and draws full-frame skeleton
maps. Camera calibration is not inferred from these files; provide an existing
Nerfstudio-style transforms.json if you have one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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

ZONE_IDENTIFIER_SUFFIX = ":Zone.Identifier"
COCO133_TO_GOLIATH308: Dict[int, int] = {}
HAS_GOLIATH_TO_COCO133_MAPPING = False


@dataclass(frozen=True)
class CameraPaths:
    source_name: str
    output_label: str
    image_dir: Path
    mask_dir: Path
    mask_fallback_dir: Path | None
    pose_dir: Path
    meta_paths: Sequence[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RAW_DATA_ROOT / "results",
        help="Root containing fallback full-frame images (<camera>/fgr) and optional masks (<camera>/pha).",
    )
    parser.add_argument(
        "--cropped-root",
        type=Path,
        default=RAW_DATA_ROOT / "results_cropped",
        help="Root containing crop metadata and optional full-resolution rgb_full/pha_full directories.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=RAW_DATA_ROOT / "outputs" / "0.6b",
        help="Root containing Sapiens pose JSON folders, one per camera.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "diffuman4d_prepared",
        help="Directory where the converted scene folder will be created.",
    )
    parser.add_argument(
        "--scene-label",
        type=str,
        default=None,
        help="Output scene label. Defaults to a label inferred from the selected cameras.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="",
        help="Comma-separated source camera directories to use. Defaults to all cameras found under --pose-root.",
    )
    parser.add_argument(
        "--frame-policy",
        choices=("longest_contiguous", "all_common"),
        default="longest_contiguous",
        help="How to choose synchronized frames shared by every selected camera.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional source-frame lower bound after synchronization filtering.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional source-frame upper bound after synchronization filtering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of selected frames after filtering.",
    )
    parser.add_argument(
        "--keep-frame-labels",
        action="store_true",
        help="Keep original source frame ids instead of renumbering frames from 000000.",
    )
    parser.add_argument(
        "--keep-camera-names",
        action="store_true",
        help="Use source camera directory names instead of 00, 01, ... as camera labels.",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("auto", "copy", "symlink", "hardlink"),
        default="auto",
        help="How to materialize images and masks in the output tree.",
    )
    parser.add_argument(
        "--binary-masks",
        action="store_true",
        help="Threshold alpha masks to binary 0/255 masks before writing them.",
    )
    parser.add_argument(
        "--target-pose-layout",
        choices=("auto", "source", "coco133"),
        default="auto",
        help=(
            "Target keypoint layout written to the output pose JSON. "
            "'auto' keeps 133-point COCO poses as-is, leaves 17-point poses unchanged, "
            "and keeps 308-point Goliath poses as source unless an explicit 308->133 mapping is available."
        ),
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold used when drawing skeleton maps.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=4,
        help="Keypoint radius in output skeleton maps.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness in output skeleton maps.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="Skeleton map background color.",
    )
    parser.add_argument(
        "--transforms-input",
        type=Path,
        default=None,
        help="Optional existing Nerfstudio-style transforms.json to filter and rewrite.",
    )
    parser.add_argument(
        "--sparse-pcd-input",
        type=Path,
        default=None,
        help="Optional sparse point cloud to copy into the scene folder as sparse_pcd.ply.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output scene directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report the conversion plan without writing files.",
    )
    return parser.parse_args()


def list_real_files(directory: Path, suffix: str) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == suffix.lower()
        and ZONE_IDENTIFIER_SUFFIX not in path.name
    )


def find_meta_paths(camera_dir: Path) -> List[Path]:
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
    return [
        path for path in candidates if len(path.relative_to(camera_dir).parts) == min_depth
    ]


def find_preferred_subdir(camera_dir: Path, subdir_name: str) -> Path | None:
    if not camera_dir.is_dir():
        return None

    direct = camera_dir / subdir_name
    if direct.is_dir():
        return direct

    candidates = sorted(
        path
        for path in camera_dir.rglob(subdir_name)
        if path.is_dir() and ZONE_IDENTIFIER_SUFFIX not in path.name
    )
    if not candidates:
        return None

    min_depth = min(len(path.relative_to(camera_dir).parts) for path in candidates)
    shallowest = [
        path for path in candidates if len(path.relative_to(camera_dir).parts) == min_depth
    ]
    return shallowest[0]


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


def discover_cameras(args: argparse.Namespace) -> List[CameraPaths]:
    if args.cameras:
        source_names = [part.strip() for part in args.cameras.split(",") if part.strip()]
    else:
        source_names = sorted(
            path.name for path in args.pose_root.iterdir() if path.is_dir()
        )

    if not source_names:
        raise ValueError(f"No cameras found under {args.pose_root}.")

    cameras: List[CameraPaths] = []
    for index, source_name in enumerate(source_names):
        results_dir = args.results_root / source_name
        cropped_dir = args.cropped_root / source_name
        pose_dir = args.pose_root / source_name

        rgb_full_dir = find_preferred_subdir(cropped_dir, "rgb_full")
        image_dir = rgb_full_dir if rgb_full_dir is not None else results_dir / "fgr"
        mask_dir = results_dir / "pha"
        mask_fallback_dir = find_preferred_subdir(cropped_dir, "pha_full")
        meta_paths = find_meta_paths(cropped_dir)

        missing = []
        if not image_dir.is_dir():
            missing.append(str(image_dir))
        if not mask_dir.is_dir() and mask_fallback_dir is None:
            missing.append(f"{mask_dir} or {cropped_dir}/**/pha_full")
        if not pose_dir.is_dir():
            missing.append(str(pose_dir))
        if not meta_paths:
            missing.append(f"{cropped_dir}/**/crop_inference_meta.jsonl")
        if missing:
            raise FileNotFoundError(
                f"Camera {source_name} is missing required inputs: {', '.join(missing)}"
            )

        output_label = source_name if args.keep_camera_names else f"{index:02d}"
        cameras.append(
            CameraPaths(
                source_name=source_name,
                output_label=output_label,
                image_dir=image_dir,
                mask_dir=mask_dir,
                mask_fallback_dir=mask_fallback_dir,
                pose_dir=pose_dir,
                meta_paths=meta_paths,
            )
        )

    return cameras


def frame_set_from_paths(paths: Iterable[Path]) -> set[int]:
    frames = set()
    for path in paths:
        stem = path.stem
        if stem.isdigit():
            frames.add(int(stem))
    return frames


def read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def resolve_mask_source(
    camera: CameraPaths,
    frame: int,
) -> Path | None:
    primary = camera.mask_dir / f"{frame:04d}.png"
    fallback = (
        camera.mask_fallback_dir / f"{frame:04d}.png"
        if camera.mask_fallback_dir is not None
        else None
    )

    prefer_full_res_mask = camera.image_dir.name == "rgb_full" and fallback is not None
    if prefer_full_res_mask:
        fallback_mask = read_mask(fallback)
        if fallback_mask is not None and bool(np.any(fallback_mask > 0)):
            return fallback

    primary_mask = read_mask(primary)
    if primary_mask is not None and bool(np.any(primary_mask > 0)):
        return primary

    if fallback is not None:
        fallback_mask = read_mask(fallback)
        if fallback_mask is not None and bool(np.any(fallback_mask > 0)):
            return fallback

    return None


def contiguous_segments(frames: Sequence[int]) -> List[List[int]]:
    if not frames:
        return []

    sorted_frames = sorted(frames)
    segments: List[List[int]] = [[sorted_frames[0]]]
    for frame in sorted_frames[1:]:
        if frame == segments[-1][-1] + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def infer_scene_label(camera_names: Sequence[str]) -> str:
    if len(camera_names) == 1:
        camera_name = camera_names[0]
        trimmed = re.sub(r"([_-]?c\d*)$", "", camera_name).rstrip("_-")
        return trimmed or camera_name

    prefix = os.path.commonprefix(list(camera_names)).rstrip("_-")
    if prefix:
        prefix = re.sub(r"([_-]?c\d*)$", "", prefix).rstrip("_-")
    return prefix or "scene"


def select_frames(
    args: argparse.Namespace,
    camera_frames: Dict[str, set[int]],
) -> List[int]:
    common = set.intersection(*(frames for frames in camera_frames.values()))
    if not common:
        raise ValueError("No synchronized frames are shared by every selected camera.")

    if args.start_frame is not None:
        common = {frame for frame in common if frame >= args.start_frame}
    if args.end_frame is not None:
        common = {frame for frame in common if frame <= args.end_frame}
    if not common:
        raise ValueError("Frame filters removed every synchronized frame.")

    if args.frame_policy == "all_common":
        selected = sorted(common)
    else:
        segments = contiguous_segments(sorted(common))
        selected = max(segments, key=len)

    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    if not selected:
        raise ValueError("No frames remain after applying the selection policy.")
    return selected


def remap_keypoints_to_full_frame(
    keypoints: np.ndarray,
    meta: dict,
) -> np.ndarray:
    pad_x, pad_y = meta["pad"]
    scale = float(meta["scale"])
    x1, y1, _, _ = meta["box_original"]
    full_w, full_h = meta["full_frame_size"]
    out_w, out_h = meta["bbox_source_size"]

    remapped = keypoints.astype(np.float32).copy()
    remapped[:, 0] = ((remapped[:, 0] - pad_x) / scale + x1) * (out_w / full_w)
    remapped[:, 1] = ((remapped[:, 1] - pad_y) / scale + y1) * (out_h / full_h)
    remapped[:, 0] = np.clip(remapped[:, 0], 0, out_w - 1)
    remapped[:, 1] = np.clip(remapped[:, 1], 0, out_h - 1)
    return remapped


def choose_primary_instance(instances: Sequence[dict]) -> int:
    if not instances:
        raise ValueError("Pose JSON has no instances.")
    scores = [
        float(np.mean(instance["keypoint_scores"])) if instance["keypoint_scores"] else -1.0
        for instance in instances
    ]
    return int(np.argmax(scores))


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


def resolve_target_num_keypoints(
    target_pose_layout: str,
    source_num_keypoints: int,
) -> int:
    if target_pose_layout == "source":
        return source_num_keypoints
    if target_pose_layout == "coco133":
        if source_num_keypoints == 308 and not HAS_GOLIATH_TO_COCO133_MAPPING:
            raise ValueError(
                "308-point Goliath poses cannot be converted to COCO-WholeBody 133 because "
                "the Sapiens dataset mapping is not available in this checkout. "
                "Use --target-pose-layout source to keep the original 308-point poses."
            )
        if source_num_keypoints not in (133, 308):
            raise ValueError(
                f"Cannot convert {source_num_keypoints} keypoints to COCO-WholeBody 133."
            )
        return 133
    if source_num_keypoints == 308 and not HAS_GOLIATH_TO_COCO133_MAPPING:
        return 308
    if source_num_keypoints in (133, 308):
        return 133
    return source_num_keypoints


def convert_pose_layout(
    keypoints: np.ndarray,
    scores: np.ndarray,
    target_num_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_num_keypoints = len(keypoints)
    if source_num_keypoints == target_num_keypoints:
        return keypoints.astype(np.float32).copy(), scores.astype(np.float32).copy()

    if source_num_keypoints == 308 and target_num_keypoints == 133:
        if not HAS_GOLIATH_TO_COCO133_MAPPING:
            raise ValueError(
                "308-point Goliath poses cannot be converted to COCO-WholeBody 133 because "
                "the Sapiens dataset mapping is not available in this checkout."
            )
        converted_keypoints = np.full((133, 2), -1.0, dtype=np.float32)
        converted_scores = np.zeros((133,), dtype=np.float32)
        for coco_index, goliath_index in COCO133_TO_GOLIATH308.items():
            converted_keypoints[coco_index] = keypoints[goliath_index]
            converted_scores[coco_index] = scores[goliath_index]
        return converted_keypoints, converted_scores

    raise ValueError(
        f"Unsupported pose conversion from {source_num_keypoints} to {target_num_keypoints} keypoints."
    )


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_asset(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode in ("auto", "symlink"):
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(src, dst)


def write_mask(
    src: Path,
    dst: Path,
    binary: bool,
    mode: str,
    target_size: tuple[int, int] | None = None,
) -> None:
    if not binary and target_size is None:
        materialize_asset(src, dst, mode)
        return

    mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {src}")
    if target_size is not None and (mask.shape[1], mask.shape[0]) != target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    out_mask = binary_mask if binary else mask
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), out_mask):
        raise IOError(f"Failed to write mask image: {dst}")


def parse_frame_id_from_transform(frame: dict) -> int | None:
    if "frame" in frame and isinstance(frame["frame"], int):
        return int(frame["frame"])
    if "frame_index" in frame and isinstance(frame["frame_index"], int):
        return int(frame["frame_index"])
    file_path = str(frame.get("file_path", ""))
    stem = Path(file_path).stem
    return int(stem) if stem.isdigit() else None


def detect_transform_camera(frame: dict) -> str | None:
    camera_label = frame.get("camera_label")
    if camera_label is not None:
        return str(camera_label)

    file_path = str(frame.get("file_path", ""))
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return None


def rescale_transform_intrinsics(
    frame: dict,
    target_width: int,
    target_height: int,
) -> dict:
    source_width = frame.get("w")
    source_height = frame.get("h")
    if not isinstance(source_width, (int, float)) or not isinstance(source_height, (int, float)):
        return dict(frame)

    source_width = float(source_width)
    source_height = float(source_height)
    if source_width <= 0.0 or source_height <= 0.0:
        return dict(frame)

    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height

    updated = dict(frame)
    for key in ("fl_x", "cx"):
        if key in updated:
            updated[key] = float(updated[key]) * scale_x
    for key in ("fl_y", "cy"):
        if key in updated:
            updated[key] = float(updated[key]) * scale_y
    updated["w"] = int(target_width)
    updated["h"] = int(target_height)
    return updated


def rewrite_transforms(
    transforms_path: Path,
    cameras: Sequence[CameraPaths],
    selected_frames: Sequence[int],
    output_frame_names: Dict[int, str],
    output_image_size_by_camera: Dict[str, tuple[int, int]],
) -> dict:
    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(
            f"{transforms_path} does not look like a Nerfstudio transforms.json with a frames list."
        )

    source_to_output = {camera.source_name: camera.output_label for camera in cameras}
    output_to_source = {camera.output_label: camera.source_name for camera in cameras}
    output_labels = {camera.output_label for camera in cameras}
    selected_set = set(selected_frames)

    rewritten = []
    for frame in data["frames"]:
        frame_camera = detect_transform_camera(frame)
        if frame_camera is None:
            continue
        if frame_camera in output_labels:
            output_label = frame_camera
            source_name = output_to_source[output_label]
        elif frame_camera in source_to_output:
            source_name = frame_camera
            output_label = source_to_output[frame_camera]
        else:
            continue

        source_frame = parse_frame_id_from_transform(frame)
        if source_frame is None or source_frame not in selected_set:
            continue

        target_height, target_width = output_image_size_by_camera[source_name]
        new_frame = rescale_transform_intrinsics(frame, target_width, target_height)
        new_frame["camera_label"] = output_label
        new_frame["file_path"] = f"images/{output_label}/{output_frame_names[source_frame]}.png"
        new_frame["frame"] = source_frame
        rewritten.append(new_frame)

    if not rewritten:
        raise ValueError(
            f"No matching frames from {transforms_path} matched the selected cameras and frames."
        )

    def frame_sort_key(frame: dict) -> tuple[str, int]:
        return str(frame["camera_label"]), int(frame["frame"])

    data["frames"] = sorted(rewritten, key=frame_sort_key)
    return data


def write_missing_transforms_note(scene_dir: Path) -> None:
    note = (
        "No calibrated transforms.json was provided.\n"
        "Diffuman4D custom-data preprocessing and multi-view inference require real camera intrinsics/extrinsics.\n"
        "Add a Nerfstudio-style transforms.json with per-frame camera_label entries, then rerun this converter with --transforms-input.\n"
    )
    (scene_dir / "MISSING_TRANSFORMS.txt").write_text(note, encoding="utf-8")


def summarize_camera_frames(
    cameras: Sequence[CameraPaths],
    frame_sets: Dict[str, set[int]],
) -> dict:
    summary = {}
    for camera in cameras:
        frames = sorted(frame_sets[camera.source_name])
        summary[camera.output_label] = {
            "source_name": camera.source_name,
            "num_frames": len(frames),
            "min_frame": frames[0] if frames else None,
            "max_frame": frames[-1] if frames else None,
            "segments": [
                {"start": segment[0], "end": segment[-1], "length": len(segment)}
                for segment in contiguous_segments(frames)
            ],
            "meta_paths": [str(path) for path in camera.meta_paths],
            "image_dir": str(camera.image_dir),
            "mask_dir": str(camera.mask_dir),
            "mask_fallback_dir": str(camera.mask_fallback_dir) if camera.mask_fallback_dir else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    cameras = discover_cameras(args)

    per_camera_frames: Dict[str, set[int]] = {}
    meta_by_camera: Dict[str, Dict[int, dict]] = {}
    image_size_by_camera: Dict[str, tuple[int, int]] = {}
    mask_source_by_camera: Dict[str, Dict[int, Path]] = {}

    for camera in cameras:
        meta_index = load_meta_index(camera.meta_paths)
        pose_frames = frame_set_from_paths(list_real_files(camera.pose_dir, ".json"))
        image_frames = frame_set_from_paths(list_real_files(camera.image_dir, ".png"))
        primary_mask_frames = frame_set_from_paths(list_real_files(camera.mask_dir, ".png"))
        fallback_mask_frames = (
            frame_set_from_paths(list_real_files(camera.mask_fallback_dir, ".png"))
            if camera.mask_fallback_dir is not None
            else set()
        )

        candidate_frames = set(meta_index) & pose_frames & image_frames & (
            primary_mask_frames | fallback_mask_frames
        )
        resolved_masks: Dict[int, Path] = {}
        usable_frames = set()
        for frame in candidate_frames:
            mask_source = resolve_mask_source(camera, frame)
            if mask_source is None:
                continue
            usable_frames.add(frame)
            resolved_masks[frame] = mask_source
        if not usable_frames:
            raise ValueError(f"Camera {camera.source_name} has no aligned frames.")

        sample_image = cv2.imread(str(camera.image_dir / f"{min(usable_frames):04d}.png"))
        if sample_image is None:
            raise FileNotFoundError(
                f"Failed to read sample image for camera {camera.source_name}."
            )

        meta_by_camera[camera.source_name] = meta_index
        per_camera_frames[camera.source_name] = usable_frames
        image_size_by_camera[camera.source_name] = (sample_image.shape[0], sample_image.shape[1])
        mask_source_by_camera[camera.source_name] = resolved_masks

    selected_frames = select_frames(args, per_camera_frames)
    scene_label = args.scene_label or infer_scene_label([camera.source_name for camera in cameras])
    scene_dir = args.out_root / scene_label

    output_frame_names = {
        frame: f"{index:06d}" if not args.keep_frame_labels else f"{frame:06d}"
        for index, frame in enumerate(selected_frames)
    }

    if args.dry_run:
        print(f"scene_label={scene_label}")
        print(f"cameras={[camera.source_name for camera in cameras]}")
        print(f"camera_labels={[camera.output_label for camera in cameras]}")
        print(f"selected_frame_count={len(selected_frames)}")
        print(f"selected_source_frames={selected_frames[:10]}{'...' if len(selected_frames) > 10 else ''}")
        print(f"target_pose_layout={args.target_pose_layout}")
        print(
            json.dumps(
                summarize_camera_frames(cameras, per_camera_frames),
                indent=2,
            )
        )
        return

    ensure_clean_dir(scene_dir, args.overwrite)

    for subdir in ("images", "fmasks", "poses_2d", "skeletons"):
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        source_name = camera.source_name
        output_label = camera.output_label
        height, width = image_size_by_camera[source_name]

        for source_frame in selected_frames:
            meta = meta_by_camera[source_name][source_frame]
            frame_src_name = f"{source_frame:04d}"
            frame_out_name = output_frame_names[source_frame]

            src_image = camera.image_dir / f"{frame_src_name}.png"
            src_mask = mask_source_by_camera[source_name][source_frame]
            src_pose = camera.pose_dir / f"{frame_src_name}.json"

            dst_image = scene_dir / "images" / output_label / f"{frame_out_name}.png"
            dst_mask = scene_dir / "fmasks" / output_label / f"{frame_out_name}.png"
            dst_pose = scene_dir / "poses_2d" / output_label / f"{frame_out_name}.json"
            dst_skeleton = scene_dir / "skeletons" / output_label / f"{frame_out_name}.png"

            materialize_asset(src_image, dst_image, args.asset_mode)
            write_mask(
                src_mask,
                dst_mask,
                args.binary_masks,
                args.asset_mode,
                target_size=(width, height),
            )

            pose_data = json.loads(src_pose.read_text(encoding="utf-8"))
            remapped_instances = []
            for instance in pose_data.get("instance_info", []):
                keypoints = np.asarray(instance["keypoints"], dtype=np.float32)
                scores = np.asarray(instance["keypoint_scores"], dtype=np.float32)
                remapped = remap_keypoints_to_full_frame(keypoints, meta)
                target_num_keypoints = resolve_target_num_keypoints(
                    args.target_pose_layout,
                    len(remapped),
                )
                converted_keypoints, converted_scores = convert_pose_layout(
                    remapped,
                    scores,
                    target_num_keypoints,
                )
                remapped_instances.append(
                    {
                        "keypoints": converted_keypoints.tolist(),
                        "keypoint_scores": converted_scores.tolist(),
                        "source_num_keypoints": int(len(remapped)),
                        "output_num_keypoints": int(target_num_keypoints),
                    }
                )

            if not remapped_instances:
                raise ValueError(f"No pose instances found in {src_pose}.")

            primary_index = choose_primary_instance(remapped_instances)
            primary = remapped_instances[primary_index]
            keypoints = np.asarray(primary["keypoints"], dtype=np.float32)
            scores = np.asarray(primary["keypoint_scores"], dtype=np.float32)
            num_keypoints = len(primary["keypoints"])

            pose_output = {
                "camera_label": output_label,
                "frame_label": frame_out_name,
                "source_camera": source_name,
                "source_frame": source_frame,
                "source_pose_path": str(src_pose),
                "source_meta_path": meta["_meta_path"],
                "image_size": {"height": height, "width": width},
                "source_num_keypoints": int(
                    len(pose_data["instance_info"][primary_index]["keypoints"])
                ),
                "num_keypoints": num_keypoints,
                "target_pose_layout": (
                    "coco133" if num_keypoints == 133 else f"source_{num_keypoints}"
                ),
                "primary_instance_index": primary_index,
                "instance_info": remapped_instances,
            }
            dst_pose.parent.mkdir(parents=True, exist_ok=True)
            dst_pose.write_text(json.dumps(pose_output, indent=2), encoding="utf-8")

            skeleton = draw_skeleton_map(
                (height, width),
                keypoints,
                scores,
                num_keypoints,
                args.kpt_thr,
                args.radius,
                args.thickness,
                args.background,
            )
            dst_skeleton.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst_skeleton), skeleton):
                raise IOError(f"Failed to write skeleton image: {dst_skeleton}")

    if args.transforms_input is not None:
        rewritten = rewrite_transforms(
            args.transforms_input,
            cameras,
            selected_frames,
            output_frame_names,
            image_size_by_camera,
        )
        (scene_dir / "transforms.json").write_text(
            json.dumps(rewritten, indent=2),
            encoding="utf-8",
        )
    else:
        write_missing_transforms_note(scene_dir)

    if args.sparse_pcd_input is not None:
        shutil.copy2(args.sparse_pcd_input, scene_dir / "sparse_pcd.ply")

    manifest = {
        "scene_label": scene_label,
        "source_roots": {
            "results_root": str(args.results_root),
            "cropped_root": str(args.cropped_root),
            "pose_root": str(args.pose_root),
        },
        "cameras": [
            {
                "output_label": camera.output_label,
                "source_name": camera.source_name,
                "meta_paths": [str(path) for path in camera.meta_paths],
                "image_dir": str(camera.image_dir),
                "mask_dir": str(camera.mask_dir),
                "mask_fallback_dir": str(camera.mask_fallback_dir) if camera.mask_fallback_dir else None,
            }
            for camera in cameras
        ],
        "selected_source_frames": selected_frames,
        "output_frame_names": {
            str(source_frame): output_frame_names[source_frame]
            for source_frame in selected_frames
        },
        "frame_policy": args.frame_policy,
        "target_pose_layout": args.target_pose_layout,
        "coco133_from_goliath308_mapping_size": len(COCO133_TO_GOLIATH308),
        "frame_coverage": summarize_camera_frames(cameras, per_camera_frames),
        "transforms_input": str(args.transforms_input) if args.transforms_input else None,
        "sparse_pcd_input": str(args.sparse_pcd_input) if args.sparse_pcd_input else None,
        "notes": [
            "images are sourced from raw_data/results_cropped/<camera>/**/rgb_full when available, otherwise from raw_data/results/<camera>/fgr",
            "full-resolution pha_full masks are preferred when using rgb_full images; otherwise raw_data/results/<camera>/pha is used when non-empty, with masks resized to the emitted image resolution if needed",
            "transforms are rescaled to the emitted image resolution when the input calibration was solved at a different source resolution",
            "poses_2d and skeletons are remapped from crop-space Sapiens outputs",
            "308-point Goliath poses stay in source layout unless an explicit 308->133 mapping is available",
        ],
    }
    (scene_dir / "scene_metadata.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote Diffuman4D scene package to {scene_dir}")


if __name__ == "__main__":
    main()
