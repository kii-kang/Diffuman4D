#!/usr/bin/env python3
"""Prepare per-performer temporal-window Diffuman4D dataset scaffolds.

This script converts performer metadata plus calibrated cameras into a
windowed dataset layout that is useful for later cropping / copying image,
mask, and skeleton assets into per-window canonical scenes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def build_window_transforms_json(cameras: list[WindowCamera]) -> dict[str, Any]:
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
        performer_dir.mkdir(parents=True, exist_ok=True)

        performer_windows_summary = []
        for window_index, window_frames in enumerate(windows):
            window_id = f"window_{window_index:03d}"
            window_dir = performer_dir / window_id
            images_dir = window_dir / "images"
            fmasks_dir = window_dir / "fmasks"
            skeletons_dir = window_dir / "skeletons"
            cameras_dir = window_dir / "cameras"
            surfs_dir = window_dir / "surfs"
            for base_dir in [images_dir, fmasks_dir, skeletons_dir, cameras_dir, surfs_dir]:
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

            camera_mappings = []
            window_cameras = []
            for camera_index, source_label in enumerate(ordered_source_camera_labels):
                spa_label = f"{camera_index:02d}"
                camera_mappings.append(
                    {"spa_label": spa_label, "source_camera_label": source_label, "camera_index": camera_index}
                )
                camera = build_local_camera(static_frames[source_label], world_center, scale, spa_label)
                window_cameras.append(camera)
                (images_dir / spa_label).mkdir(parents=True, exist_ok=True)
                (fmasks_dir / spa_label).mkdir(parents=True, exist_ok=True)
                (skeletons_dir / spa_label).mkdir(parents=True, exist_ok=True)

            transforms_payload = build_window_transforms_json(window_cameras)
            write_json(window_dir / "transforms.json", transforms_payload, overwrite=args.overwrite)
            write_json(cameras_dir / "transforms.json", transforms_payload, overwrite=args.overwrite)
            write_easyvolcap_intrinsics(cameras_dir / "intri.yml", window_cameras)
            write_easyvolcap_extrinsics(
                cameras_dir / "extri.yml",
                window_cameras,
                near=float(args.near),
                far=float(args.far),
                bounds_min=canonical_bounds_min,
                bounds_max=canonical_bounds_max,
            )
            write_json(
                cameras_dir / "scene_norm.json",
                {"center": [0.0, 0.0, 0.0], "scale": 1.0},
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
                    camera_mappings=camera_mappings,
                    window_frames=window_frames,
                    images_dir=images_dir,
                    fmasks_dir=fmasks_dir,
                    skeletons_dir=skeletons_dir,
                    overwrite=args.overwrite,
                )

            input_spa_indices = default_input_spa_indices(len(window_cameras))
            inference_overrides = {
                "data": "custom_png",
                "exp": "demo_4d_custom_png",
                "data.data_dir": str(window_dir.resolve()),
                "data.scene_label": ".",
                "data.camera_path_pat": "{data_dir}/{scene_label}/cameras",
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
                "min_views": min(2, len(window_cameras)) if window_cameras else 0,
            }
            inference_command = (
                "python inference.py "
                f"data={inference_overrides['data']} "
                f"exp={inference_overrides['exp']} "
                f"data.data_dir={window_dir.resolve()} "
                "data.scene_label=. "
                "'data.camera_path_pat=\"{data_dir}/{scene_label}/cameras\"' "
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
            window_metadata = {
                "performer_id": performer_id,
                "window_id": window_id,
                "camera_mappings": camera_mappings,
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
                "frame_records": frame_records,
                "expected_assets": {
                    "images_dir": relpath_str(images_dir, window_dir),
                    "fmasks_dir": relpath_str(fmasks_dir, window_dir),
                    "skeletons_dir": relpath_str(skeletons_dir, window_dir),
                    "cameras_dir": relpath_str(cameras_dir, window_dir),
                },
                "source_scene_dir": str(args.source_scene_dir.resolve()),
                "copied_asset_counts": copied_assets,
                "ready_for_inference": not args.skip_assets,
                "missing_assets": [] if not args.skip_assets else ["images", "fmasks", "skeletons"],
                "carve_visual_hull": carve_config,
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
                        "Populate images/, fmasks/, and skeletons/ before running inference.",
                        "The camera_path_pat override points Diffuman4D at cameras/ so scene_norm.json keeps local canonical coordinates unchanged.",
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
