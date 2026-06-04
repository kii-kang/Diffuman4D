#!/usr/bin/env python3
"""Estimate static camera calibration from surveyed 3D room points and clicked 2D points.

Input:
- a JSON spec with named 3D world points
- per-camera clicked 2D image points for a single reference frame
- an intrinsic guess, either directly in pixels or via focal length / sensor size
- optional distortion coefficients

Output:
- a Nerfstudio-style transforms.json with one frame entry per source image
- optional EasyVolcap intri.yml / extri.yml
- a calibration report and per-camera reprojection visualizations

The JSON schema is intentionally simple. Start from `raw_data/room_calibration_template.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_NEAR = 0.2
DEFAULT_FAR = 20.0
DEFAULT_BOUNDS = [[-1_000_000.0, -1_000_000.0, -1_000_000.0], [1_000_000.0, 1_000_000.0, 1_000_000.0]]
DEFAULT_PNP_REPROJECTION_ERROR = 8.0
DEFAULT_REPROJECTION_WARNING_PX = 3.0
DEFAULT_CAMERA_MODEL = "OPENCV"
DEFAULT_DISTORTION = [0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass
class CameraCalibration:
    name: str
    width: int
    height: int
    image_path: Path
    frame_sources: list[Path]
    point_names: list[str]
    object_points: np.ndarray
    image_points: np.ndarray
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    camera_center: np.ndarray
    transform_matrix: np.ndarray
    reprojection_rmse: float
    reprojection_errors: list[float]
    inlier_indices: list[int]
    xy_error: float | None
    unconstrained_camera_center: np.ndarray | None = None
    unconstrained_reprojection_rmse: float | None = None
    unconstrained_xy_error: float | None = None
    fixed_xy_second_pass_used: bool = False
    focal_search_scale: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to the room-calibration JSON spec.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec and print the plan without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_output_path(base_dir: Path, value: str | None, default_name: str) -> Path:
    path = resolve_path(base_dir, value)
    if path is not None:
        return path
    return (base_dir / default_name).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read calibration image: {path}")
    height, width = image.shape[:2]
    return width, height


def resolve_image_size(
    camera_name: str,
    camera_spec: dict[str, Any],
    defaults: dict[str, Any],
    image_path: Path,
) -> tuple[int, int]:
    image_size = camera_spec.get("image_size", defaults.get("image_size"))
    if image_size is not None:
        if len(image_size) != 2:
            raise ValueError(f"{camera_name}: image_size must be [width, height].")
        return int(image_size[0]), int(image_size[1])
    return read_image_shape(image_path)


def build_camera_matrix(
    camera_name: str,
    camera_spec: dict[str, Any],
    defaults: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    intrinsics = camera_spec.get("intrinsics", defaults.get("intrinsics"))
    if intrinsics is not None:
        fx = float(intrinsics.get("fx", intrinsics.get("fl_x")))
        fy = float(intrinsics.get("fy", intrinsics.get("fl_y", fx)))
        cx = float(intrinsics.get("cx", width / 2.0))
        cy = float(intrinsics.get("cy", height / 2.0))
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    focal_length_mm = camera_spec.get("focal_length_mm", defaults.get("focal_length_mm"))
    sensor_width_mm = camera_spec.get("sensor_width_mm", defaults.get("sensor_width_mm"))
    sensor_height_mm = camera_spec.get("sensor_height_mm", defaults.get("sensor_height_mm"))
    if focal_length_mm is None or sensor_width_mm is None or sensor_height_mm is None:
        raise ValueError(
            f"{camera_name}: provide either intrinsics.fx/fy/cx/cy or focal_length_mm + sensor_width_mm + sensor_height_mm."
        )

    fx = float(focal_length_mm) * width / float(sensor_width_mm)
    fy = float(focal_length_mm) * height / float(sensor_height_mm)
    principal_point = camera_spec.get("principal_point", defaults.get("principal_point"))
    if principal_point is None:
        cx = width / 2.0
        cy = height / 2.0
    else:
        if len(principal_point) != 2:
            raise ValueError(f"{camera_name}: principal_point must be [cx, cy].")
        cx = float(principal_point[0])
        cy = float(principal_point[1])

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_camera_matrix_focal(K: np.ndarray, scale: float) -> np.ndarray:
    scaled = K.copy()
    scaled[0, 0] *= float(scale)
    scaled[1, 1] *= float(scale)
    return scaled


def build_distortion(
    camera_spec: dict[str, Any],
    defaults: dict[str, Any],
) -> np.ndarray:
    values = camera_spec.get("distortion", defaults.get("distortion", DEFAULT_DISTORTION))
    if len(values) not in (4, 5, 8):
        raise ValueError("distortion must have length 4, 5, or 8.")
    coeffs = np.zeros((8,), dtype=np.float64)
    coeffs[: len(values)] = np.asarray(values, dtype=np.float64)
    return coeffs.reshape(-1, 1)


def gather_correspondences(
    camera_name: str,
    camera_spec: dict[str, Any],
    world_points: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    image_points = camera_spec.get("image_points", {})
    point_names = camera_spec.get("point_names")
    if point_names is None:
        point_names = sorted(set(world_points) & set(image_points))

    if len(point_names) < 4:
        raise ValueError(
            f"{camera_name}: at least 4 named correspondences are required; found {len(point_names)}."
        )

    object_points = []
    clicked_points = []
    for point_name in point_names:
        if point_name not in world_points:
            raise KeyError(f"{camera_name}: world point '{point_name}' is missing.")
        if point_name not in image_points:
            raise KeyError(f"{camera_name}: image point '{point_name}' is missing.")

        world_xyz = world_points[point_name]
        image_xy = image_points[point_name]
        if len(world_xyz) != 3:
            raise ValueError(f"{camera_name}: world point '{point_name}' must be [x, y, z].")
        if len(image_xy) != 2:
            raise ValueError(f"{camera_name}: image point '{point_name}' must be [u, v].")

        object_points.append([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])])
        clicked_points.append([float(image_xy[0]), float(image_xy[1])])

    return (
        list(point_names),
        np.asarray(object_points, dtype=np.float64),
        np.asarray(clicked_points, dtype=np.float64),
    )


def get_pnp_flag(name: str | None) -> int | None:
    if name is None:
        return None
    normalized = name.strip().upper()
    if normalized == "AUTO":
        return None
    table = {
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "EPNP": cv2.SOLVEPNP_EPNP,
        "P3P": cv2.SOLVEPNP_P3P,
        "AP3P": cv2.SOLVEPNP_AP3P,
        "IPPE": getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE),
        "IPPE_SQUARE": getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        "SQPNP": getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE),
    }
    if normalized not in table:
        raise ValueError(f"Unknown pnp_flag: {name}")
    return table[normalized]


def is_coplanar(object_points: np.ndarray, eps: float = 1e-6) -> bool:
    if len(object_points) < 4:
        return True
    centered = object_points - np.mean(object_points, axis=0, keepdims=True)
    rank = np.linalg.matrix_rank(centered, tol=eps)
    return rank <= 2


def choose_auto_pnp_flag(object_points: np.ndarray) -> int:
    if len(object_points) == 4 and is_coplanar(object_points):
        return getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    if len(object_points) == 4:
        return cv2.SOLVEPNP_AP3P
    if is_coplanar(object_points):
        return getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    return getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)


def build_calibration_flags(refine_spec: dict[str, Any]) -> int:
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if refine_spec.get("fix_principal_point", True):
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if refine_spec.get("fix_aspect_ratio", True):
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if refine_spec.get("zero_tangent_dist", False):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if refine_spec.get("fix_k1", False):
        flags |= cv2.CALIB_FIX_K1
    if refine_spec.get("fix_k2", False):
        flags |= cv2.CALIB_FIX_K2
    if refine_spec.get("fix_k3", True):
        flags |= cv2.CALIB_FIX_K3
    if refine_spec.get("fix_k4", True):
        flags |= cv2.CALIB_FIX_K4
    if refine_spec.get("fix_k5", True):
        flags |= cv2.CALIB_FIX_K5
    if refine_spec.get("fix_k6", True):
        flags |= cv2.CALIB_FIX_K6
    return flags


def refine_intrinsics_with_room_points(
    object_points: np.ndarray,
    image_points: np.ndarray,
    width: int,
    height: int,
    K: np.ndarray,
    D: np.ndarray,
    refine_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flags = build_calibration_flags(refine_spec)
    rms, refined_K, refined_D, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=[object_points.astype(np.float32)],
        imagePoints=[image_points.astype(np.float32)],
        imageSize=(width, height),
        cameraMatrix=K.copy(),
        distCoeffs=D.copy(),
        flags=flags,
    )
    if not math.isfinite(float(rms)):
        raise ValueError("OpenCV calibrateCamera returned a non-finite RMS error.")
    return refined_K, refined_D, rvecs[0], tvecs[0]


def solve_camera_pose(
    camera_name: str,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pnp_flag: int | None,
    ransac_error_px: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if pnp_flag is None:
        pnp_flag = choose_auto_pnp_flag(object_points)

    use_direct_solve = len(object_points) <= 4 or pnp_flag in {
        getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE),
        getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        cv2.SOLVEPNP_P3P,
        cv2.SOLVEPNP_AP3P,
    }

    if use_direct_solve:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=K,
            distCoeffs=D,
            flags=pnp_flag,
        )
        inliers = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1) if success else None
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=K,
            distCoeffs=D,
            flags=pnp_flag,
            reprojectionError=float(ransac_error_px),
            iterationsCount=1000,
            confidence=0.999,
        )
    if not success:
        raise ValueError(f"{camera_name}: pose solve failed.")

    if inliers is None or len(inliers) < 4:
        inlier_indices = list(range(len(object_points)))
    else:
        inlier_indices = [int(idx) for idx in inliers.reshape(-1)]

    object_inliers = object_points[inlier_indices]
    image_inliers = image_points[inlier_indices]

    cv2.solvePnPRefineLM(
        objectPoints=object_inliers,
        imagePoints=image_inliers,
        cameraMatrix=K,
        distCoeffs=D,
        rvec=rvec,
        tvec=tvec,
    )
    return rvec, tvec, inlier_indices


def project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    return projected.reshape(-1, 2)


def compute_reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[float, list[float], np.ndarray]:
    projected = project_points(object_points, rvec, tvec, K, D)
    errors = np.linalg.norm(projected - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return rmse, [float(value) for value in errors], projected


def camera_center_from_extrinsics(rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return (-rotation.T @ tvec.reshape(3, 1)).reshape(3)


def tvec_from_camera_center(rotation: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    return (-rotation @ camera_center.reshape(3, 1)).reshape(3, 1)


def pose_from_fixed_xy(
    rvec: np.ndarray,
    known_xy: np.ndarray,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_center = np.array([float(known_xy[0]), float(known_xy[1]), float(z_value)], dtype=np.float64)
    tvec = tvec_from_camera_center(rotation, camera_center)
    return rotation, tvec, camera_center


def opencv_pose_to_nerfstudio_transform(rotation: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = tvec.reshape(3)
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1.0
    return c2w


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


def write_easyvolcap_intrinsics(
    path: Path,
    calibrations: list[CameraCalibration],
) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for calibration in calibrations:
        lines.append(f'  - "{calibration.name}"')
    for calibration in calibrations:
        label = calibration.name
        lines.append(format_opencv_matrix(f"K_{label}", calibration.K))
        lines.append(f"H_{label}: {format_float(calibration.height)}")
        lines.append(f"W_{label}: {format_float(calibration.width)}")
        lines.append(format_opencv_matrix(f"D_{label}", calibration.D[:5].reshape(5, 1)))
        lines.append(format_opencv_matrix(f"ccm_{label}", np.eye(3, dtype=np.float64)))
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_easyvolcap_extrinsics(
    path: Path,
    calibrations: list[CameraCalibration],
    near: float,
    far: float,
    bounds: list[list[float]],
) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for calibration in calibrations:
        lines.append(f'  - "{calibration.name}"')
    bounds_array = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    for calibration in calibrations:
        label = calibration.name
        lines.append(format_opencv_matrix(f"R_{label}", calibration.rvec.reshape(3, 1)))
        lines.append(format_opencv_matrix(f"Rot_{label}", calibration.rotation))
        lines.append(format_opencv_matrix(f"T_{label}", calibration.tvec.reshape(3, 1)))
        lines.append(f"t_{label}: {format_float(0.0)}")
        lines.append(f"n_{label}: {format_float(near)}")
        lines.append(f"f_{label}: {format_float(far)}")
        lines.append(format_opencv_matrix(f"bounds_{label}", bounds_array))
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_relpath(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def collect_frame_sources(
    camera_name: str,
    camera_spec: dict[str, Any],
    image_path: Path,
    spec_dir: Path,
) -> list[Path]:
    frame_labels = camera_spec.get("frame_labels")
    frame_glob = camera_spec.get("frame_glob")

    if frame_labels is not None and frame_glob is not None:
        raise ValueError(f"{camera_name}: use either frame_labels or frame_glob, not both.")

    if frame_glob is not None:
        pattern = Path(frame_glob)
        pattern_str = str(pattern if pattern.is_absolute() else (spec_dir / pattern))
        paths = sorted(Path(path).resolve() for path in glob(pattern_str) if Path(path).is_file())
        if not paths:
            raise FileNotFoundError(f"{camera_name}: frame_glob matched no files: {frame_glob}")
        return paths

    if frame_labels is not None:
        suffix = image_path.suffix
        return [(image_path.parent / f"{str(label)}{suffix}").resolve() for label in frame_labels]

    return [image_path.resolve()]


def draw_reprojection_overlay(
    image_path: Path,
    point_names: list[str],
    clicked_points: np.ndarray,
    projected_points: np.ndarray,
    errors: list[float],
    out_path: Path,
) -> None:
    if not image_path.is_file():
        return

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return

    for point_name, clicked, projected, error in zip(point_names, clicked_points, projected_points, errors):
        clicked_xy = tuple(int(round(value)) for value in clicked)
        projected_xy = tuple(int(round(value)) for value in projected)
        cv2.circle(image, clicked_xy, 8, (0, 0, 255), 2)
        cv2.circle(image, projected_xy, 5, (0, 255, 0), -1)
        cv2.line(image, clicked_xy, projected_xy, (0, 255, 255), 1)
        label = f"{point_name} ({error:.2f}px)"
        cv2.putText(
            image,
            label,
            (clicked_xy[0] + 8, clicked_xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (clicked_xy[0] + 8, clicked_xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    ensure_parent(out_path)
    cv2.imwrite(str(out_path), image)


def build_transforms_json(
    calibrations: list[CameraCalibration],
    transforms_path: Path,
    camera_model: str,
) -> dict[str, Any]:
    frames = []
    for calibration in calibrations:
        K = calibration.K
        D = calibration.D.reshape(-1)
        for frame_source in calibration.frame_sources:
            frame_stem = frame_source.stem
            entry = {
                "camera_model": camera_model,
                "camera_label": calibration.name,
                "w": int(calibration.width),
                "h": int(calibration.height),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "k1": float(D[0]),
                "k2": float(D[1]),
                "p1": float(D[2]),
                "p2": float(D[3]),
                "transform_matrix": calibration.transform_matrix.tolist(),
                "file_path": try_relpath(frame_source, transforms_path.parent),
            }
            if len(D) >= 5 and abs(float(D[4])) > 0.0:
                entry["k3"] = float(D[4])
            if frame_stem.isdigit():
                entry["frame"] = int(frame_stem)
            frames.append(entry)

    def sort_key(frame: dict[str, Any]) -> tuple[str, int, str]:
        frame_id = int(frame["frame"]) if "frame" in frame else -1
        return str(frame["camera_label"]), frame_id, str(frame["file_path"])

    return {"frames": sorted(frames, key=sort_key)}


def build_report(
    calibrations: list[CameraCalibration],
    warning_px: float,
) -> dict[str, Any]:
    report = {
        "warning_reprojection_threshold_px": float(warning_px),
        "cameras": {},
    }
    for calibration in calibrations:
        report["cameras"][calibration.name] = {
            "width": calibration.width,
            "height": calibration.height,
            "image_path": str(calibration.image_path),
            "num_correspondences": len(calibration.point_names),
            "inlier_indices": calibration.inlier_indices,
            "reprojection_rmse_px": calibration.reprojection_rmse,
            "reprojection_errors_px": calibration.reprojection_errors,
            "camera_center_world": calibration.camera_center.tolist(),
            "xy_error": calibration.xy_error,
            "fixed_xy_second_pass_used": calibration.fixed_xy_second_pass_used,
            "intrinsics": {
                "fx": float(calibration.K[0, 0]),
                "fy": float(calibration.K[1, 1]),
                "cx": float(calibration.K[0, 2]),
                "cy": float(calibration.K[1, 2]),
            },
            "distortion": calibration.D.reshape(-1)[:5].tolist(),
        }
        if calibration.unconstrained_camera_center is not None:
            report["cameras"][calibration.name]["unconstrained_camera_center_world"] = calibration.unconstrained_camera_center.tolist()
        if calibration.unconstrained_reprojection_rmse is not None:
            report["cameras"][calibration.name]["unconstrained_reprojection_rmse_px"] = calibration.unconstrained_reprojection_rmse
        if calibration.unconstrained_xy_error is not None:
            report["cameras"][calibration.name]["unconstrained_xy_error"] = calibration.unconstrained_xy_error
        if calibration.focal_search_scale is not None:
            report["cameras"][calibration.name]["focal_search_scale"] = calibration.focal_search_scale
    return report


def merge_configs(defaults: dict[str, Any] | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_known_xy(camera_name: str, camera_spec: dict[str, Any]) -> np.ndarray | None:
    known_xy = camera_spec.get("known_xy")
    if known_xy is None:
        return None
    if not isinstance(known_xy, (list, tuple)):
        raise ValueError(f"{camera_name}: known_xy must be [x, y] or null.")
    if len(known_xy) == 0:
        return None
    if len(known_xy) != 2:
        raise ValueError(f"{camera_name}: known_xy must be [x, y].")
    if known_xy[0] is None or known_xy[1] is None:
        return None

    known_xy_array = np.asarray(known_xy, dtype=np.float64)
    if known_xy_array.shape != (2,) or not np.all(np.isfinite(known_xy_array)):
        raise ValueError(f"{camera_name}: known_xy must contain two finite numbers.")
    return known_xy_array


def evaluate_pose_rmse_for_intrinsics(
    camera_name: str,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pnp_flag: int | None,
    ransac_error_px: float,
    known_xy: np.ndarray | None,
    second_pass_spec: dict[str, Any] | None,
) -> float:
    rvec, tvec, _ = solve_camera_pose(
        camera_name=camera_name,
        object_points=object_points,
        image_points=image_points,
        K=K,
        D=D,
        pnp_flag=pnp_flag,
        ransac_error_px=ransac_error_px,
    )
    if known_xy is not None and bool((second_pass_spec or {}).get("enabled", True)):
        rvec, tvec, _, _ = refine_pose_with_fixed_xy(
            object_points=object_points,
            image_points=image_points,
            K=K,
            D=D,
            known_xy=known_xy,
            initial_rvec=rvec,
            initial_tvec=tvec,
            second_pass_spec=second_pass_spec or {},
        )
    rmse, _, _ = compute_reprojection_errors(object_points, image_points, rvec, tvec, K, D)
    return rmse


def search_focal_scale(
    camera_name: str,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pnp_flag: int | None,
    ransac_error_px: float,
    known_xy: np.ndarray | None,
    second_pass_spec: dict[str, Any] | None,
    focal_search_spec: dict[str, Any],
) -> tuple[np.ndarray, float]:
    min_scale = float(focal_search_spec.get("min_scale", 0.8))
    max_scale = float(focal_search_spec.get("max_scale", 1.2))
    num_steps = int(focal_search_spec.get("num_steps", 9))
    passes = int(focal_search_spec.get("passes", 3))

    if not (min_scale > 0.0 and max_scale > 0.0 and max_scale >= min_scale):
        raise ValueError(f"{camera_name}: focal_search min_scale/max_scale must be positive and ordered.")
    if num_steps < 3:
        raise ValueError(f"{camera_name}: focal_search num_steps must be at least 3.")
    if passes < 1:
        raise ValueError(f"{camera_name}: focal_search passes must be at least 1.")

    low = min_scale
    high = max_scale
    best_scale = 1.0
    best_rmse = math.inf

    for _ in range(passes):
        scales = np.linspace(low, high, num_steps, dtype=np.float64)
        local_best_idx = -1
        local_best_rmse = math.inf

        for idx, scale in enumerate(scales):
            candidate_K = scale_camera_matrix_focal(K, float(scale))
            try:
                rmse = evaluate_pose_rmse_for_intrinsics(
                    camera_name=camera_name,
                    object_points=object_points,
                    image_points=image_points,
                    K=candidate_K,
                    D=D,
                    pnp_flag=pnp_flag,
                    ransac_error_px=ransac_error_px,
                    known_xy=known_xy,
                    second_pass_spec=second_pass_spec,
                )
            except Exception:
                continue

            if rmse < local_best_rmse:
                local_best_rmse = rmse
                local_best_idx = idx

        if local_best_idx < 0 or not math.isfinite(local_best_rmse):
            raise ValueError(f"{camera_name}: focal_search failed to find a valid focal candidate.")

        best_scale = float(scales[local_best_idx])
        best_rmse = local_best_rmse

        if local_best_idx == 0 or local_best_idx == len(scales) - 1:
            break

        left = float(scales[local_best_idx - 1])
        right = float(scales[local_best_idx + 1])
        if right - left <= 1e-9:
            break
        low = left
        high = right

    if not math.isfinite(best_rmse):
        raise ValueError(f"{camera_name}: focal_search ended with a non-finite reprojection error.")
    return scale_camera_matrix_focal(K, best_scale), best_scale


def fixed_xy_residuals(
    params: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    known_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rvec = params[:3].reshape(3, 1)
    z_value = float(params[3])
    rotation, tvec, camera_center = pose_from_fixed_xy(rvec, known_xy, z_value)
    projected = project_points(object_points, rvec, tvec, K, D)
    residuals = (projected - image_points).reshape(-1)
    return residuals, rotation, tvec, camera_center, projected


def finite_difference_jacobian(
    params: np.ndarray,
    residual_fn,
    rotation_eps: float,
    z_eps: float,
) -> np.ndarray:
    base_residuals = residual_fn(params)[0]
    jacobian = np.zeros((base_residuals.size, params.size), dtype=np.float64)
    for index in range(params.size):
        eps = float(rotation_eps if index < 3 else max(z_eps, abs(params[index]) * z_eps))
        delta = np.zeros_like(params)
        delta[index] = eps
        plus = residual_fn(params + delta)[0]
        minus = residual_fn(params - delta)[0]
        jacobian[:, index] = (plus - minus) / (2.0 * eps)
    return jacobian


def refine_pose_with_fixed_xy(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    known_xy: np.ndarray,
    initial_rvec: np.ndarray,
    initial_tvec: np.ndarray,
    second_pass_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial_rotation, _ = cv2.Rodrigues(initial_rvec.reshape(3, 1))
    initial_center = camera_center_from_extrinsics(initial_rotation, initial_tvec)
    initial_z = float(second_pass_spec.get("z_init", initial_center[2]))
    params = np.concatenate([initial_rvec.reshape(3), [initial_z]]).astype(np.float64)

    max_iterations = int(second_pass_spec.get("max_iterations", 50))
    lambda_value = float(second_pass_spec.get("lambda_init", 1e-3))
    rotation_eps = float(second_pass_spec.get("rotation_eps", 1e-5))
    z_eps = float(second_pass_spec.get("z_eps", 1e-4))
    step_tolerance = float(second_pass_spec.get("step_tolerance", 1e-8))
    cost_tolerance = float(second_pass_spec.get("cost_tolerance", 1e-10))
    damping_scale = float(second_pass_spec.get("damping_scale", 1.0))
    max_backtracks = int(second_pass_spec.get("max_backtracks", 8))

    residual_fn = lambda candidate: fixed_xy_residuals(candidate, object_points, image_points, K, D, known_xy)
    residuals, rotation, tvec, camera_center, _ = residual_fn(params)
    cost = 0.5 * float(residuals @ residuals)

    for _ in range(max_iterations):
        jacobian = finite_difference_jacobian(params, residual_fn, rotation_eps, z_eps)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        if np.linalg.norm(gradient, ord=np.inf) < step_tolerance:
            break

        diagonal = np.maximum(np.diag(hessian), 1.0)
        accepted = False
        local_lambda = lambda_value
        for _ in range(max_backtracks):
            system = hessian + np.diag(diagonal * local_lambda * damping_scale)
            try:
                delta = -np.linalg.solve(system, gradient)
            except np.linalg.LinAlgError:
                local_lambda *= 10.0
                continue

            if np.linalg.norm(delta) < step_tolerance:
                accepted = True
                break

            candidate = params + delta
            candidate_residuals, candidate_rotation, candidate_tvec, candidate_center, _ = residual_fn(candidate)
            candidate_cost = 0.5 * float(candidate_residuals @ candidate_residuals)
            if candidate_cost + cost_tolerance < cost:
                params = candidate
                residuals = candidate_residuals
                rotation = candidate_rotation
                tvec = candidate_tvec
                camera_center = candidate_center
                if abs(cost - candidate_cost) < cost_tolerance:
                    cost = candidate_cost
                    accepted = True
                    lambda_value = max(local_lambda * 0.3, 1e-12)
                    break
                cost = candidate_cost
                lambda_value = max(local_lambda * 0.3, 1e-12)
                accepted = True
                break

            local_lambda *= 10.0

        if not accepted:
            break

    final_rvec = params[:3].reshape(3, 1)
    final_rotation, final_tvec, final_center = pose_from_fixed_xy(final_rvec, known_xy, params[3])
    return final_rvec, final_tvec, final_rotation, final_center


def solve_camera_calibration(
    camera_name: str,
    camera_spec: dict[str, Any],
    defaults: dict[str, Any],
    world_points: dict[str, Any],
    spec_dir: Path,
) -> CameraCalibration:
    image_path = resolve_path(spec_dir, camera_spec.get("image_path"))
    if image_path is None:
        raise ValueError(f"{camera_name}: image_path is required.")
    image_size_override = camera_spec.get("image_size", defaults.get("image_size"))
    if not image_path.is_file() and image_size_override is None:
        raise FileNotFoundError(f"{camera_name}: image_path does not exist: {image_path}")

    width, height = resolve_image_size(camera_name, camera_spec, defaults, image_path)
    point_names, object_points, image_points = gather_correspondences(camera_name, camera_spec, world_points)
    K = build_camera_matrix(camera_name, camera_spec, defaults, width, height)
    D = build_distortion(camera_spec, defaults)
    D = D[:5].reshape(-1, 1)

    pnp_flag = get_pnp_flag(camera_spec.get("pnp_flag", defaults.get("pnp_flag")))
    ransac_error_px = float(camera_spec.get("pnp_ransac_error_px", defaults.get("pnp_ransac_error_px", DEFAULT_PNP_REPROJECTION_ERROR)))
    known_xy_array = parse_known_xy(camera_name, camera_spec)
    second_pass_spec = (
        merge_configs(defaults.get("lock_xy_second_pass"), camera_spec.get("lock_xy_second_pass"))
        if known_xy_array is not None
        else {}
    )
    focal_search_spec = merge_configs(defaults.get("focal_search"), camera_spec.get("focal_search"))
    focal_search_scale = None
    if bool(focal_search_spec.get("enabled", False)):
        K, focal_search_scale = search_focal_scale(
            camera_name=camera_name,
            object_points=object_points,
            image_points=image_points,
            K=K,
            D=D,
            pnp_flag=pnp_flag,
            ransac_error_px=ransac_error_px,
            known_xy=known_xy_array,
            second_pass_spec=second_pass_spec,
            focal_search_spec=focal_search_spec,
        )

    rvec, tvec, inlier_indices = solve_camera_pose(camera_name, object_points, image_points, K, D, pnp_flag, ransac_error_px)

    refine_spec = defaults.get("refine_intrinsics", {})
    camera_refine_spec = camera_spec.get("refine_intrinsics")
    if camera_refine_spec is not None:
        refine_spec = camera_refine_spec
    if refine_spec.get("enabled", False):
        object_inliers = object_points[inlier_indices]
        image_inliers = image_points[inlier_indices]
        K, D, rvec, tvec = refine_intrinsics_with_room_points(
            object_points=object_inliers,
            image_points=image_inliers,
            width=width,
            height=height,
            K=K,
            D=D,
            refine_spec=refine_spec,
        )
        D = D.reshape(-1, 1)

    rotation, _ = cv2.Rodrigues(rvec)
    camera_center = camera_center_from_extrinsics(rotation, tvec)
    xy_error = None
    unconstrained_camera_center = None
    unconstrained_reprojection_rmse = None
    unconstrained_xy_error = None
    fixed_xy_second_pass_used = False

    if known_xy_array is not None:
        unconstrained_camera_center = camera_center.copy()
        unconstrained_reprojection_rmse = compute_reprojection_errors(object_points, image_points, rvec, tvec, K, D)[0]
        unconstrained_xy_error = float(np.linalg.norm(camera_center[:2] - known_xy_array))

        second_pass_enabled = bool(second_pass_spec.get("enabled", True))
        if second_pass_enabled:
            rvec, tvec, rotation, camera_center = refine_pose_with_fixed_xy(
                object_points=object_points,
                image_points=image_points,
                K=K,
                D=D,
                known_xy=known_xy_array,
                initial_rvec=rvec,
                initial_tvec=tvec,
                second_pass_spec=second_pass_spec,
            )
            fixed_xy_second_pass_used = True
        xy_error = float(np.linalg.norm(camera_center[:2] - known_xy_array))

    transform_matrix = opencv_pose_to_nerfstudio_transform(rotation, tvec)
    rmse, errors, projected_points = compute_reprojection_errors(object_points, image_points, rvec, tvec, K, D)

    frame_sources = collect_frame_sources(camera_name, camera_spec, image_path, spec_dir)
    viz_path = resolve_path(spec_dir, camera_spec.get("viz_path"))
    if viz_path is not None:
        draw_reprojection_overlay(image_path, point_names, image_points, projected_points, errors, viz_path)

    return CameraCalibration(
        name=camera_name,
        width=width,
        height=height,
        image_path=image_path,
        frame_sources=frame_sources,
        point_names=point_names,
        object_points=object_points,
        image_points=image_points,
        K=K,
        D=D,
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        camera_center=camera_center,
        transform_matrix=transform_matrix,
        reprojection_rmse=rmse,
        reprojection_errors=errors,
        inlier_indices=inlier_indices,
        xy_error=xy_error,
        unconstrained_camera_center=unconstrained_camera_center,
        unconstrained_reprojection_rmse=unconstrained_reprojection_rmse,
        unconstrained_xy_error=unconstrained_xy_error,
        fixed_xy_second_pass_used=fixed_xy_second_pass_used,
        focal_search_scale=focal_search_scale,
    )


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    spec_dir = spec_path.parent
    spec = load_json(spec_path)

    defaults = spec.get("defaults", {})
    world_points = spec.get("world_points", {})
    cameras_spec = spec.get("cameras", {})
    if not world_points:
        raise ValueError("Spec must contain world_points.")
    if not cameras_spec:
        raise ValueError("Spec must contain cameras.")

    outputs = spec.get("outputs", {})
    transforms_path = resolve_output_path(spec_dir, outputs.get("transforms_path"), "transforms_room.json")
    intri_path = resolve_path(spec_dir, outputs.get("intri_path"))
    extri_path = resolve_path(spec_dir, outputs.get("extri_path"))
    report_path = resolve_output_path(spec_dir, outputs.get("report_path"), "calibration_report.json")
    viz_dir = resolve_output_path(spec_dir, outputs.get("viz_dir"), "calibration_viz")

    calibrations: list[CameraCalibration] = []
    for camera_name, camera_spec in cameras_spec.items():
        camera_spec = dict(camera_spec)
        if "viz_path" not in camera_spec:
            camera_spec["viz_path"] = str(viz_dir / f"{camera_name}.png")
        calibration = solve_camera_calibration(camera_name, camera_spec, defaults, world_points, spec_dir)
        calibrations.append(calibration)

    warning_px = float(defaults.get("reprojection_warning_px", DEFAULT_REPROJECTION_WARNING_PX))
    if args.dry_run:
        print(f"spec={spec_path}")
        print(f"num_cameras={len(calibrations)}")
        print(f"transforms_path={transforms_path}")
        for calibration in calibrations:
            warning = " WARNING" if calibration.reprojection_rmse > warning_px else ""
            xy_info = f", xy_error={calibration.xy_error:.4f}" if calibration.xy_error is not None else ""
            focal_info = (
                f", focal_scale={calibration.focal_search_scale:.4f}"
                if calibration.focal_search_scale is not None
                else ""
            )
            unconstrained_info = ""
            if calibration.unconstrained_reprojection_rmse is not None:
                unconstrained_info = (
                    f", unconstrained_rmse={calibration.unconstrained_reprojection_rmse:.4f}px"
                    f", unconstrained_xy_error={calibration.unconstrained_xy_error:.4f}"
                )
            print(
                f"{calibration.name}: rmse={calibration.reprojection_rmse:.4f}px, "
                f"center={calibration.camera_center.tolist()}{xy_info}{focal_info}{unconstrained_info}{warning}"
            )
        return

    transforms = build_transforms_json(
        calibrations=calibrations,
        transforms_path=transforms_path,
        camera_model=str(defaults.get("camera_model", DEFAULT_CAMERA_MODEL)),
    )
    ensure_parent(transforms_path)
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    if intri_path is not None:
        write_easyvolcap_intrinsics(intri_path, calibrations)
    if extri_path is not None:
        write_easyvolcap_extrinsics(
            extri_path,
            calibrations,
            near=float(defaults.get("near", DEFAULT_NEAR)),
            far=float(defaults.get("far", DEFAULT_FAR)),
            bounds=defaults.get("bounds", DEFAULT_BOUNDS),
        )

    report = build_report(calibrations, warning_px)
    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote transforms to {transforms_path}")
    if intri_path is not None:
        print(f"Wrote intrinsics to {intri_path}")
    if extri_path is not None:
        print(f"Wrote extrinsics to {extri_path}")
    print(f"Wrote report to {report_path}")

    for calibration in calibrations:
        warning = " WARNING" if calibration.reprojection_rmse > warning_px else ""
        xy_info = f", xy_error={calibration.xy_error:.4f}" if calibration.xy_error is not None else ""
        focal_info = (
            f", focal_scale={calibration.focal_search_scale:.4f}"
            if calibration.focal_search_scale is not None
            else ""
        )
        unconstrained_info = ""
        if calibration.unconstrained_reprojection_rmse is not None:
            unconstrained_info = (
                f", unconstrained_rmse={calibration.unconstrained_reprojection_rmse:.4f}px"
                f", unconstrained_xy_error={calibration.unconstrained_xy_error:.4f}"
            )
        print(
            f"{calibration.name}: rmse={calibration.reprojection_rmse:.4f}px, "
            f"center={calibration.camera_center.tolist()}{xy_info}{focal_info}{unconstrained_info}{warning}"
        )


if __name__ == "__main__":
    main()
