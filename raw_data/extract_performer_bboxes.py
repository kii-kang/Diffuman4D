#!/usr/bin/env python3
"""Extract per-performer 3D bounding boxes from triangulated multiview poses.

This script reads the JSON structure emitted by `triangulate_multiview_poses.py`,
tracks performer instances across frames by 3D centroid distance, and writes a
metadata file suitable for temporal windowing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INVALID_VALUE = -1e6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=Path("raw_data/triangulated_poses_3d/test2"),
        help="Directory containing triangulated per-frame pose JSONs.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("performer_metadata.json"),
        help="Where to save the extracted performer metadata JSON.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.5,
        help="Padding in world units added to each side of the performer bbox.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.5,
        help="Maximum centroid distance used for frame-to-frame performer matching.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=2,
        help="Allow a track to reconnect across this many missing / bad frames.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Drop tracked performers that appear in fewer than this many frames.",
    )
    parser.add_argument(
        "--min-keypoint-score",
        type=float,
        default=0.05,
        help="Ignore triangulated keypoints whose score is below this threshold when tracking and building bboxes.",
    )
    return parser.parse_args()


def iter_pose_files(poses_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in poses_dir.glob("*.json")
        if path.is_file() and path.name != "manifest.json"
    )


def invalid_cutoff(invalid_value: float) -> float:
    if invalid_value < 0:
        return invalid_value * 0.1
    return invalid_value * 10.0


def extract_valid_keypoints(
    instance_data: dict[str, np.ndarray | None],
    invalid_value: float,
    min_keypoint_score: float,
) -> np.ndarray:
    keypoints_3d = np.asarray(instance_data["keypoints"], dtype=np.float32)
    finite = np.isfinite(keypoints_3d).all(axis=1)
    cutoff = invalid_cutoff(invalid_value)
    if invalid_value < 0:
        sentinel = (keypoints_3d <= cutoff).any(axis=1)
    else:
        sentinel = (keypoints_3d >= cutoff).any(axis=1)
    valid_mask = finite & ~sentinel

    scores = instance_data.get("scores")
    if scores is not None:
        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores_array.shape[0] == keypoints_3d.shape[0]:
            valid_mask &= scores_array >= float(min_keypoint_score)
    return keypoints_3d[valid_mask]


def load_triangulated_poses(
    poses_dir: Path,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray | None]]], float]:
    """Load triangulated poses into {frame_idx: {local_instance_id: instance_data}}."""
    pose_files = iter_pose_files(poses_dir)
    if not pose_files:
        raise FileNotFoundError(f"No pose JSON files found in {poses_dir}")

    poses: dict[int, dict[str, dict[str, np.ndarray | None]]] = {}
    dataset_invalid_value = DEFAULT_INVALID_VALUE

    for json_file in pose_files:
        with json_file.open("r", encoding="utf-8") as handle:
            frame_data = json.load(handle)
        frame_idx = int(frame_data.get("source_frame", json_file.stem))

        invalid_value = float(frame_data.get("invalid_value", DEFAULT_INVALID_VALUE))
        dataset_invalid_value = invalid_value

        if "instance_info" in frame_data:
            frame_instances = {}
            for index, instance in enumerate(frame_data.get("instance_info", [])):
                local_id = str(instance.get("performer_id", f"instance_{index:03d}"))
                keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
                if keypoints.ndim != 2 or keypoints.shape[1] < 3:
                    continue
                scores = instance.get("keypoint_scores")
                scores_array = None if scores is None else np.asarray(scores, dtype=np.float32).reshape(-1)
                frame_instances[local_id] = {
                    "keypoints": keypoints[:, :3],
                    "scores": scores_array,
                }
        else:
            # Backward compatibility with a simpler {performer_id: keypoints} layout.
            frame_instances = {}
            for local_id, keypoints in frame_data.items():
                if not isinstance(keypoints, list):
                    continue
                keypoints_array = np.asarray(keypoints, dtype=np.float32)
                if keypoints_array.ndim != 2 or keypoints_array.shape[1] < 3:
                    continue
                frame_instances[str(local_id)] = {
                    "keypoints": keypoints_array[:, :3],
                    "scores": None,
                }

        poses[frame_idx] = frame_instances

    return poses, dataset_invalid_value


def performer_centers(
    frame_poses: dict[str, dict[str, np.ndarray | None]],
    invalid_value: float,
    min_keypoint_score: float,
) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    for local_id, instance_data in frame_poses.items():
        valid = extract_valid_keypoints(instance_data, invalid_value, min_keypoint_score)
        if len(valid) == 0:
            continue
        centers[local_id] = valid.mean(axis=0)
    return centers


def track_performers_across_frames(
    poses: dict[int, dict[str, dict[str, np.ndarray | None]]],
    invalid_value: float,
    distance_threshold: float = 1.5,
    max_frame_gap: int = 2,
    min_keypoint_score: float = 0.05,
) -> dict[str, list[tuple[int, str]]]:
    """Track performers by greedy nearest-neighbor matching across recent frames."""
    frame_indices = sorted(poses.keys())
    if not frame_indices:
        return {}

    global_performers: dict[str, list[tuple[int, str]]] = {}
    active_tracks: dict[str, tuple[int, np.ndarray]] = {}
    next_global_id = 0

    for frame_idx in frame_indices:
        current_centers = performer_centers(poses[frame_idx], invalid_value, min_keypoint_score)
        candidate_pairs: list[tuple[float, int, str, str]] = []

        for global_id, (last_frame_idx, last_center) in active_tracks.items():
            frame_gap = frame_idx - last_frame_idx
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                continue
            for curr_local_id, curr_center in current_centers.items():
                dist = float(np.linalg.norm(last_center - curr_center))
                if dist <= distance_threshold:
                    candidate_pairs.append((dist, frame_gap, global_id, curr_local_id))

        candidate_pairs.sort(key=lambda item: (item[0], item[1]))
        matched_globals: set[str] = set()
        matched_currents: set[str] = set()

        for _, _, global_id, curr_local_id in candidate_pairs:
            if global_id in matched_globals or curr_local_id in matched_currents:
                continue
            global_performers[global_id].append((frame_idx, curr_local_id))
            active_tracks[global_id] = (frame_idx, current_centers[curr_local_id])
            matched_globals.add(global_id)
            matched_currents.add(curr_local_id)

        for curr_local_id in sorted(current_centers):
            if curr_local_id in matched_currents:
                continue
            global_id = f"performer_{next_global_id:03d}"
            global_performers[global_id] = [(frame_idx, curr_local_id)]
            active_tracks[global_id] = (frame_idx, current_centers[curr_local_id])
            next_global_id += 1

        active_tracks = {
            global_id: state
            for global_id, state in active_tracks.items()
            if frame_idx - state[0] <= max_frame_gap
        }

    return global_performers


def get_performer_bbox_3d(
    instance_data: dict[str, np.ndarray | None],
    invalid_value: float,
    padding: float = 0.5,
    min_keypoint_score: float = 0.05,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    valid = extract_valid_keypoints(instance_data, invalid_value, min_keypoint_score)
    if len(valid) == 0:
        return None, None
    min_corner = valid.min(axis=0) - padding
    max_corner = valid.max(axis=0) + padding
    return min_corner, max_corner


def build_performer_metadata(
    poses: dict[int, dict[str, dict[str, np.ndarray | None]]],
    global_performers: dict[str, list[tuple[int, str]]],
    invalid_value: float,
    bbox_padding: float = 0.5,
    min_frames: int = 5,
    min_keypoint_score: float = 0.05,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    for global_id, frame_list in sorted(global_performers.items()):
        if len(frame_list) < min_frames:
            continue

        frame_indices = [frame_idx for frame_idx, _ in frame_list]
        frame_bboxes: dict[str, dict[str, Any]] = {}
        all_mins = []
        all_maxs = []

        for frame_idx, local_id in frame_list:
            instance_data = poses[frame_idx].get(local_id)
            if instance_data is None:
                continue

            min_corner, max_corner = get_performer_bbox_3d(
                instance_data,
                invalid_value=invalid_value,
                padding=bbox_padding,
                min_keypoint_score=min_keypoint_score,
            )
            if min_corner is None or max_corner is None:
                continue

            center = (min_corner + max_corner) / 2.0
            size = max_corner - min_corner
            frame_bboxes[str(frame_idx)] = {
                "local_id": local_id,
                "min": min_corner.tolist(),
                "max": max_corner.tolist(),
                "center": center.tolist(),
                "size": size.tolist(),
            }
            all_mins.append(min_corner)
            all_maxs.append(max_corner)

        if not frame_bboxes:
            continue

        all_mins_array = np.asarray(all_mins, dtype=np.float32)
        all_maxs_array = np.asarray(all_maxs, dtype=np.float32)
        global_min = all_mins_array.min(axis=0)
        global_max = all_maxs_array.max(axis=0)
        global_center = (global_min + global_max) / 2.0
        global_size = global_max - global_min

        metadata[global_id] = {
            "frame_count": len(frame_bboxes),
            "frame_range": [int(min(frame_indices)), int(max(frame_indices))],
            "frame_list": [[int(frame_idx), str(local_id)] for frame_idx, local_id in frame_list],
            "global_bbox": {
                "min": global_min.tolist(),
                "max": global_max.tolist(),
                "center": global_center.tolist(),
                "size": global_size.tolist(),
            },
            "frame_bboxes": frame_bboxes,
        }

    return metadata


def save_metadata(metadata: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved metadata to {output_path}")


def main() -> None:
    args = parse_args()

    print("Loading triangulated poses...")
    poses, invalid_value = load_triangulated_poses(args.poses_dir)
    print(f"Loaded poses for {len(poses)} frames")

    print("Tracking performers across frames...")
    global_performers = track_performers_across_frames(
        poses,
        invalid_value=invalid_value,
        distance_threshold=args.distance_threshold,
        max_frame_gap=args.max_frame_gap,
        min_keypoint_score=args.min_keypoint_score,
    )
    print(f"Identified {len(global_performers)} tracked performer paths before filtering")

    print("Building metadata...")
    metadata = build_performer_metadata(
        poses,
        global_performers,
        invalid_value=invalid_value,
        bbox_padding=args.bbox_padding,
        min_frames=args.min_frames,
        min_keypoint_score=args.min_keypoint_score,
    )
    print(f"Final metadata for {len(metadata)} performers (min {args.min_frames} frames each)")

    for performer_id, data in sorted(metadata.items()):
        size = data["global_bbox"]["size"]
        frame_start, frame_end = data["frame_range"]
        print(
            f"  {performer_id}: frames {frame_start}-{frame_end}, "
            f"global bbox {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
        )

    save_metadata(metadata, args.output_metadata)


if __name__ == "__main__":
    main()
