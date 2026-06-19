import os
import os.path as osp
import json
import fire
import shutil

from copy import deepcopy
from PIL import Image
from src.utils import RankedLogger
from src.data.utils.crop_utils import skeleton_to_mask
from scripts.preprocess.remove_background import remove_background

log = RankedLogger(__name__, rank_zero_only=True)


def _list_result_image_labels(images_dir: str) -> dict[str, list[str]]:
    if not osp.isdir(images_dir):
        return {}
    labels_by_camera = {}
    for camera_label in sorted(os.listdir(images_dir)):
        camera_dir = osp.join(images_dir, camera_label)
        if not osp.isdir(camera_dir):
            continue
        labels = []
        for filename in sorted(os.listdir(camera_dir)):
            path = osp.join(camera_dir, filename)
            if osp.isfile(path):
                labels.append(osp.splitext(filename)[0])
        if labels:
            labels_by_camera[camera_label] = labels
    return labels_by_camera


def _iter_result_images(images_dir: str):
    if not osp.isdir(images_dir):
        return
    for camera_label in sorted(os.listdir(images_dir)):
        camera_dir = osp.join(images_dir, camera_label)
        if not osp.isdir(camera_dir):
            continue
        for filename in sorted(os.listdir(camera_dir)):
            image_path = osp.join(camera_dir, filename)
            if not osp.isfile(image_path):
                continue
            image_label, _ = osp.splitext(filename)
            yield camera_label, image_label, image_path


def _load_dataset_mask(data_dir: str, camera_label: str, image_label: str) -> Image.Image | None:
    fmask_path = osp.join(data_dir, "fmasks", camera_label, f"{image_label}.png")
    if osp.isfile(fmask_path):
        return Image.open(fmask_path).convert("L")

    skeleton_path = osp.join(data_dir, "skeletons", camera_label, f"{image_label}.png")
    if osp.isfile(skeleton_path):
        skeleton = Image.open(skeleton_path).convert("RGB")
        return skeleton_to_mask(skeleton).convert("L")

    return None


def _export_alpha_from_dataset_masks(data_dir: str, result_dir: str) -> tuple[int, int]:
    images_dir = osp.join(result_dir, "images")
    out_fmasks_dir = osp.join(result_dir, "fmasks")
    out_images_alpha_dir = osp.join(result_dir, "images_alpha")

    exported = 0
    missing = 0
    for camera_label, image_label, image_path in _iter_result_images(images_dir):
        mask = _load_dataset_mask(data_dir, camera_label, image_label)
        if mask is None:
            missing += 1
            continue

        image = Image.open(image_path).convert("RGB")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        rel_dir = osp.relpath(osp.dirname(image_path), images_dir)
        out_mask_path = osp.join(out_fmasks_dir, rel_dir, f"{image_label}.png")
        out_alpha_path = osp.join(out_images_alpha_dir, rel_dir, f"{image_label}.png")

        os.makedirs(osp.dirname(out_mask_path), exist_ok=True)
        os.makedirs(osp.dirname(out_alpha_path), exist_ok=True)

        mask.save(out_mask_path)
        image.putalpha(mask)
        image.save(out_alpha_path)
        exported += 1

    return exported, missing


def _expand_frames_for_result_images(cameras: dict, result_images_dir: str) -> dict:
    labels_by_camera = _list_result_image_labels(result_images_dir)
    if not labels_by_camera:
        return cameras

    frames = cameras.get("frames", [])
    frames_by_camera = {}
    for frame in frames:
        frames_by_camera.setdefault(frame["camera_label"], []).append(frame)

    expanded_frames = []
    changed = False

    for camera_label, image_labels in labels_by_camera.items():
        camera_frames = frames_by_camera.get(camera_label, [])
        if not camera_frames:
            continue

        if len(camera_frames) == len(image_labels):
            # Already has one transform entry per image.
            expanded_frames.extend(camera_frames)
            continue

        if len(camera_frames) != 1:
            # Keep the original frames if the structure is ambiguous.
            expanded_frames.extend(camera_frames)
            continue

        changed = True
        base_frame = camera_frames[0]
        for frame_index, image_label in enumerate(image_labels):
            frame = deepcopy(base_frame)
            frame["frame"] = frame_index
            frame["file_path"] = f"images/{camera_label}/{image_label}.png"
            expanded_frames.append(frame)

    if not changed:
        return cameras

    other_frames = []
    known_labels = set(labels_by_camera.keys())
    for frame in frames:
        if frame["camera_label"] not in known_labels:
            other_frames.append(frame)

    cameras = deepcopy(cameras)
    cameras["frames"] = expanded_frames + other_frames
    return cameras


def diffuman4d_to_nerfstudio(
    data_dir: str,
    result_dir: str,
    input_cameras: list[str] = None,
    mask_source: str = "dataset",
):
    # copy nerfstudio cameras
    cameras_path = f"{data_dir}/transforms.json"
    cameras = json.load(open(cameras_path))
    cameras = _expand_frames_for_result_images(cameras, f"{result_dir}/images")

    if input_cameras is not None:
        cameras_input = deepcopy(cameras)
        cameras_input["frames"] = []

    for frame in cameras["frames"]:
        ext = osp.splitext(frame["file_path"])[1]
        frame["file_path"] = frame["file_path"].replace(ext, ".png").replace("images/", "images_alpha/")
        # record input cameras
        if input_cameras is not None and frame["camera_label"] in input_cameras:
            cameras_input["frames"].append(frame)

    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/transforms.json", "w") as f:
        json.dump(cameras, f, indent=4)
    with open(f"{result_dir}/transforms_input.json", "w") as f:
        json.dump(cameras_input, f, indent=4)
    log.info(f"Saved nerfstudio cameras to {result_dir}/transforms.json and {result_dir}/transforms_input.json")

    # copy point cloud if available
    sparse_pcd_path = f"{data_dir}/sparse_pcd.ply"
    if osp.isfile(sparse_pcd_path):
        shutil.copy(sparse_pcd_path, f"{result_dir}/sparse_pcd.ply")
        log.info(f"Saved point cloud to {result_dir}/sparse_pcd.ply")
    else:
        log.warning(
            f"Point cloud not found at {sparse_pcd_path}. "
            "Skipping sparse_pcd.ply copy for Nerfstudio export."
        )

    if mask_source not in {"dataset", "auto", "birefnet"}:
        raise ValueError(f"Unsupported mask_source: {mask_source}")

    missing_masks = None
    if mask_source in {"dataset", "auto"}:
        exported, missing_masks = _export_alpha_from_dataset_masks(data_dir=data_dir, result_dir=result_dir)
        log.info(
            f"Saved {exported} foreground masks from dataset assets to {result_dir}/fmasks"
        )
        if mask_source == "dataset" and missing_masks:
            raise FileNotFoundError(
                f"Missing {missing_masks} dataset masks while exporting Nerfstudio assets from {data_dir}."
            )

    if mask_source == "birefnet" or (mask_source == "auto" and missing_masks):
        remove_background(
            images_dir=f"{result_dir}/images",
            out_fmasks_dir=f"{result_dir}/fmasks",
            out_images_alpha_dir=f"{result_dir}/images_alpha",
            model_name="ZhengPeng7/BiRefNet",
            image_ext=".jpg",
            mask_ext=".png",
            rotate_clockwise=0,
            batch_size=4,  # decrease it if OOM
            skip_exists=True,
        )
        log.info(f"Saved foreground masks to {result_dir}/fmasks")


if __name__ == "__main__":
    fire.Fire(diffuman4d_to_nerfstudio)
