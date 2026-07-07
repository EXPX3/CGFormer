import glob
import os
import pickle
import re
import warnings

import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from PIL import Image
from torch.utils.data import Dataset


OCCUFLY_RAW_TO_TRAIN = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    16: 14,
    17: 15,
    21: 16,
    22: 17,
    33: 18,
    34: 19,
    35: 20,
    36: 21,
    255: 255,
}

OCCUFLY_TRAIN_TO_RAW = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 16,
    15: 17,
    16: 21,
    17: 22,
    18: 33,
    19: 34,
    20: 35,
    21: 36,
    255: 255,
}

GRID_DIMS = (192, 128, 128)
ALTITUDES = (30, 40, 50)
SPLITS = {
    "train": ["scene_%02d" % i for i in range(1, 6)],
    "val": ["scene_%02d" % i for i in range(6, 8)],
    "validation": ["scene_%02d" % i for i in range(6, 8)],
    "test": ["scene_%02d" % i for i in range(8, 10)],
    "all": ["scene_%02d" % i for i in range(1, 10)],
}


def _as_path(path):
    return None if path in (None, "", "None") else os.path.abspath(os.path.expanduser(path))


def _resolve_dataset_root(data_root):
    data_root = _as_path(data_root)
    if data_root is None:
        raise ValueError("data_root must point to OccuFly_Dataset or its parent")
    if os.path.isdir(os.path.join(data_root, "scene_01")):
        return data_root
    nested = os.path.join(data_root, "OccuFly_Dataset")
    if os.path.isdir(os.path.join(nested, "scene_01")):
        return nested
    raise FileNotFoundError(
        "Could not find OccuFly scenes under %s. Expected scene_01 or OccuFly_Dataset/scene_01" % data_root
    )


def _default_depth_root(data_root):
    parent = os.path.dirname(data_root.rstrip(os.sep))
    cand = os.path.join(parent, "OccuFly_Predicted_DepthMaps")
    return cand


SUPPORTED_VJEPA_DEPTH_SUFFIXES = (".npy", ".npz", ".pt", ".pth", ".png", ".tif", ".tiff")


def _make_label_lut():
    lut = np.full((256,), 255, dtype=np.uint8)
    for raw_id, train_id in OCCUFLY_RAW_TO_TRAIN.items():
        lut[int(raw_id)] = np.uint8(train_id)
    return lut


_LABEL_LUT = _make_label_lut()


def remap_raw_labels_to_train(labels):
    labels = labels.astype(np.uint8, copy=False)
    return _LABEL_LUT[labels]


def _numbers_from_text(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", text)]


def read_occufly_calibration(calib_path):
    """Return a 4x4 pinhole intrinsic matrix from OccuFly calibration.txt.

    The public docs define calibration.txt as focal length, principal point, and
    image dimensions. This parser accepts key/value files, compact matrix lines,
    or simple four-number fx fy cx cy files.
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError("Missing OccuFly calibration: %s" % calib_path)

    text = open(calib_path, "r").read()
    lower = text.lower()

    vals = {}
    for key in ("fx", "fy", "cx", "cy", "width", "height", "w", "h"):
        m = re.search(r"(?:^|[^a-zA-Z0-9_])%s\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)" % key, lower)
        if m:
            vals[key] = float(m.group(1))

    if all(k in vals for k in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = vals["fx"], vals["fy"], vals["cx"], vals["cy"]
    else:
        # Try matrix-like lines: K:, P:, intrinsic:, or any line with 9/12 numbers.
        fx = fy = cx = cy = None
        for line in text.splitlines():
            nums = _numbers_from_text(line)
            if len(nums) >= 12:
                fx, fy, cx, cy = nums[0], nums[5], nums[2], nums[6]
                break
            if len(nums) == 9:
                fx, fy, cx, cy = nums[0], nums[4], nums[2], nums[5]
                break
        if fx is None:
            nums = _numbers_from_text(text)
            if len(nums) >= 4:
                fx, fy, cx, cy = nums[0], nums[1], nums[2], nums[3]
            else:
                raise ValueError("Could not parse fx/fy/cx/cy from %s" % calib_path)

    K = np.eye(4, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _load_pkl_grid(path, key="1_1"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if key not in data:
        raise KeyError("%s missing key %s" % (path, key))
    arr = np.asarray(data[key], dtype=np.uint8)
    if arr.shape != GRID_DIMS:
        arr = arr.reshape(GRID_DIMS)
    return arr


def _load_raw_label(path):
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size != np.prod(GRID_DIMS):
        raise ValueError("%s has %d voxels, expected %d" % (path, arr.size, np.prod(GRID_DIMS)))
    return arr.reshape(GRID_DIMS)


@DATASETS.register_module()
class OccuFlyDataset(Dataset):
    def __init__(
        self,
        data_root,
        depth_root=None,
        gt_depth_root=None,
        pipeline=None,
        split="train",
        camera_used=("visual",),
        occ_size=(192, 128, 128),
        pc_range=(-48.0, -32.0, 0.0, 48.0, 32.0, 64.0),
        altitudes=(30, 40, 50),
        scenes=None,
        test_mode=False,
        remap_labels=True,
        label_key="1_1",
        require_depth=True,
        max_samples_per_scene_altitude=None,
        load_gt=True,
        vjepa_feature_root=None,
        paired_training=False,
        pair_frame_offset=1,
        pair_search_radius=5,
        pair_max_translation_m=30.0,
        pair_require_projection=True,
        pair_min_projected_voxels=1,
        vjepa_depth_prior_teacher_root=None,
        require_vjepa_depth_prior_teacher=False,
    ):
        super().__init__()
        self.data_root = _resolve_dataset_root(data_root)
        self.depth_root = _as_path(depth_root) or _default_depth_root(self.data_root)
        self.gt_depth_root = _as_path(gt_depth_root) or self.data_root
        self.split = split
        self.camera_used = camera_used
        self.occ_size = tuple(int(x) for x in occ_size)
        self.pc_range = tuple(float(x) for x in pc_range)
        self.altitudes = tuple(int(a) for a in altitudes)
        self.test_mode = bool(test_mode)
        self.remap_labels = bool(remap_labels)
        self.label_key = label_key
        self.require_depth = bool(require_depth)
        self.max_samples_per_scene_altitude = max_samples_per_scene_altitude
        self.load_gt = bool(load_gt)
        self.vjepa_feature_root = _as_path(vjepa_feature_root)
        self.paired_training = bool(paired_training)
        self.pair_frame_offset = max(1, int(pair_frame_offset))
        self.pair_search_radius = max(self.pair_frame_offset, int(pair_search_radius))
        self.pair_max_translation_m = float(pair_max_translation_m)
        self.pair_require_projection = bool(pair_require_projection)
        self.pair_min_projected_voxels = max(0, int(pair_min_projected_voxels))
        self.vjepa_depth_prior_teacher_root = _as_path(vjepa_depth_prior_teacher_root)
        self.require_vjepa_depth_prior_teacher = bool(require_vjepa_depth_prior_teacher)
        self._pair_probe_points = None

        if scenes is None:
            if split not in SPLITS:
                raise KeyError("Unknown split %s; expected one of %s" % (split, sorted(SPLITS)))
            self.scenes = SPLITS[split]
        else:
            self.scenes = list(scenes)

        self.data_infos = self.load_annotations()
        if not self.data_infos:
            raise RuntimeError(
                "No OccuFly samples found for split=%s scenes=%s altitudes=%s data_root=%s depth_root=%s" % (
                    split, self.scenes, self.altitudes, self.data_root, self.depth_root
                )
            )
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_data(idx)
        while True:
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_data(self, index):
        input_dict = self.get_data_info(index)
        if self.pipeline is None:
            return input_dict
        return self.pipeline(input_dict)

    def _depth_path(self, root, scene, altitude, frame_id):
        if root is None:
            return None
        direct = os.path.join(root, scene, str(altitude), "depth_maps", frame_id + ".npy")
        if os.path.exists(direct):
            return direct
        nested = os.path.join(root, "OccuFly_Predicted_DepthMaps", scene, str(altitude), "depth_maps", frame_id + ".npy")
        if os.path.exists(nested):
            return nested
        nested_gt = os.path.join(root, "OccuFly_Dataset", scene, str(altitude), "depth_maps", frame_id + ".npy")
        if os.path.exists(nested_gt):
            return nested_gt
        return direct

    def _vjepa_feature_path(self, scene, altitude, frame_id):
        if self.vjepa_feature_root is None:
            return None
        return os.path.join(self.vjepa_feature_root, scene, str(altitude), frame_id + ".pt")

    def _vjepa_depth_prior_candidates(self, scene, altitude, frame_id):
        if self.vjepa_depth_prior_teacher_root is None:
            return []
        stems = [
            os.path.join(scene, str(altitude), "depth_maps", frame_id),
            os.path.join(scene, str(altitude), frame_id),
            "%s_%s_%s" % (scene, altitude, frame_id),
            "%s_%s" % (scene, frame_id),
            frame_id,
        ]
        return [
            os.path.join(self.vjepa_depth_prior_teacher_root, stem + suffix)
            for stem in stems
            for suffix in SUPPORTED_VJEPA_DEPTH_SUFFIXES
        ]

    def _vjepa_depth_prior_path(self, scene, altitude, frame_id):
        candidates = self._vjepa_depth_prior_candidates(scene, altitude, frame_id)
        for path in candidates:
            if os.path.exists(path):
                return path, candidates
        return None, candidates

    def _load_vjepa_teacher(self, path):
        if path is None or not os.path.exists(path):
            return None
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "tokens_2d" in payload:
                return payload["tokens_2d"].float()
            if "tokens" in payload:
                tokens = payload["tokens"]
                if torch.is_tensor(tokens):
                    return tokens.float()
        if torch.is_tensor(payload):
            return payload.float()
        raise TypeError("Unsupported V-JEPA feature cache item: %s" % path)

    def _load_pose(self, pkl_path):
        if pkl_path is None or not os.path.exists(pkl_path):
            return np.eye(4, dtype=np.float32)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        pose = data.get("pose") if isinstance(data, dict) else None
        if pose is None:
            return np.eye(4, dtype=np.float32)
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError("%s pose has shape %s, expected (4, 4)" % (pkl_path, pose.shape))
        return pose

    def _image_size(self, image_path):
        with Image.open(image_path) as img:
            return np.array([img.height, img.width], dtype=np.int32)

    def _pair_probe_grid(self):
        if self._pair_probe_points is not None:
            return self._pair_probe_points
        nx, ny, nz = (min(int(self.occ_size[0]), 32),
                      min(int(self.occ_size[1]), 24),
                      min(int(self.occ_size[2]), 24))
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        xs = np.linspace(xmin, xmax, nx, endpoint=False, dtype=np.float32) + (xmax - xmin) / (2.0 * nx)
        ys = np.linspace(ymin, ymax, ny, endpoint=False, dtype=np.float32) + (ymax - ymin) / (2.0 * ny)
        zs = np.linspace(zmin, zmax, nz, endpoint=False, dtype=np.float32) + (zmax - zmin) / (2.0 * nz)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        points = np.stack([gx, gy, gz, np.ones_like(gx)], axis=-1).reshape(-1, 4).T
        self._pair_probe_points = points.astype(np.float32, copy=False)
        return self._pair_probe_points

    def _pair_projection_count(self, source, target):
        points = self._pair_probe_grid()
        target_from_source = np.linalg.inv(target["pose"]).dot(source["pose"])
        xyz = target_from_source.dot(points)[:3]
        z_raw = xyz[2]
        z = np.maximum(z_raw, 1e-4)
        k = target["P"]
        u = k[0, 0] * (xyz[0] / z) + k[0, 2]
        v = k[1, 1] * (xyz[1] / z) + k[1, 2]
        img_h, img_w = target["image_size"]
        valid = (z_raw > 1e-4) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        return int(valid.sum())

    def _pair_translation(self, source, target):
        return float(np.linalg.norm(source["pose"][:3, 3] - target["pose"][:3, 3]))

    def _candidate_pair_offsets(self):
        offsets = []
        for step in range(1, self.pair_search_radius + 1):
            offsets.extend((step, -step))
        if self.pair_frame_offset > 1:
            preferred = [self.pair_frame_offset, -self.pair_frame_offset]
            offsets = preferred + [x for x in offsets if x not in preferred]
        return offsets

    def _select_pair_target(self, items, pos):
        source = items[pos][1]
        n = len(items)
        fallback = None
        for offset in self._candidate_pair_offsets():
            target_pos = pos + offset
            if target_pos < 0 or target_pos >= n:
                continue
            target = items[target_pos][1]
            distance = self._pair_translation(source, target)
            if fallback is None or distance < fallback[0]:
                fallback = (distance, target)
            if self.pair_max_translation_m > 0 and distance > self.pair_max_translation_m:
                continue
            if self.pair_require_projection:
                projected = self._pair_projection_count(source, target)
                if projected < self.pair_min_projected_voxels:
                    continue
            return target, True
        if fallback is not None:
            return fallback[1], False
        return source, False

    def _attach_pairs(self, scans):
        if not self.paired_training:
            return scans
        grouped = {}
        for idx, info in enumerate(scans):
            grouped.setdefault(info["sequence"], []).append((idx, info))
        num_fallback = 0
        for items in grouped.values():
            items.sort(key=lambda item: int(item[1]["frame_id"]) if item[1]["frame_id"].isdigit() else item[1]["frame_id"])
            for pos, (idx, info) in enumerate(items):
                target, is_valid = self._select_pair_target(items, pos)
                if not is_valid:
                    num_fallback += 1
                info["target_frame_id"] = target["frame_id"]
                info["target_pose"] = target["pose"]
                info["target_P"] = target["P"]
                info["target_image_size"] = target["image_size"]
                info["target_vjepa_feature_path"] = target.get("vjepa_feature_path")
        if num_fallback:
            warnings.warn(
                "OccuFly paired training used fallback targets for %d samples because no candidate within "
                "radius=%d, max_translation=%.2fm, and projection threshold=%d was found." % (
                    num_fallback,
                    self.pair_search_radius,
                    self.pair_max_translation_m,
                    self.pair_min_projected_voxels,
                )
            )
        return scans

    def load_annotations(self):
        scans = []
        skipped_missing_depth = 0
        for scene in self.scenes:
            for altitude in self.altitudes:
                scene_alt_dir = os.path.join(self.data_root, scene, str(altitude))
                if not os.path.isdir(scene_alt_dir):
                    warnings.warn("Missing OccuFly scene/altitude directory: %s" % scene_alt_dir)
                    continue
                calib_path = os.path.join(self.data_root, scene, "calibration.txt")
                K = read_occufly_calibration(calib_path)
                image_dir = os.path.join(scene_alt_dir, "images", "visual")
                preprocess_dir = os.path.join(scene_alt_dir, "preprocess")
                gt_dir = os.path.join(scene_alt_dir, "ground_truth")
                frame_files = sorted(glob.glob(os.path.join(preprocess_dir, "*.pkl")))
                if frame_files:
                    frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in frame_files]
                    pkl_by_frame = {os.path.splitext(os.path.basename(p))[0]: p for p in frame_files}
                else:
                    frame_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gt_dir, "*")) if os.path.isdir(p))
                    pkl_by_frame = {}
                if self.max_samples_per_scene_altitude is not None:
                    frame_ids = frame_ids[: int(self.max_samples_per_scene_altitude)]
                for frame_id in frame_ids:
                    img_path = os.path.join(image_dir, frame_id + ".png")
                    if not os.path.exists(img_path):
                        continue
                    pred_depth_path = self._depth_path(self.depth_root, scene, altitude, frame_id)
                    if self.require_depth and not os.path.exists(pred_depth_path):
                        skipped_missing_depth += 1
                        continue
                    gt_depth_path = self._depth_path(self.gt_depth_root, scene, altitude, frame_id)
                    vjepa_depth_prior_teacher_path, vjepa_depth_candidates = self._vjepa_depth_prior_path(scene, altitude, frame_id)
                    if self.require_vjepa_depth_prior_teacher and vjepa_depth_prior_teacher_path is None:
                        raise FileNotFoundError(
                            "Missing V-JEPA depth-prior teacher for %s altitude=%s frame=%s. Searched:\n%s" % (
                                scene, altitude, frame_id, "\n".join(vjepa_depth_candidates[:32])
                            )
                        )
                    pkl_path = pkl_by_frame.get(frame_id)
                    raw_label_path = os.path.join(gt_dir, frame_id, frame_id + ".label")
                    if self.load_gt:
                        if pkl_path is None and not os.path.exists(raw_label_path):
                            continue
                    else:
                        pkl_path = None
                        raw_label_path = None
                    sequence = "%s_%s" % (scene, altitude)
                    scans.append({
                        "img_path": img_path,
                        "image_size": self._image_size(img_path),
                        "scene": scene,
                        "altitude": int(altitude),
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "P": K,
                        "T_cam_grid": np.eye(4, dtype=np.float32),
                        "proj_matrix": K.copy(),
                        "pose": self._load_pose(pkl_path),
                        "voxel_path": pkl_path,
                        "raw_label_path": raw_label_path,
                        "stereo_depth_path": pred_depth_path,
                        "gt_depth_path": gt_depth_path,
                        "vjepa_depth_prior_teacher_path": vjepa_depth_prior_teacher_path,
                        "vjepa_feature_path": self._vjepa_feature_path(scene, altitude, frame_id),
                    })
        if skipped_missing_depth:
            warnings.warn("Skipped %d OccuFly frames because predicted depth was missing under %s" % (skipped_missing_depth, self.depth_root))
        return self._attach_pairs(scans)

    def get_data_info(self, index):
        info = self.data_infos[index]
        source_pose = info.get("pose", np.eye(4, dtype=np.float32))
        target_pose = info.get("target_pose", source_pose)
        target_P = info.get("target_P", info["P"])
        source_image_size = info.get("image_size", np.array([0, 0], dtype=np.int32))
        target_image_size = info.get("target_image_size", source_image_size)
        target_frame_id = info.get("target_frame_id", info["frame_id"])
        target_vjepa_feature_path = info.get("target_vjepa_feature_path", info.get("vjepa_feature_path"))
        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info["sequence"],
            frame_id=info["frame_id"],
            scene=info["scene"],
            altitude=info["altitude"],
            img_filename=[info["img_path"]],
            lidar2img=[info["proj_matrix"]],
            cam_intrinsic=[info["P"]],
            lidar2cam=[np.eye(4, dtype=np.float32)],
            focal_length=float(info["P"][0, 0]),
            baseline=0.0,
            stereo_depth_path=info["stereo_depth_path"],
            gt_depth_path=info["gt_depth_path"],
            vjepa_depth_prior_teacher_path=info.get("vjepa_depth_prior_teacher_path"),
            vjepa_feature_path=info.get("vjepa_feature_path"),
            source_pose=source_pose,
            target_pose=target_pose,
            target_cam_intrinsic=target_P,
            source_image_size=source_image_size,
            target_image_size=target_image_size,
            target_frame_id=target_frame_id,
            target_vjepa_feature_path=target_vjepa_feature_path,
        )
        vjepa_teacher = self._load_vjepa_teacher(info.get("vjepa_feature_path"))
        if vjepa_teacher is not None:
            input_dict["vjepa_teacher"] = vjepa_teacher
        target_vjepa_teacher = self._load_vjepa_teacher(target_vjepa_feature_path)
        if target_vjepa_teacher is None:
            target_vjepa_teacher = vjepa_teacher
        if target_vjepa_teacher is not None:
            input_dict["target_vjepa_teacher"] = target_vjepa_teacher
        input_dict["gt_occ"] = self.get_ann_info(index) if self.load_gt else None
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if info.get("voxel_path") is not None and os.path.exists(info["voxel_path"]):
            labels = _load_pkl_grid(info["voxel_path"], self.label_key)
        elif info.get("raw_label_path") is not None and os.path.exists(info["raw_label_path"]):
            labels = _load_raw_label(info["raw_label_path"])
        else:
            return None
        if self.remap_labels:
            labels = remap_raw_labels_to_train(labels)
        return labels
