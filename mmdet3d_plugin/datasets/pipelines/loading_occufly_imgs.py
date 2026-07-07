import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from mmdet.datasets.builder import PIPELINES


def _infer_depth_shape(flat_size, image_hw=None):
    # image_hw is (H, W) at RGB resolution. OccuFly depth is normally 1/4 RGB.
    if image_hw is not None:
        h, w = image_hw
        for div in (4, 1, 2, 8):
            dh, dw = h // div, w // div
            if dh > 0 and dw > 0 and dh * dw == flat_size:
                return dh, dw
    for h, w in ((912, 1368), (576, 864), (1024, 1536), (768, 1152)):
        if h * w == flat_size:
            return h, w
    # Fallback to 3:2 landscape aspect ratio, rounded to a factor pair.
    h = int(round((flat_size * 2.0 / 3.0) ** 0.5))
    while h > 1 and flat_size % h != 0:
        h -= 1
    if h <= 1:
        h = int(round(flat_size ** 0.5))
    w = flat_size // h
    return h, w


def load_depth_npy(path, image_hw=None, clip_max=None):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError("Depth map not found: %s" % path)
    depth = np.load(path)
    if depth.ndim == 1:
        h, w = _infer_depth_shape(depth.size, image_hw=image_hw)
        depth = depth.reshape(h, w)
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    if clip_max is not None:
        depth[depth > clip_max] = 0.0
    return depth


def load_depth_any(path, image_hw=None, clip_max=None):
    suffix = os.path.splitext(path)[1].lower()
    confidence = None
    if suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = np.asarray(data[key])
        if "confidence" in data:
            confidence = np.asarray(data["confidence"], dtype=np.float32)
    elif suffix in (".pt", ".pth"):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            depth = payload.get("depth", payload.get("metric_depth", payload.get("tensor")))
            confidence = payload.get("confidence")
            if depth is None:
                raise KeyError("Depth-prior file has no depth/metric_depth/tensor key: %s" % path)
        else:
            depth = payload
        depth = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
        if torch.is_tensor(confidence):
            confidence = confidence.detach().cpu().numpy()
    elif suffix in (".png", ".tif", ".tiff"):
        depth = np.array(Image.open(path), copy=True)
    else:
        depth = load_depth_npy(path, image_hw=image_hw, clip_max=clip_max)
    if depth.ndim == 1:
        h, w = _infer_depth_shape(depth.size, image_hw=image_hw)
        depth = depth.reshape(h, w)
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    if clip_max is not None:
        depth[depth > clip_max] = 0.0
    return depth, None if confidence is None else np.asarray(confidence, dtype=np.float32)


@PIPELINES.register_module()
class LoadOccuFlyImageFromFiles(object):
    """Single-camera OccuFly RGB/depth loader compatible with CGFormer.

    This mirrors CGFormer's LoadMultiViewImageFromFiles but reshapes OccuFly's
    flattened depth .npy files before applying the same resize/crop/flip as RGB.
    """

    def __init__(self, data_config, is_train=False, img_norm_cfg=None,
                 load_stereo_depth=True, load_gt_depth=True,
                 color_jitter=(0.4, 0.4, 0.4), clip_depth_max=128.0):
        super().__init__()
        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg
        self.load_stereo_depth = load_stereo_depth
        self.load_gt_depth = load_gt_depth
        self.clip_depth_max = clip_depth_max
        self.color_jitter = transforms.ColorJitter(*color_jitter) if color_jitter else None
        self.normalize_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ToTensor = transforms.ToTensor()

    def get_rot(self, h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate, resample=None):
        if resample is None:
            resample = Image.BILINEAR
        img = img.resize(resize_dims, resample=resample)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        if rotate:
            img = img.rotate(rotate, resample=resample)
        return img

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate, resample=Image.BILINEAR)
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        return img, post_rot, post_tran

    def _load_and_transform_depth(self, path, image_hw, img_augs):
        resize, resize_dims, crop, flip, rotate = img_augs
        depth, _ = load_depth_any(path, image_hw=image_hw, clip_max=self.clip_depth_max)
        depth_img = Image.fromarray(depth.astype(np.float32), mode='F')
        depth_img = self.img_transform_core(depth_img, resize_dims=resize_dims, crop=crop,
                                            flip=flip, rotate=rotate, resample=Image.BILINEAR)
        depth = np.asarray(depth_img, dtype=np.float32)
        return self.ToTensor(depth)

    def _load_and_transform_teacher_depth(self, path, image_hw, img_augs):
        resize, resize_dims, crop, flip, rotate = img_augs
        depth, confidence = load_depth_any(path, image_hw=image_hw, clip_max=self.clip_depth_max)
        depth_img = Image.fromarray(depth.astype(np.float32), mode='F')
        depth_img = self.img_transform_core(depth_img, resize_dims=resize_dims, crop=crop,
                                            flip=flip, rotate=rotate, resample=Image.BILINEAR)
        out_depth = self.ToTensor(np.asarray(depth_img, dtype=np.float32))
        out_conf = None
        if confidence is not None:
            conf_img = Image.fromarray(confidence.astype(np.float32), mode='F')
            conf_img = self.img_transform_core(conf_img, resize_dims=resize_dims, crop=crop,
                                               flip=flip, rotate=rotate, resample=Image.BILINEAR)
            out_conf = self.ToTensor(np.asarray(conf_img, dtype=np.float32))
        return out_depth, out_conf

    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']
        focal_length = results['focal_length']
        baseline = results['baseline']
        data_lists = []
        raw_img_list = []
        img_augs = None
        original_hw = None
        for i, img_filename in enumerate(img_filenames):
            img = Image.open(img_filename).convert('RGB')
            original_hw = (img.height, img.width)
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)
            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize, resize_dims=resize_dims,
                crop=crop, flip=flip, rotate=rotate)
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            intrin = torch.Tensor(results['cam_intrinsic'][i])
            lidar2cam = torch.Tensor(results['lidar2cam'][i])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]
            canvas = np.array(img)
            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)
            img = self.normalize_img(img)
            result = [img, rot, tran, intrin, post_rot, post_tran, cam2lidar]
            result = [x[None] for x in result]
            data_lists.append(result)
            raw_img_list.append(canvas)

        if self.load_stereo_depth:
            results['stereo_depth'] = self._load_and_transform_depth(
                results['stereo_depth_path'], original_hw, img_augs)
        if self.load_gt_depth and 'gt_depth_path' in results and results['gt_depth_path'] is not None:
            try:
                results['gt_depths'] = self._load_and_transform_depth(
                    results['gt_depth_path'], original_hw, img_augs)
            except FileNotFoundError:
                if 'stereo_depth' in results:
                    results['gt_depths'] = results['stereo_depth'].clone()
        elif self.load_stereo_depth and 'stereo_depth' in results:
            results['gt_depths'] = results['stereo_depth'].clone()

        if results.get('vjepa_depth_prior_teacher_path'):
            teacher, confidence = self._load_and_transform_teacher_depth(
                results['vjepa_depth_prior_teacher_path'], original_hw, img_augs)
            results['vjepa_depth_prior_teacher'] = teacher
            if confidence is not None:
                results['vjepa_depth_prior_teacher_confidence'] = confidence

        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))
        results['focal_length'] = torch.tensor(focal_length, dtype=torch.float32)
        results['baseline'] = torch.tensor(baseline, dtype=torch.float32)
        results['raw_img'] = raw_img_list
        return result_list

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results
