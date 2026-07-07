import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d_plugin.models.utils.aero_depth_teacher import altitude_depth_teacher_loss

@DETECTORS.register_module()
class CGFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        aero_jepa=None,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.depth_loss = depth_loss
        self.aero_jepa = aero_jepa or {}
        self.use_vjepa_distill = bool(self.aero_jepa.get('use_vjepa_distill', False))
        self.use_latent_view = bool(self.aero_jepa.get('use_latent_view', False))
        self.use_masked_3d = bool(self.aero_jepa.get('use_masked_3d', False))
        self.use_output_consistency = bool(self.aero_jepa.get('use_output_consistency', False))
        self.lambda_vjepa_distill = float(self.aero_jepa.get('lambda_vjepa_distill', 0.0))
        self.lambda_latent_view = float(self.aero_jepa.get('lambda_latent_view', 0.0))
        self.lambda_masked_3d = float(self.aero_jepa.get('lambda_masked_3d', 0.0))
        self.lambda_output_consistency = float(self.aero_jepa.get('lambda_output_consistency', 0.0))
        self.require_vjepa_teacher = bool(self.aero_jepa.get('require_vjepa_teacher', False))
        self.masked_3d_ratio = float(self.aero_jepa.get('masked_3d_ratio', 0.15))
        self.m4_mask_type = str(self.aero_jepa.get('m4_mask_type', 'random')).lower()
        self.m4_total_mask_ratio = float(self.aero_jepa.get('m4_total_mask_ratio', 0.20))
        self.m4_max_mask_ratio = float(self.aero_jepa.get('m4_max_mask_ratio', 0.50))
        self.m4_min_masked_voxels = int(self.aero_jepa.get('m4_min_masked_voxels', 1))
        self.m4_random_ratio = float(self.aero_jepa.get('m4_random_ratio', self.masked_3d_ratio))
        self.m4_block_num_min = int(self.aero_jepa.get('m4_block_num_min', 2))
        self.m4_block_num_max = int(self.aero_jepa.get('m4_block_num_max', 8))
        self.m4_block_size_min = tuple(int(x) for x in self.aero_jepa.get('m4_block_size_min', [3, 3, 2]))
        self.m4_block_size_max = tuple(int(x) for x in self.aero_jepa.get('m4_block_size_max', [12, 12, 6]))
        self.m4_column_num_min = int(self.aero_jepa.get('m4_column_num_min', 2))
        self.m4_column_num_max = int(self.aero_jepa.get('m4_column_num_max', 8))
        self.m4_column_size_min = tuple(int(x) for x in self.aero_jepa.get('m4_column_size_min', [2, 2]))
        self.m4_column_size_max = tuple(int(x) for x in self.aero_jepa.get('m4_column_size_max', [10, 10]))
        self.m4_frustum_num_min = int(self.aero_jepa.get('m4_frustum_num_min', 1))
        self.m4_frustum_num_max = int(self.aero_jepa.get('m4_frustum_num_max', 4))
        self.m4_frustum_angle_width_min = float(self.aero_jepa.get('m4_frustum_angle_width_min', 5.0))
        self.m4_frustum_angle_width_max = float(self.aero_jepa.get('m4_frustum_angle_width_max', 25.0))
        self.m4_frustum_depth_min = float(self.aero_jepa.get('m4_frustum_depth_min', 0.2))
        self.m4_frustum_depth_max = float(self.aero_jepa.get('m4_frustum_depth_max', 1.0))
        self.m4_mix_random_ratio = float(self.aero_jepa.get('m4_mix_random_ratio', 0.30))
        self.m4_mix_block_ratio = float(self.aero_jepa.get('m4_mix_block_ratio', 0.30))
        self.m4_mix_column_ratio = float(self.aero_jepa.get('m4_mix_column_ratio', 0.25))
        self.m4_mix_frustum_ratio = float(self.aero_jepa.get('m4_mix_frustum_ratio', 0.15))
        self.m4_ignore_index = int(self.aero_jepa.get('m4_ignore_index', 255))
        self.output_consistency_confidence = float(self.aero_jepa.get('output_consistency_confidence', 0.6))
        self.altitude_lifting_variant = str(self.aero_jepa.get('altitude_lifting_variant', 'none'))
        self.use_gt_metric_depth_teacher = bool(self.aero_jepa.get('use_gt_metric_depth_teacher', False))
        self.use_vjepa_depth_prior_teacher = bool(self.aero_jepa.get('use_vjepa_depth_prior_teacher', False))
        self.teacher_used_at_inference = bool(self.aero_jepa.get('teacher_used_at_inference', False))
        self.lambda_depth_gt_kl = float(self.aero_jepa.get('lambda_depth_gt_kl', 0.0))
        self.lambda_depth_gt_l1 = float(self.aero_jepa.get('lambda_depth_gt_l1', 0.0))
        self.lambda_depth_vjepa_kl = float(self.aero_jepa.get('lambda_depth_vjepa_kl', 0.0))
        self.lambda_depth_vjepa_l1 = float(self.aero_jepa.get('lambda_depth_vjepa_l1', 0.0))
        self.teacher_depth_soft_target = str(self.aero_jepa.get('teacher_depth_soft_target', 'gaussian_log_depth'))
        self.teacher_depth_sigma_log = float(self.aero_jepa.get('teacher_depth_sigma_log', 0.10))
        self.vjepa_teacher_dim = int(self.aero_jepa.get('vjepa_teacher_dim', 1664))
        self.context_channels = int(self.aero_jepa.get('context_channels', 128))
        self.masked_3d_channels = int(self.aero_jepa.get('masked_3d_channels', 128))
        self.vjepa_projector = (
            nn.Conv2d(self.context_channels, self.vjepa_teacher_dim, kernel_size=1)
            if self.use_vjepa_distill else None
        )
        self.latent_view_projector = (
            nn.Conv2d(self.context_channels, self.vjepa_teacher_dim, kernel_size=1)
            if self.use_latent_view else None
        )
        self.masked_3d_head = (
            nn.Conv3d(self.masked_3d_channels, self.masked_3d_channels, kernel_size=1)
            if self.use_masked_3d else None
        )
        self.register_buffer("_latent_grid_cache", torch.empty(0), persistent=False)
        self._latent_grid_cache_key = None

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0])

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)
        
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=img_inputs[1:7],
            lss_volume=coarse_queries,
            img_metas=img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)]
        )

        depth_out = {
            'depth_logits': getattr(self.depth_net, 'last_depth_logits', None),
            'depth_probs': getattr(self.depth_net, 'last_depth_probs', depth),
            'depth_pred': getattr(self.depth_net, 'last_depth_pred', None),
        }

        return x, depth, context, depth_out

    def _batch_tensor(self, data_dict, img_metas, *keys):
        for key in keys:
            if key in data_dict and data_dict[key] is not None:
                value = data_dict[key]
                break
            if isinstance(img_metas, dict) and key in img_metas and img_metas[key] is not None:
                value = img_metas[key]
                break
        else:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if torch.is_tensor(value):
            return value.float()
        return torch.as_tensor(value).float()

    def _depth_teacher_losses(self, data_dict, img_metas, depth_out):
        depth_logits = depth_out.get('depth_logits') if depth_out else None
        if depth_logits is None:
            if self.use_gt_metric_depth_teacher or self.use_vjepa_depth_prior_teacher:
                raise RuntimeError("M1-real depth teacher requested but CGFormer depth logits were not exposed.")
            return {}
        if depth_logits.ndim == 5 and depth_logits.shape[1] == 1:
            depth_logits = depth_logits[:, 0]
        K = self._batch_tensor(data_dict, img_metas, 'camera_intrinsics', 'cam_intrinsic', 'target_cam_intrinsic')
        if K is None:
            img_inputs = data_dict.get('img_inputs')
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 3:
                K = img_inputs[3]
        altitude = self._batch_tensor(data_dict, img_metas, 'altitude')
        gt_depth = self._batch_tensor(data_dict, img_metas, 'gt_metric_depth', 'depth_gt', 'gt_depths')
        vj_depth = self._batch_tensor(data_dict, img_metas, 'vjepa_depth_prior_teacher')
        vj_conf = self._batch_tensor(data_dict, img_metas, 'vjepa_depth_prior_teacher_confidence')
        if self.training:
            if self.use_gt_metric_depth_teacher and gt_depth is None:
                raise RuntimeError("M1-real-GT requested OccuFly GT metric depth, but gt_depths/gt_metric_depth is missing from the batch.")
            if self.use_vjepa_depth_prior_teacher and vj_depth is None:
                raise RuntimeError("M1-real-VJ requested V-JEPA depth-prior teacher, but vjepa_depth_prior_teacher is missing from the batch.")
        if not (self.use_gt_metric_depth_teacher or self.use_vjepa_depth_prior_teacher):
            return {}
        losses = altitude_depth_teacher_loss(
            depth_logits,
            K,
            altitude,
            gt_metric_depth=gt_depth if self.use_gt_metric_depth_teacher else None,
            vjepa_teacher_depth=vj_depth if self.use_vjepa_depth_prior_teacher else None,
            vjepa_teacher_confidence=vj_conf,
            lambda_depth_gt_kl=self.lambda_depth_gt_kl,
            lambda_depth_gt_l1=self.lambda_depth_gt_l1,
            lambda_depth_vjepa_kl=self.lambda_depth_vjepa_kl,
            lambda_depth_vjepa_l1=self.lambda_depth_vjepa_l1,
            sigma_log=self.teacher_depth_sigma_log,
            target_type=self.teacher_depth_soft_target,
        )
        return {key: value for key, value in losses.items() if key != 'loss'}

    def _teacher_to_2d(self, teacher):
        if isinstance(teacher, (list, tuple)):
            teacher = teacher[0]
        if teacher is None:
            return None
        if teacher.ndim == 5 and teacher.shape[1] == 1:
            teacher = teacher[:, 0]
        if teacher.ndim == 4:
            return teacher.float()
        if teacher.ndim == 3:
            b, n, d = teacher.shape
            side = int(n ** 0.5)
            if side * side != n:
                return None
            return teacher.transpose(1, 2).reshape(b, d, side, side).float()
        return None

    def _cosine_map_loss(self, student, teacher):
        if teacher is None:
            return None
        if student.ndim == 5:
            student = student.flatten(0, 1)
        if teacher.ndim == 5:
            teacher = teacher.flatten(0, 1)
        if self.vjepa_projector is None:
            return None
        student = self.vjepa_projector(student.float())
        if student.shape[-2:] != teacher.shape[-2:]:
            student = F.interpolate(student, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
        if student.shape[1] != teacher.shape[1]:
            raise RuntimeError(
                f"V-JEPA teacher channel count {teacher.shape[1]} does not match configured "
                f"projection output {student.shape[1]}."
            )
        return (1.0 - F.cosine_similarity(student, teacher.detach(), dim=1)).mean()

    def _paired_vjepa_latent_loss(self, context, teacher):
        """Pairwise proxy for M3 latent-view supervision.

        CGFormer's VoxFormer head requires a local batch size of 1. For DDP
        training, rank-neighbor samples are therefore used as the nearby
        source/target pair. With the OccuFly dataloader shuffle disabled, rank
        0/1, 2/3, ... receive adjacent ordered samples at each step.
        """
        if context is None or teacher is None:
            return context.sum() * 0.0 if context is not None else None
        if (
            context.shape[0] == 1 and
            dist.is_available() and
            dist.is_initialized() and
            dist.get_world_size() > 1
        ):
            gathered = [torch.zeros_like(teacher) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, teacher.contiguous())
            rank = dist.get_rank()
            partner = rank ^ 1
            if partner >= dist.get_world_size():
                partner = rank - 1
            return self._cosine_map_loss(context, gathered[partner].to(context.device))
        if context.shape[0] < 2:
            return context.sum() * 0.0 if context is not None else None
        usable = (min(context.shape[0], teacher.shape[0]) // 2) * 2
        if usable < 2:
            return context.sum() * 0.0
        context = context[:usable]
        teacher = teacher[:usable]
        pair_index = torch.arange(usable, device=context.device)
        pair_index = pair_index.view(-1, 2)[:, [1, 0]].reshape(-1)
        return self._cosine_map_loss(context, teacher[pair_index].to(context.device))

    def _meta_tensor(self, img_metas, key, device, dtype=torch.float32):
        value = img_metas.get(key)
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, (list, tuple)):
            value = torch.as_tensor(value, dtype=dtype, device=device)
        elif torch.is_tensor(value):
            value = value.to(device=device, dtype=dtype)
        else:
            value = torch.as_tensor(value, dtype=dtype, device=device)
        if value.ndim == 2 and value.shape == (4, 4):
            value = value.unsqueeze(0)
        if value.ndim == 3 and value.shape[-2:] == (4, 4):
            return value
        return value

    def _target_image_hw(self, img_metas, target_k, device, dtype):
        image_hw = self._meta_tensor(img_metas, 'target_image_size', device, dtype=dtype)
        if image_hw is not None:
            if image_hw.ndim == 1 and image_hw.numel() == 2:
                image_hw = image_hw.unsqueeze(0)
            if image_hw.ndim == 2 and image_hw.shape[-1] >= 2:
                return image_hw[:, :2].clamp_min(1.0)
        # Compatibility fallback for older cached configs. New OccuFly runs
        # attach explicit PNG height/width from the dataset adapter.
        h = target_k[:, 1, 2] * 2.0
        w = target_k[:, 0, 2] * 2.0
        return torch.stack([h, w], dim=1).clamp_min(1.0)

    def _volume_centers(self, volume, img_metas):
        _, _, nx, ny, nz = volume.shape
        pc_range = img_metas.get('pc_range', [-48.0, -32.0, 0.0, 48.0, 32.0, 64.0])
        if isinstance(pc_range, (list, tuple)) and len(pc_range) == 1:
            pc_range = pc_range[0]
        if torch.is_tensor(pc_range):
            pc_range = pc_range.detach().cpu().flatten().tolist()
        pc_range = [float(x) for x in pc_range]
        key = (volume.device, volume.dtype, nx, ny, nz, tuple(pc_range))
        if self._latent_grid_cache_key == key and self._latent_grid_cache.numel() > 0:
            return self._latent_grid_cache
        xs = torch.linspace(pc_range[0], pc_range[3], nx + 1, device=volume.device, dtype=volume.dtype)[:-1]
        ys = torch.linspace(pc_range[1], pc_range[4], ny + 1, device=volume.device, dtype=volume.dtype)[:-1]
        zs = torch.linspace(pc_range[2], pc_range[5], nz + 1, device=volume.device, dtype=volume.dtype)[:-1]
        xs = xs + (pc_range[3] - pc_range[0]) / (2.0 * nx)
        ys = ys + (pc_range[4] - pc_range[1]) / (2.0 * ny)
        zs = zs + (pc_range[5] - pc_range[2]) / (2.0 * nz)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).reshape(-1, 3)
        self._latent_grid_cache = grid
        self._latent_grid_cache_key = key
        return grid

    def _project_volume_to_tokens(self, volume, img_metas, teacher):
        if volume is None or volume.ndim != 5 or teacher is None:
            return None, None
        source_pose = self._meta_tensor(img_metas, 'source_pose', volume.device, volume.dtype)
        target_pose = self._meta_tensor(img_metas, 'target_pose', volume.device, volume.dtype)
        target_k = self._meta_tensor(img_metas, 'target_cam_intrinsic', volume.device, volume.dtype)
        if source_pose is None or target_pose is None or target_k is None:
            return None, None
        if target_k.ndim == 2 and target_k.shape == (4, 4):
            target_k = target_k.unsqueeze(0)

        b, c, nx, ny, nz = volume.shape
        th, tw = teacher.shape[-2:]
        image_hw = self._target_image_hw(img_metas, target_k, volume.device, volume.dtype)
        points = self._volume_centers(volume, img_metas)
        ones = torch.ones((points.shape[0], 1), device=volume.device, dtype=volume.dtype)
        source_h = torch.cat([points, ones], dim=1).t()
        projected = []
        valid_masks = []
        flat_volume = volume.reshape(b, c, -1)

        for bi in range(b):
            sp = source_pose[min(bi, source_pose.shape[0] - 1)]
            tp = target_pose[min(bi, target_pose.shape[0] - 1)]
            k = target_k[min(bi, target_k.shape[0] - 1)]
            hw = image_hw[min(bi, image_hw.shape[0] - 1)]
            target_from_source = torch.linalg.inv(tp).matmul(sp)
            target_xyz = target_from_source.matmul(source_h)[:3]
            z_raw = target_xyz[2]
            z = z_raw.clamp_min(1e-4)
            u = k[0, 0] * (target_xyz[0] / z) + k[0, 2]
            v = k[1, 1] * (target_xyz[1] / z) + k[1, 2]
            img_h = hw[0].clamp_min(1.0)
            img_w = hw[1].clamp_min(1.0)
            tx = torch.floor(u / img_w * tw).long()
            ty = torch.floor(v / img_h * th).long()
            valid = (z_raw > 1e-4) & (tx >= 0) & (tx < tw) & (ty >= 0) & (ty < th)
            idx = ty.clamp(0, th - 1) * tw + tx.clamp(0, tw - 1)
            out = flat_volume.new_zeros((c, th * tw))
            count = flat_volume.new_zeros((1, th * tw))
            if valid.any():
                valid_idx = idx[valid]
                valid_depth = z_raw[valid]
                depth = flat_volume.new_full((th * tw,), float("inf"))
                depth.scatter_reduce_(0, valid_idx, valid_depth, reduce='amin', include_self=True)
                visible = valid & (z_raw <= depth[idx] + 1e-3)
                visible_idx = idx[visible]
                out.scatter_add_(1, visible_idx.unsqueeze(0).expand(c, -1), flat_volume[bi, :, visible])
                count.scatter_add_(
                    1,
                    visible_idx.unsqueeze(0),
                    torch.ones((1, int(visible.sum())), device=volume.device, dtype=volume.dtype),
                )
            out = out / count.clamp_min(1.0)
            projected.append(out.reshape(c, th, tw))
            valid_masks.append(count.reshape(1, th, tw) > 0)
        return torch.stack(projected, dim=0), torch.stack(valid_masks, dim=0)

    def _geometric_latent_view_loss(self, volume, img_metas, target_teacher):
        teacher = self._teacher_to_2d(target_teacher)
        if teacher is None:
            return None
        projected, valid = self._project_volume_to_tokens(volume, img_metas, teacher)
        if projected is None or valid is None:
            return None
        if self.latent_view_projector is None:
            return None
        pred = self.latent_view_projector(projected.float())
        teacher = teacher.to(pred.device).detach().float()
        if pred.shape[-2:] != teacher.shape[-2:]:
            pred = F.interpolate(pred, size=teacher.shape[-2:], mode='bilinear', align_corners=False)
            valid = F.interpolate(valid.float(), size=teacher.shape[-2:], mode='nearest').bool()
        if pred.shape[1] != teacher.shape[1]:
            raise RuntimeError(
                f"V-JEPA target channel count {teacher.shape[1]} does not match projected "
                f"latent channel count {pred.shape[1]}."
            )
        loss_map = 1.0 - F.cosine_similarity(pred, teacher, dim=1, eps=1e-6)
        mask = valid.squeeze(1)
        if mask.any():
            return loss_map[mask].mean()
        return loss_map.mean() * 0.0

    def _m4_valid_mask(self, gt_occ, spatial_shape, device):
        if gt_occ is None:
            return torch.ones((spatial_shape[0], 1, *spatial_shape[1:]), device=device, dtype=torch.bool)
        labels = gt_occ
        if isinstance(labels, (list, tuple)):
            labels = labels[0]
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=device)
        labels = labels.to(device=device)
        if labels.ndim == 3:
            labels = labels.unsqueeze(0)
        if labels.ndim == 5 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 4:
            raise RuntimeError(f"M4 expected gt_occ as [B,X,Y,Z], got shape {tuple(labels.shape)}")
        labels = labels.float().unsqueeze(1)
        if tuple(labels.shape[-3:]) != tuple(spatial_shape[1:]):
            labels = F.interpolate(labels, size=spatial_shape[1:], mode='nearest')
        labels = labels.long()
        return (labels != self.m4_ignore_index) & (labels != -1)

    def _m4_random_mask(self, valid_mask, ratio):
        return (torch.rand(valid_mask.shape, device=valid_mask.device) < ratio) & valid_mask

    def _m4_randint(self, low, high, device):
        low = int(low)
        high = int(high)
        if high <= low:
            return low
        return int(torch.randint(low, high + 1, (), device=device).item())

    def _m4_block_mask(self, valid_mask):
        # Feature volume convention is [B, C, X, Y, Z]. Blocks are cuboids in
        # this encoded voxel space and are intersected with valid labels.
        b, _, x, y, z = valid_mask.shape
        mask = torch.zeros_like(valid_mask)
        for bi in range(b):
            n = self._m4_randint(self.m4_block_num_min, self.m4_block_num_max, valid_mask.device)
            for _ in range(n):
                sx = self._m4_randint(self.m4_block_size_min[0], self.m4_block_size_max[0], valid_mask.device)
                sy = self._m4_randint(self.m4_block_size_min[1], self.m4_block_size_max[1], valid_mask.device)
                sz = self._m4_randint(self.m4_block_size_min[2], self.m4_block_size_max[2], valid_mask.device)
                sx, sy, sz = min(sx, x), min(sy, y), min(sz, z)
                x0 = self._m4_randint(0, max(0, x - sx), valid_mask.device)
                y0 = self._m4_randint(0, max(0, y - sy), valid_mask.device)
                z0 = self._m4_randint(0, max(0, z - sz), valid_mask.device)
                mask[bi, :, x0:x0 + sx, y0:y0 + sy, z0:z0 + sz] = True
        return mask & valid_mask

    def _m4_column_mask(self, valid_mask):
        # OccuFly/CGFormer uses [X,Y,Z] spatial order; Z is treated as vertical.
        # A selected BEV patch in X/Y masks every Z cell in that column.
        b, _, x, y, _ = valid_mask.shape
        mask = torch.zeros_like(valid_mask)
        for bi in range(b):
            n = self._m4_randint(self.m4_column_num_min, self.m4_column_num_max, valid_mask.device)
            for _ in range(n):
                sx = self._m4_randint(self.m4_column_size_min[0], self.m4_column_size_max[0], valid_mask.device)
                sy = self._m4_randint(self.m4_column_size_min[1], self.m4_column_size_max[1], valid_mask.device)
                sx, sy = min(sx, x), min(sy, y)
                x0 = self._m4_randint(0, max(0, x - sx), valid_mask.device)
                y0 = self._m4_randint(0, max(0, y - sy), valid_mask.device)
                mask[bi, :, x0:x0 + sx, y0:y0 + sy, :] = True
        return mask & valid_mask

    def _m4_frustum_mask(self, valid_mask):
        # Fallback BEV wedge approximation. True camera frusta are not available
        # at this encoded feature stage, so wedges originate near the front
        # center of the X/Y plane and extend through all vertical Z cells.
        b, _, x, y, z = valid_mask.shape
        yy, xx = torch.meshgrid(
            torch.arange(y, device=valid_mask.device, dtype=torch.float32),
            torch.arange(x, device=valid_mask.device, dtype=torch.float32),
            indexing='ij',
        )
        origin_x = (x - 1) * 0.5
        origin_y = 0.0
        dx = xx - origin_x
        dy = yy - origin_y
        angle = torch.atan2(dy, dx)
        depth = torch.sqrt(dx * dx + dy * dy)
        depth = depth / depth.max().clamp_min(1.0)
        mask = torch.zeros_like(valid_mask)
        for bi in range(b):
            n = self._m4_randint(self.m4_frustum_num_min, self.m4_frustum_num_max, valid_mask.device)
            for _ in range(n):
                center = (torch.rand((), device=valid_mask.device) * 2.0 - 1.0) * torch.pi
                width_deg = torch.empty((), device=valid_mask.device).uniform_(
                    self.m4_frustum_angle_width_min, self.m4_frustum_angle_width_max)
                width = width_deg * torch.pi / 180.0
                d0 = torch.empty((), device=valid_mask.device).uniform_(0.0, self.m4_frustum_depth_min)
                d1 = torch.empty((), device=valid_mask.device).uniform_(self.m4_frustum_depth_min, self.m4_frustum_depth_max)
                delta = torch.atan2(torch.sin(angle - center), torch.cos(angle - center)).abs()
                bev = (delta <= width * 0.5) & (depth >= d0) & (depth <= d1)
                mask[bi, 0] |= bev.t().unsqueeze(-1).expand(x, y, z)
        return mask & valid_mask

    def _m4_clamp_mask(self, mask, valid_mask):
        b = mask.shape[0]
        out = torch.zeros_like(mask)
        max_ratio = max(0.0, min(1.0, self.m4_max_mask_ratio))
        for bi in range(b):
            valid = valid_mask[bi]
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                continue
            flat_mask = (mask[bi] & valid).flatten()
            flat_valid = valid.flatten()
            max_count = max(self.m4_min_masked_voxels, int(valid_count * max_ratio))
            count = int(flat_mask.sum().item())
            if count > max_count:
                idx = flat_mask.nonzero(as_tuple=False).flatten()
                keep = idx[torch.randperm(idx.numel(), device=idx.device)[:max_count]]
                flat_mask[:] = False
                flat_mask[keep] = True
            elif count < self.m4_min_masked_voxels:
                idx = flat_valid.nonzero(as_tuple=False).flatten()
                take = idx[torch.randperm(idx.numel(), device=idx.device)[:self.m4_min_masked_voxels]]
                flat_mask[take] = True
            out[bi] = flat_mask.reshape_as(out[bi])
        return out & valid_mask

    def _build_m4_mask(self, valid_mask):
        mask_type = self.m4_mask_type
        if mask_type in ('random', 'm4-random'):
            mask = self._m4_random_mask(valid_mask, self.m4_random_ratio)
            comps = {'random': mask}
        elif mask_type == 'block':
            mask = self._m4_block_mask(valid_mask)
            comps = {'block': mask}
        elif mask_type == 'column':
            mask = self._m4_column_mask(valid_mask)
            comps = {'column': mask}
        elif mask_type == 'frustum':
            mask = self._m4_frustum_mask(valid_mask)
            comps = {'frustum': mask}
        elif mask_type == 'mixed':
            total = max(0.0, min(self.m4_total_mask_ratio, self.m4_max_mask_ratio))
            weights = {
                'random': max(0.0, self.m4_mix_random_ratio),
                'block': max(0.0, self.m4_mix_block_ratio),
                'column': max(0.0, self.m4_mix_column_ratio),
                'frustum': max(0.0, self.m4_mix_frustum_ratio),
            }
            denom = sum(weights.values()) or 1.0
            random = self._m4_random_mask(valid_mask, total * weights['random'] / denom)
            block = self._m4_block_mask(valid_mask) if weights['block'] > 0 else torch.zeros_like(valid_mask)
            column = self._m4_column_mask(valid_mask) if weights['column'] > 0 else torch.zeros_like(valid_mask)
            frustum = self._m4_frustum_mask(valid_mask) if weights['frustum'] > 0 else torch.zeros_like(valid_mask)
            mask = random | block | column | frustum
            comps = {'random': random, 'block': block, 'column': column, 'frustum': frustum}
        else:
            raise RuntimeError(f"Unsupported M4 mask type: {mask_type}")
        return self._m4_clamp_mask(mask, valid_mask), comps

    def _masked_3d_loss(self, volume, gt_occ=None):
        if volume is None or volume.ndim != 5:
            return None
        valid_mask = self._m4_valid_mask(gt_occ, (volume.shape[0], volume.shape[2], volume.shape[3], volume.shape[4]), volume.device)
        mask, components = self._build_m4_mask(valid_mask)
        if not mask.any():
            return volume.sum() * 0.0
        if self.masked_3d_head is None:
            return None
        if volume.shape[1] != self.masked_3d_channels:
            raise RuntimeError(
                f"Masked-3D channel count {volume.shape[1]} does not match configured "
                f"{self.masked_3d_channels}."
            )
        pred = self.masked_3d_head(volume.masked_fill(mask, 0.0))
        target = volume.detach()
        loss = (pred - target).abs().mean(dim=1, keepdim=True)[mask].mean()
        valid_count = valid_mask.float().sum().clamp_min(1.0)
        self._last_m4_stats = {
            'mask_ratio_actual': (mask.float().sum() / valid_count).detach(),
            'num_masked_voxels': mask.float().sum().detach(),
            'random_component_ratio': (components.get('random', torch.zeros_like(mask)).float().sum() / valid_count).detach(),
            'block_component_ratio': (components.get('block', torch.zeros_like(mask)).float().sum() / valid_count).detach(),
            'column_component_ratio': (components.get('column', torch.zeros_like(mask)).float().sum() / valid_count).detach(),
            'frustum_component_ratio': (components.get('frustum', torch.zeros_like(mask)).float().sum() / valid_count).detach(),
        }
        return loss

    def _batch_consistency_loss(self, logits):
        if logits is None or logits.ndim != 5 or logits.shape[0] < 2:
            return logits.sum() * 0.0
        prob = logits.softmax(dim=1)
        peer = prob.roll(shifts=1, dims=0).detach()
        conf = torch.maximum(prob.max(dim=1, keepdim=True).values, peer.max(dim=1, keepdim=True).values)
        mask = conf >= self.output_consistency_confidence
        mix = 0.5 * (prob + peer)
        loss_map = 0.5 * (
            F.kl_div(mix.clamp_min(1e-6).log(), prob.clamp_min(1e-6), reduction='none') +
            F.kl_div(mix.clamp_min(1e-6).log(), peer.clamp_min(1e-6), reduction='none')
        ).sum(dim=1, keepdim=True)
        if mask.any():
            return loss_map[mask].mean()
        return loss_map.mean() * 0.0
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, context, depth_out = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)
        losses.update(self._depth_teacher_losses(data_dict, img_metas, depth_out))

        if self.use_vjepa_distill and self.lambda_vjepa_distill > 0:
            teacher = self._teacher_to_2d(data_dict.get('vjepa_teacher'))
            if teacher is None:
                if self.require_vjepa_teacher:
                    raise RuntimeError("M2/M3/M5/M6 requested V-JEPA distillation but batch has no vjepa_teacher. Build/cache features first.")
            else:
                distill = self._cosine_map_loss(context, teacher.to(context.device))
                if distill is not None:
                    losses['loss_vjepa_distill'] = self.lambda_vjepa_distill * distill

        if self.use_latent_view and self.lambda_latent_view > 0:
            latent_view = self._geometric_latent_view_loss(
                img_voxel_feats, img_metas, data_dict.get('target_vjepa_teacher'))
            if latent_view is None:
                raise RuntimeError(
                    "M3 latent-view requested full geometric projection, but the batch lacks "
                    "target_pose/target_cam_intrinsic/target_vjepa_teacher. Check paired OccuFly loading."
                )
            if latent_view is not None:
                losses['loss_latent_view'] = self.lambda_latent_view * latent_view

        if self.use_masked_3d and self.lambda_masked_3d > 0:
            masked = self._masked_3d_loss(voxel_feats_enc[0], gt_occ)
            if masked is not None:
                losses['loss_masked_3d'] = self.lambda_masked_3d * masked
                for key, value in getattr(self, '_last_m4_stats', {}).items():
                    losses[f'm4/{key}'] = value
                losses['m4/mask_type_id'] = voxel_feats_enc[0].new_tensor(
                    {'random': 0, 'm4-random': 0, 'block': 1, 'column': 2, 'frustum': 3, 'mixed': 4}.get(self.m4_mask_type, -1))

        if self.use_output_consistency and self.lambda_output_consistency > 0:
            losses['loss_output_consistency'] = self.lambda_output_consistency * self._batch_consistency_loss(output['output_voxels'])

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ,
            'depth_logits': depth_out.get('depth_logits'),
            'depth_probs': depth_out.get('depth_probs'),
            'depth_pred': depth_out.get('depth_pred'),
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, context, depth_out = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ,
            'depth_logits': depth_out.get('depth_logits'),
            'depth_probs': depth_out.get('depth_probs'),
            'depth_pred': depth_out.get('depth_pred'),
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
