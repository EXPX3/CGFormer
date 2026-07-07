import math

import torch
import torch.nn.functional as F


def _as_batched_k(K, batch, device, dtype):
    if K is None:
        K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
    elif not torch.is_tensor(K):
        K = torch.as_tensor(K, device=device, dtype=dtype)
    else:
        K = K.to(device=device, dtype=dtype)
    if K.ndim == 4:
        K = K[:, 0]
    if K.ndim == 2:
        K = K.unsqueeze(0)
    if K.shape[-1] == 4:
        K = K[..., :3, :3]
    if K.shape[0] != batch:
        K = K[:1].repeat(batch, 1, 1)
    return K


def _as_altitude(altitude, batch, device, dtype):
    if altitude is None:
        altitude = torch.ones(batch, device=device, dtype=dtype)
    elif not torch.is_tensor(altitude):
        altitude = torch.as_tensor(altitude, device=device, dtype=dtype)
    else:
        altitude = altitude.to(device=device, dtype=dtype)
    altitude = altitude.flatten()
    if altitude.numel() == 1 and batch > 1:
        altitude = altitude.repeat(batch)
    return altitude[:batch].clamp_min(1.0e-6)


def make_pixel_rays(K, height, width, device=None, dtype=None):
    K = K.to(device=device or K.device, dtype=dtype or K.dtype)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype),
        torch.arange(width, device=K.device, dtype=K.dtype),
        indexing="ij",
    )
    fx = K[:, 0, 0].view(-1, 1, 1).clamp_min(1.0e-6)
    fy = K[:, 1, 1].view(-1, 1, 1).clamp_min(1.0e-6)
    cx = K[:, 0, 2].view(-1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1)
    x = (xs.view(1, height, width) - cx) / fx
    y = (ys.view(1, height, width) - cy) / fy
    z = torch.ones_like(x)
    return torch.stack([x, y, z], dim=1)


def ground_plane_depth_from_altitude(rays, altitude):
    down = rays[:, 1].abs().clamp_min(1.0e-3)
    return altitude.view(-1, 1, 1, 1) / down.unsqueeze(1)


def altitude_depth_bins(K, altitude, height, width, num_bins, min_log_residual=-2.0, max_log_residual=2.0):
    rays = make_pixel_rays(K, height, width, device=K.device, dtype=K.dtype)
    d0 = ground_plane_depth_from_altitude(rays, altitude).clamp_min(1.0e-6)
    log_bins = torch.linspace(min_log_residual, max_log_residual, num_bins, device=K.device, dtype=K.dtype)
    return d0 * torch.exp(log_bins.view(1, num_bins, 1, 1))


def metric_depth_to_altitude_soft_distribution(
    metric_depth,
    K,
    altitude,
    num_depth_bins,
    min_log_residual=-2.0,
    max_log_residual=2.0,
    valid_mask=None,
    sigma_log=0.10,
    target_type="gaussian_log_depth",
):
    if metric_depth.ndim == 3:
        metric_depth = metric_depth.unsqueeze(1)
    batch, _, height, width = metric_depth.shape
    K = _as_batched_k(K, batch, metric_depth.device, metric_depth.dtype)
    altitude = _as_altitude(altitude, batch, metric_depth.device, metric_depth.dtype)
    rays = make_pixel_rays(K, height, width, device=metric_depth.device, dtype=metric_depth.dtype)
    d0 = ground_plane_depth_from_altitude(rays, altitude).clamp_min(1.0e-6)
    valid = torch.isfinite(metric_depth) & (metric_depth > 0)
    if valid_mask is not None:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape[-2:] != (height, width):
            valid_mask = F.interpolate(valid_mask.float(), size=(height, width), mode="nearest").bool()
        valid = valid & valid_mask.to(metric_depth.device).bool()
    log_residual = torch.log(metric_depth.clamp_min(1.0e-6) / d0)
    bins = torch.linspace(min_log_residual, max_log_residual, num_depth_bins, device=metric_depth.device, dtype=metric_depth.dtype)
    distances = log_residual - bins.view(1, num_depth_bins, 1, 1)
    if target_type == "gaussian_log_depth":
        target = torch.exp(-0.5 * (distances / max(float(sigma_log), 1.0e-6)).pow(2))
    elif target_type == "triangular_log_depth":
        width_log = (max_log_residual - min_log_residual) / max(num_depth_bins - 1, 1)
        target = (1.0 - distances.abs() / max(width_log, 1.0e-6)).clamp_min(0.0)
    elif target_type == "nearest_bin_ce":
        idx = distances.abs().argmin(dim=1, keepdim=True)
        target = torch.zeros(batch, num_depth_bins, height, width, device=metric_depth.device, dtype=metric_depth.dtype)
        target.scatter_(1, idx, 1.0)
    else:
        raise ValueError("Unknown teacher depth soft target: %s" % target_type)
    target = target * valid.to(metric_depth.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return target, valid


def _resize_teacher(teacher, logits):
    if teacher is None:
        return None
    if teacher.ndim == 2:
        teacher = teacher.unsqueeze(0).unsqueeze(0)
    elif teacher.ndim == 3:
        teacher = teacher.unsqueeze(1)
    if teacher.ndim == 5 and teacher.shape[1] == 1:
        teacher = teacher[:, 0]
    teacher = teacher.to(device=logits.device, dtype=logits.dtype)
    if teacher.shape[-2:] != logits.shape[-2:]:
        teacher = F.interpolate(teacher, size=logits.shape[-2:], mode="nearest")
    return teacher


def _terms(logits, bins, teacher, K, altitude, valid_mask, confidence, lambda_kl, lambda_l1, sigma_log, target_type):
    teacher = _resize_teacher(teacher, logits)
    target, valid = metric_depth_to_altitude_soft_distribution(
        teacher, K, altitude, logits.shape[1],
        valid_mask=valid_mask, sigma_log=sigma_log, target_type=target_type,
    )
    weight = valid.to(logits.dtype)
    if confidence is not None:
        if confidence.ndim == 3:
            confidence = confidence.unsqueeze(1)
        confidence = confidence.to(device=logits.device, dtype=logits.dtype)
        if confidence.shape[-2:] != logits.shape[-2:]:
            confidence = F.interpolate(confidence, size=logits.shape[-2:], mode="nearest")
        weight = weight * confidence.clamp_min(0.0)
    probs = logits.softmax(dim=1)
    expected = (probs * bins).sum(dim=1, keepdim=True)
    denom = weight.sum().clamp_min(1.0)
    kl = (-(target * logits.log_softmax(dim=1)).sum(dim=1, keepdim=True) * weight).sum() / denom
    l1 = ((expected - teacher).abs() * weight).sum() / denom
    rmse = ((((expected - teacher) ** 2) * weight).sum() / denom).sqrt()
    return lambda_kl * kl, lambda_l1 * l1, valid.float().mean().detach(), rmse.detach()


def altitude_depth_teacher_loss(
    depth_logits,
    K,
    altitude,
    gt_metric_depth=None,
    vjepa_teacher_depth=None,
    gt_valid_mask=None,
    vjepa_valid_mask=None,
    vjepa_teacher_confidence=None,
    lambda_depth_gt_kl=0.5,
    lambda_depth_gt_l1=0.1,
    lambda_depth_vjepa_kl=0.3,
    lambda_depth_vjepa_l1=0.05,
    sigma_log=0.10,
    target_type="gaussian_log_depth",
):
    batch, _, height, width = depth_logits.shape
    K = _as_batched_k(K, batch, depth_logits.device, depth_logits.dtype)
    altitude = _as_altitude(altitude, batch, depth_logits.device, depth_logits.dtype)
    bins = altitude_depth_bins(K, altitude, height, width, depth_logits.shape[1])
    zero = depth_logits.sum() * 0.0
    out = {
        "loss_depth_gt_kl": zero,
        "loss_depth_gt_l1": zero,
        "loss_depth_vjepa_kl": zero,
        "loss_depth_vjepa_l1": zero,
        "gt_depth_valid_ratio": zero.detach(),
        "vjepa_depth_valid_ratio": zero.detach(),
        "depth_rmse_gt": zero.detach(),
        "depth_rmse_vjepa": zero.detach(),
    }
    if gt_metric_depth is not None and (lambda_depth_gt_kl or lambda_depth_gt_l1):
        kl, l1, ratio, rmse = _terms(depth_logits, bins, gt_metric_depth, K, altitude, gt_valid_mask, None,
                                     lambda_depth_gt_kl, lambda_depth_gt_l1, sigma_log, target_type)
        out.update(loss_depth_gt_kl=kl, loss_depth_gt_l1=l1, gt_depth_valid_ratio=ratio, depth_rmse_gt=rmse)
    if vjepa_teacher_depth is not None and (lambda_depth_vjepa_kl or lambda_depth_vjepa_l1):
        kl, l1, ratio, rmse = _terms(depth_logits, bins, vjepa_teacher_depth, K, altitude, vjepa_valid_mask,
                                     vjepa_teacher_confidence, lambda_depth_vjepa_kl, lambda_depth_vjepa_l1,
                                     sigma_log, target_type)
        out.update(loss_depth_vjepa_kl=kl, loss_depth_vjepa_l1=l1, vjepa_depth_valid_ratio=ratio,
                   depth_rmse_vjepa=rmse)
    return out
