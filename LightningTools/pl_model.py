import os
import csv
import json
import torch
import numpy as np
import torch.distributed as dist
from datetime import datetime, timezone
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from .metric import SSCMetrics
from mmdet3d.models import build_model
from .utils import get_inv_map
from mmcv.runner.checkpoint import load_checkpoint


class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        if 'load_from' in config:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])
        self.val_metrics_by_altitude = {
            int(altitude): SSCMetrics(config['num_class'])
            for altitude in getattr(config, 'altitudes', [30, 40, 50])
        }
        self.save_path = config['save_path']
        self.results_out_root = os.environ.get('CGFORMER_RESULTS_OUT_ROOT')
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']

    def _metric_stats(self, metric):
        values = np.concatenate([
            np.asarray([
                metric.completion_tp,
                metric.completion_fp,
                metric.completion_fn,
                metric.valid_voxels,
                metric.occupied_gt_voxels,
                metric.occupied_pred_voxels,
                metric.count,
            ], dtype=np.float64),
            metric.tps.astype(np.float64),
            metric.fps.astype(np.float64),
            metric.fns.astype(np.float64),
            metric.gt_voxels.astype(np.float64),
            metric.pred_voxels.astype(np.float64),
            metric.occ_class_fps.astype(np.float64),
            metric.occ_class_fns.astype(np.float64),
        ])
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        values = tensor.detach().cpu().numpy()
        n = self.num_class
        completion_tp, completion_fp, completion_fn = values[:3]
        valid_voxels, occupied_gt_voxels, occupied_pred_voxels, samples = values[3:7]
        offset = 7
        tps = values[offset:offset + n]
        fps = values[offset + n:offset + 2 * n]
        fns = values[offset + 2 * n:offset + 3 * n]
        gt_voxels = values[offset + 3 * n:offset + 4 * n]
        pred_voxels = values[offset + 4 * n:offset + 5 * n]
        occ_class_fps = values[offset + 5 * n:offset + 6 * n]
        occ_class_fns = values[offset + 6 * n:offset + 7 * n]
        class_denoms = tps + fps + fns
        class_iou = np.divide(
            tps,
            class_denoms,
            out=np.full_like(tps, np.nan, dtype=np.float64),
            where=class_denoms != 0,
        )
        denom = completion_tp + completion_fp + completion_fn
        precision_denom = completion_tp + completion_fp
        recall_denom = completion_tp + completion_fn
        return {
            "samples": int(round(samples)),
            "valid_voxels": int(valid_voxels),
            "occupied_gt_voxels": int(occupied_gt_voxels),
            "occupied_pred_voxels": int(occupied_pred_voxels),
            "IoU": float("nan" if denom == 0 else completion_tp / denom),
            "mIoU": float(np.nanmean(class_iou[1:])),
            "Precision": float("nan" if precision_denom == 0 else completion_tp / precision_denom),
            "Recall": float("nan" if recall_denom == 0 else completion_tp / recall_denom),
            "class_iou": {
                name: float(class_iou[idx])
                for idx, name in enumerate(self.class_names)
            },
            "class_rows": self._class_rows(
                tps, fps, fns, gt_voxels, pred_voxels, occ_class_fps, occ_class_fns),
        }

    def _class_rows(self, tps, fps, fns, gt_voxels, pred_voxels, occ_class_fps, occ_class_fns):
        rows = []
        for idx in range(self.num_class):
            semantic_denom = tps[idx] + fps[idx] + fns[idx]
            if idx == 0:
                class_occ_iou = float("nan")
            else:
                class_occ_denom = tps[idx] + occ_class_fps[idx] + occ_class_fns[idx]
                class_occ_iou = float("nan" if class_occ_denom == 0 else tps[idx] / class_occ_denom)
            rows.append({
                "class_id": idx,
                "class_name": self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                "tp": int(tps[idx]),
                "fp": int(fps[idx]),
                "fn": int(fns[idx]),
                "gt_voxels": int(gt_voxels[idx]),
                "pred_voxels": int(pred_voxels[idx]),
                "semantic_iou": float("nan" if semantic_denom == 0 else tps[idx] / semantic_denom),
                "class_occupancy_iou": class_occ_iou,
            })
        return rows

    def _csv_value(self, value):
        if isinstance(value, float) and np.isnan(value):
            return "nan"
        return value

    def _write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: self._csv_value(row.get(key, "")) for key in fieldnames})

    def _write_metrics_readme(self, path):
        with open(path, "w") as f:
            f.write(
                "OccuFly SSC metric diagnostics\n"
                "==============================\n\n"
                "run_summary.csv contains one row per evaluated split/checkpoint with only run-level counts and aggregate metrics.\n"
                "class_metrics.csv contains the complete per-class diagnostic table: TP, FP, FN, GT voxel count, prediction voxel count, semantic IoU, and class occupancy IoU.\n"
                "label_mapping_check.csv contains only label/config sanity checks; it intentionally does not repeat performance metrics.\n"
                "visualization_index.csv is written only by visualization exporters and should contain file paths only, not metrics.\n\n"
                "Label rules: label 0 is empty. Label 255 is invalid/unknown and ignored in all metrics. Valid occupied semantic classes are labels 1..C.\n"
                "Semantic mIoU is the mean of per-class semantic IoU over valid occupied classes only; empty and ignore labels are excluded.\n"
                "Completion IoU/Precision/Recall are binary occupied-vs-empty metrics where occupied means any valid semantic class 1..C.\n"
                "Undefined divisions are written as nan, for example when a class has no union or precision has no predicted positives.\n\n"
                "class_metrics.csv is the only per-class diagnostic table, replacing redundant per-class IoU, prediction histogram, and per-class occupancy IoU files.\n")

    def _metadata_value(self, img_metas, key, default=None):
        value = img_metas.get(key, default)
        while isinstance(value, (list, tuple)) and len(value) > 0:
            value = value[0]
        if hasattr(value, "item"):
            value = value.item()
        return value

    def _write_epoch_results(self, prefix, overall, by_altitude):
        if not self.results_out_root or not self.trainer.is_global_zero:
            return
        split = "val" if prefix == "val" else prefix
        out_root = os.path.abspath(self.results_out_root)
        epoch = int(self.current_epoch)
        checkpoint = f"epoch_{epoch:04d}_step_{int(self.global_step):08d}"
        timestamp = datetime.now(timezone.utc).isoformat()
        epoch_dir = os.path.join(out_root, "per_epoch", f"epoch_{epoch:04d}")
        os.makedirs(epoch_dir, exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": int(self.global_step),
            "split": split,
            **overall,
            "by_altitude": by_altitude,
        }
        by_alt_payload = {
            str(altitude): {"epoch": epoch, "split": split, **stats}
            for altitude, stats in by_altitude.items()
        }
        targets = [
            (os.path.join(epoch_dir, "metrics.json"), payload),
            (os.path.join(out_root, "metrics.json"), payload),
            (os.path.join(epoch_dir, "metrics-by-altitude.json"), by_alt_payload),
            (os.path.join(out_root, "metrics-by-altitude.json"), by_alt_payload),
        ]
        for path, data in targets:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
        for path in [
            os.path.join(epoch_dir, "run_summary.csv"),
            os.path.join(out_root, "run_summary.csv"),
        ]:
            self._write_csv(path, [{
                "split": split,
                "checkpoint": checkpoint,
                "num_samples": overall["samples"],
                "num_valid_voxels": overall["valid_voxels"],
                "num_occupied_gt_voxels": overall["occupied_gt_voxels"],
                "num_occupied_pred_voxels": overall["occupied_pred_voxels"],
                "completion_iou": overall["IoU"],
                "completion_precision": overall["Precision"],
                "completion_recall": overall["Recall"],
                "semantic_miou": overall["mIoU"],
                "timestamp": timestamp,
            }], [
                "split", "checkpoint", "num_samples", "num_valid_voxels",
                "num_occupied_gt_voxels", "num_occupied_pred_voxels",
                "completion_iou", "completion_precision", "completion_recall",
                "semantic_miou", "timestamp",
            ])
        for path in [
            os.path.join(epoch_dir, "class_metrics.csv"),
            os.path.join(out_root, "class_metrics.csv"),
        ]:
            class_rows = [{"split": split, **row} for row in overall["class_rows"]]
            self._write_csv(path, class_rows, [
                "split", "class_id", "class_name", "tp", "fp", "fn",
                "gt_voxels", "pred_voxels", "semantic_iou", "class_occupancy_iou",
            ])
        label_row = {
            "checkpoint": checkpoint,
            "empty_label_id": 0,
            "ignore_label_id": 255,
            "valid_class_ids": " ".join(str(x) for x in range(1, self.num_class)),
            "prediction_channel_count": self.num_class,
            "num_metric_classes": self.num_class,
            "ignore_255_in_metrics": True,
            "exclude_empty_from_semantic_miou": True,
            "class_id_channel_alignment_ok": len(self.class_names) == self.num_class and self.class_names[0] == "empty",
            "warning": "",
        }
        if not label_row["class_id_channel_alignment_ok"]:
            label_row["warning"] = "class name count or empty label alignment does not match metric classes"
        for path in [
            os.path.join(epoch_dir, "label_mapping_check.csv"),
            os.path.join(out_root, "label_mapping_check.csv"),
        ]:
            self._write_csv(path, [label_row], [
                "checkpoint", "empty_label_id", "ignore_label_id", "valid_class_ids",
                "prediction_channel_count", "num_metric_classes", "ignore_255_in_metrics",
                "exclude_empty_from_semantic_miou", "class_id_channel_alignment_ok", "warning",
            ])
        for path in [
            os.path.join(epoch_dir, "README_metrics.txt"),
            os.path.join(out_root, "README_metrics.txt"),
        ]:
            self._write_metrics_readme(path)
    
    def forward(self, data_dict):
        return self.model(data_dict)
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
            
            self.train_metrics.add_batch(pred, gt_occ)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.val_metrics.add_batch(pred, gt_occ)
            altitude = self._metadata_value(batch['img_metas'], 'altitude')
            if altitude is not None:
                altitude = int(altitude)
                if altitude not in self.val_metrics_by_altitude:
                    self.val_metrics_by_altitude[altitude] = SSCMetrics(self.num_class)
                self.val_metrics_by_altitude[altitude].add_batch(pred, gt_occ)
    
    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            if prefix == "val":
                overall = self._metric_stats(metric)
                by_altitude = {
                    int(altitude): self._metric_stats(alt_metric)
                    for altitude, alt_metric in self.val_metrics_by_altitude.items()
                }
                self._write_epoch_results(prefix, overall, by_altitude)
            metric.reset()
        for metric in self.val_metrics_by_altitude.values():
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        gt_occ = output_dict['gt_occ']
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                print('\n save to {}'.format(save_file))
            
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
    
    def test_epoch_end(self, outputs):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                print(name + ":", iou)

            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()
