"""Compatibility shims for 3LC + recent Ultralytics versions."""

from __future__ import annotations


def apply_ultralytics_83_compat() -> None:
    """
    Normalize dict-style detection predictions to the tensor format expected by
    older 3LC validator code.
    """

    import torch
    from tlc_ultralytics.detect import validator as det_val

    if getattr(det_val.TLCDetectionValidator, "_ua_detrac_compat_patched", False):
        return

    original = det_val.TLCDetectionValidator._process_detection_predictions

    def pred_dict_to_tensor(predictions):
        if isinstance(predictions, torch.Tensor):
            return predictions
        bboxes = predictions["bboxes"]
        conf = predictions["conf"]
        cls = predictions["cls"]
        if bboxes.shape[0] == 0:
            return bboxes.new_empty((0, 6))
        return torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1).float()], dim=1)

    def patched(self, preds, batch):
        from tlc_ultralytics.detect.utils import construct_bbox_struct
        from ultralytics.utils import metrics, ops

        predicted_boxes = []
        for index, predictions in enumerate(preds):
            predictions = pred_dict_to_tensor(predictions)
            ori_shape = batch["ori_shape"][index]
            resized_shape = batch["resized_shape"][index]
            ratio_pad = batch["ratio_pad"][index]
            height, width = ori_shape

            if len(predictions) == 0:
                predicted_boxes.append(
                    construct_bbox_struct([], image_width=width, image_height=height)
                )
                continue

            predictions = predictions.clone()
            predictions = predictions[predictions[:, 4] > self._settings.conf_thres]
            predictions = predictions[
                predictions[:, 4].argsort(descending=True)[: self._settings.max_det]
            ]

            pred_box = predictions[:, :4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)

            prepared_batch = self._prepare_batch(index, batch)
            gt_bbox = prepared_batch.get("bbox", prepared_batch.get("bboxes"))
            if gt_bbox is not None and gt_bbox.shape[0]:
                ious = metrics.box_iou(gt_bbox, pred_scaled)
                box_ious = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0]

            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)
            conf = predictions[:, 4].cpu().tolist()
            pred_cls = predictions[:, 5].cpu().tolist()

            annotations = [
                {
                    "score": conf[prediction_index],
                    "category_id": self.data["range_to_3lc_class"][
                        int(pred_cls[prediction_index])
                    ],
                    "bbox": pred_xywh[prediction_index, :].cpu().tolist(),
                    "iou": box_ious[prediction_index],
                }
                for prediction_index in range(len(predictions))
            ]

            predicted_boxes.append(
                construct_bbox_struct(
                    annotations,
                    image_width=width,
                    image_height=height,
                )
            )

        return predicted_boxes

    det_val.TLCDetectionValidator._process_detection_predictions = patched
    det_val.TLCDetectionValidator._ua_detrac_compat_patched = True
    det_val.TLCDetectionValidator._ua_detrac_compat_original = original

