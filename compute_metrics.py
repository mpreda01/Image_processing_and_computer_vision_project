import numpy as np
from skimage.draw import polygon2mask


def iou_mask(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def polygon_to_mask(points, shape):
    """
    Converts polygon points to a binary mask.
    Args:
        points (list of tuples): Polygon points in (x, y) format
        shape (tuple): Image shape (height, width)
    Returns:
        np.ndarray: Boolean mask
    """
    poly = np.array(points)
    mask = polygon2mask(shape, poly)
    return mask.astype(bool)


def evaluate_instance_segmentation(gt, pred, image_shape):
    """
    Evaluates instance segmentation based on average IoU per matched instance.

    Instead of thresholding IoU to count TPs or FPs, we compute IoU directly
    for matched instances, and treat unmatched ones as IoU=0.

    Args:
        gt (list): Ground truth instances with "label" and "points"
        pred (list): Predicted instances with "label" and "points"
        image_shape (tuple): Shape of the image (height, width)

    Returns:
        dict: Per-class average IoU, overall macro average IoU

    Input format example:
        gt = [
            {"label": "model_1", "points": [(x1, y1), (x2, y2), ...]},
            ...
        ]
    """
    per_class_iou = {}

    # For each class, collect GT and Pred masks separately
    class_gt = {}
    class_pred = {}

    for inst in gt:
        class_gt.setdefault(inst["label"], []).append(polygon_to_mask(inst["points"], image_shape))

    for inst in pred:
        class_pred.setdefault(inst["label"], []).append(polygon_to_mask(inst["points"], image_shape))

    all_classes = set(class_gt.keys()) | set(class_pred.keys())

    for cls in all_classes:
        gt_masks = class_gt.get(cls, [])
        pred_masks = class_pred.get(cls, [])

        num_gt = len(gt_masks)
        num_pred = len(pred_masks)

        iou_matrix = np.zeros((num_pred, num_gt))

        # Compute IoU between all pairs of predicted and GT masks
        for i, p_mask in enumerate(pred_masks):
            for j, g_mask in enumerate(gt_masks):
                iou_matrix[i, j] = iou_mask(p_mask, g_mask)

        # Greedy matching: assign best IoU pairs first
        matched_gt = set()
        matched_pred = set()
        ious = []

        while True:
            max_iou = -1
            max_i = max_j = -1

            # Find the best unmatched pair
            for i in range(num_pred):
                if i in matched_pred:
                    continue
                for j in range(num_gt):
                    if j in matched_gt:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i = i
                        max_j = j

            if max_iou == -1:
                break  # No more matches

            # Register match
            matched_pred.add(max_i)
            matched_gt.add(max_j)
            ious.append(max_iou)

        # For unmatched GT or Pred, count IoU=0
        unmatched = (num_gt + num_pred) - 2 * len(ious)
        ious.extend([0.0] * unmatched)

        per_class_iou[cls] = np.mean(ious) if ious else 0.0

    # Compute macro average IoU across all classes
    macro_avg_iou = np.mean(list(per_class_iou.values())) if per_class_iou else 0.0

    return {
        "PerClassIoU": per_class_iou,
        "MacroAvgIoU": macro_avg_iou
    }



def evaluate_instance_counting(gt, pred):
    """
    Evaluates instance detection based only on per-class instance counts.

    Args:
        gt (dict): Ground truth counts per class
        pred (dict): Predicted counts per class

    Returns:
        dict: Per-class precision, recall, F1, and macro-averaged metrics

    Input format example:
        gt = {"model_1": 3, "model_2": 5, ...}
        pred = {"model_1": 2, "model_3": 4, ...}
    """
    classes = set(gt.keys()) | set(pred.keys())
    results = {}

    precisions, recalls, f1s = [], [], []

    for cls in classes:
        gt_count = gt.get(cls, 0)
        pred_count = pred.get(cls, 0)

        tp = min(gt_count, pred_count)
        fn = max(gt_count - pred_count, 0)
        fp = max(pred_count - gt_count, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        results[cls] = {
            "GT": gt_count,
            "Pred": pred_count,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    n = len(classes) or 1
    results["MacroAvg"] = {
        "MacroPrecision": sum(precisions) / n,
        "MacroRecall": sum(recalls) / n,
        "MacroF1": sum(f1s) / n
    }

    return results