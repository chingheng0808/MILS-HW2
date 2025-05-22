import torch
import torch.nn.functional as F
from torchvision.ops import nms
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import JaccardIndex, Accuracy
import numpy as np
import math 

def yolo_v1_NMS(outputs, conf_thres=0.0, iou_thres=0.5, S=32, B=2, C=10, img_w=512, img_h=512):
    """
    Args:
        outputs: Tensor [B, (B*5+C), S, S] (tx, ty, conf 已經是 sigmoid後, tw,th 是 sqrt)
    Returns:
        results: list of dict, each dict: {'boxes': Tensor[N,4], 'labels': Tensor[N], 'scores': Tensor[N]}
    """
    BATCH = outputs.shape[0]
    results = []
    for b in range(BATCH):
        preds = outputs[b].detach().cpu()
        boxes = []
        scores = []
        labels = []
        for row in range(S):      # y
            for col in range(S):  # x
                for slot in range(B):
                    offset = slot * 5
                    bx = float(preds[offset+0, row, col])   # 已經是 sigmoid
                    by = float(preds[offset+1, row, col])   # 已經是 sigmoid
                    bw = float(preds[offset+2, row, col]) ** 2  # sqrt
                    bh = float(preds[offset+3, row, col]) ** 2  # sqrt
                    pconf = float(preds[offset+4, row, col])    # 已經是 sigmoid
                    if pconf < conf_thres:
                        continue
                    # class conf
                    cls_probs = preds[B*5:, row, col]
                    cls_id = int(torch.argmax(cls_probs))
                    cls_score = float(torch.softmax(cls_probs, dim=0)[cls_id])
                    score = pconf * cls_score
                    if score < conf_thres:
                        continue
                    # decode to global
                    cx = (col + bx) / S * img_w
                    cy = (row + by) / S * img_h
                    w = bw * img_w
                    h = bh * img_h
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    # clamp防止出界
                    x1 = max(0, min(img_w-1, x1))
                    y1 = max(0, min(img_h-1, y1))
                    x2 = max(0, min(img_w-1, x2))
                    y2 = max(0, min(img_h-1, y2))
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    labels.append(cls_id)
        if len(boxes) == 0:
            results.append({'boxes': torch.empty((0,4)), 'labels': torch.empty((0,), dtype=torch.long), 'scores': torch.empty((0,))})
            continue
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)
        keep = nms(boxes, scores, iou_thres)
        results.append({'boxes': boxes[keep], 'labels': labels[keep], 'scores': scores[keep]})
    return results

def convert_coco_to_xyxy(targets, img_size=None):
    """
    targets: list of dicts, each with
       'boxes': Tensor[N,4] in [x_min,y_min,w,h] (pixel)
       'labels': Tensor[N]
    returns: list of dicts with
       'boxes': Tensor[N,4] in [x1,y1,x2,y2] (pixel or normalized if img_size is set)
       'labels': Tensor[N] unchanged
    """
    new = []
    for t in targets:
        xywh = t['boxes']
        if xywh.numel() == 0:
            boxes = torch.empty((0,4))
        else:
            x1 = xywh[:,0]
            y1 = xywh[:,1]
            x2 = xywh[:,0] + xywh[:,2]
            y2 = xywh[:,1] + xywh[:,3]
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            # 若要normalize，則加這一行
            if img_size is not None:
                img_w, img_h = img_size
                boxes[:, 0::2] /= img_w
                boxes[:, 1::2] /= img_h
        new.append({'boxes': boxes, 'labels': t['labels']})
    return new



def seg_decode(feat_seg, img_size=(512, 512)):
    seg_logits = F.interpolate(
            feat_seg,
            size=img_size,
            mode='bilinear',
            align_corners=False
        )    
    return seg_logits # [B, C, H, W]

def cls_decode(
    feature_map: torch.Tensor,
    threshold: float = 0.25
) -> torch.Tensor:
    """
    把 (B, C, H, W) 的 feature_map 轉成 (B, C)：
      1. 對所有元素做 sigmoid；
      2. 把小於 threshold 的值設為 0；
      3. 只對 >= threshold 的像素做空間平均；
      4. 輸出 (B, C) 的 tensor（no. of pixels = 0 時回傳 0）。
    """
    # 1) sigmoid 轉機率
    prob = torch.sigmoid(feature_map)              # [B, C, H, W]
    # 2) 產生 mask
    mask = prob >= threshold                       # 同形狀 bool
    # 3) 把低於門檻的值清掉
    prob_thresh = prob * mask.float()              # [B, C, H, W]
    # 4) sum & count
    sums   = prob_thresh.sum(dim=(2,3))            # [B, C]
    counts = mask.sum(dim=(2,3)).clamp(min=1).float()  # [B, C]
    # 5) 只平均大於門檻的那些像素
    logits = sums / counts                         # [B, C]
    return logits

########################### LOSS FUNCTIONS ############################
def encode_yolo_targets(target_list, S=32, B=2, C=10, img_size=(512,512), device='cuda'):
    """
    Args:
        target_list: list of dict, 每個 dict: {'boxes': [N,4] (pixel xywh), 'labels': [N]}
        S: grid size
        B: 每 cell 預測幾個 box
        C: class 數
        img_size: (W, H)
    Returns:
        Tensor [batch, B*5+C, S, S]
    """
    batch_size = len(target_list)
    W, H = img_size
    yolo_t = torch.zeros(batch_size, B*5 + C, S, S, device=device)

    for i, tgt in enumerate(target_list):
        boxes = tgt['boxes']
        labels = tgt['labels']
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu()
        for box, cls in zip(boxes, labels):
            x, y, w, h = box.tolist()  # [x_min, y_min, w, h], pixel
            # 物件中心點 (pixel)
            cx = x + w / 2
            cy = y + h / 2
            # normalize to [0,1]
            cx_norm = cx / W
            cy_norm = cy / H
            w_norm = w / W
            h_norm = h / H
            # 找 cell
            col = min(S-1, int(cx_norm * S))  # x → col
            row = min(S-1, int(cy_norm * S))  # y → row
            # cell 內偏移量 (0~1)
            bx = cx_norm * S - col
            by = cy_norm * S - row
            # sqrt 編碼寬高
            bw = math.sqrt(max(w_norm, 1e-8))
            bh = math.sqrt(max(h_norm, 1e-8))
            # 填 slot
            for slot in range(B):
                if yolo_t[i, slot*5+4, row, col] == 0:
                    off = slot*5
                    yolo_t[i, off+0, row, col] = bx     # sigmoid後的 tx
                    yolo_t[i, off+1, row, col] = by     # sigmoid後的 ty
                    yolo_t[i, off+2, row, col] = bw     # sqrt
                    yolo_t[i, off+3, row, col] = bh     # sqrt
                    yolo_t[i, off+4, row, col] = 1.0    # conf
                    # one-hot class
                    cls = int(cls)
                    if 0 <= cls < C:
                        yolo_t[i, B*5 + cls, row, col] = 1.0
                    else:
                        print(f"Warning: class index {cls} out of range")
                    break
    return yolo_t

class DETLoss(nn.Module):
    def __init__(self, S, B, C, lambda_coord=5.0, lambda_noobj=1.0):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        N, _, S, _ = preds.shape
        p = preds.permute(0,2,3,1).contiguous()
        t = targets.permute(0,2,3,1).contiguous().to(preds.device)
        box_p = p[..., :self.B*5].view(N, S, S, self.B, 5)
        box_t = t[..., :self.B*5].view(N, S, S, self.B, 5)
        cls_p = p[..., self.B*5:]  # [N, S, S, C]
        cls_t = t[..., self.B*5:]  # [N, S, S, C]

        # 解碼：預設是sigmoid(tx/ty/conf)，tw/th直接預測sqrt
        tx_p = box_p[...,0]
        ty_p = box_p[...,1]
        tw_p = box_p[...,2]
        th_p = box_p[...,3]
        conf_p = box_p[...,4]

        tx_t = box_t[...,0]
        ty_t = box_t[...,1]
        tw_t = box_t[...,2]
        th_t = box_t[...,3]
        conf_t = box_t[...,4]

        resp = conf_t[...,0] > 0    # [N, S, S]
        num_resp = resp.sum()

        # 1) coordinate loss (slot 0 only)
        if num_resp > 0:
            coord_loss = (
                F.mse_loss(tx_p[...,0][resp], tx_t[...,0][resp], reduction='sum') +
                F.mse_loss(ty_p[...,0][resp], ty_t[...,0][resp], reduction='sum') +
                F.mse_loss(tw_p[...,0][resp], tw_t[...,0][resp], reduction='sum') +
                F.mse_loss(th_p[...,0][resp], th_t[...,0][resp], reduction='sum')
            )
        else:
            coord_loss = torch.tensor(0., device=preds.device)
        coord_loss = self.lambda_coord * coord_loss

        # 2) confidence loss
        obj_loss   = F.mse_loss(conf_p[...,0][resp], conf_t[...,0][resp], reduction='sum') if num_resp > 0 else torch.tensor(0., device=preds.device)
        noobj_loss = F.mse_loss(conf_p[...,0][~resp], conf_t[...,0][~resp], reduction='sum')
        conf_loss  = obj_loss + self.lambda_noobj * noobj_loss

        # 3) classification loss (for cells with object)
        if num_resp > 0:
            cls_loss = F.mse_loss(cls_p[resp], cls_t[resp], reduction='sum')
        else:
            cls_loss = torch.tensor(0., device=preds.device)

        # 4) combine & normalize
        total_loss = (coord_loss + conf_loss + cls_loss) / (N * S * S)
        return total_loss


class SEGLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(SEGLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] 未經 softmax
        targets: [B, H, W] int64
        """
        targets = targets.long().to(preds.device)
        targets = torch.where(
            (targets >= 1) & (targets <= 20),
            targets - 1,   # 1→0, 2→1, …, 20→19
            targets
        )
        targets = targets.masked_fill(targets == 255, -100)
        return self.criterion(preds, targets)


class CLSLoss(nn.Module):
    def __init__(self):
        super(CLSLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        preds: [B, C],  targets: [B]
        """
        return self.criterion(preds, targets.long().to(preds.device))
    
########################################## Evaluator ##########################################
class Evaluator:
    def __init__(self, num_seg_classes: int=20, num_cls_classes: int=10):
        # Detection mAP (COCO-style)
        self.det_metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
        # Segmentation mIoU
        self.seg_metric = JaccardIndex(task="multiclass", num_classes=num_seg_classes, ignore_index=-100)
        # Classification Top-1 Accuracy
        self.cls_metric = Accuracy(task="multiclass", num_classes=num_cls_classes, top_k=1)
        
        self.results = {}
        self.summary = {}

    def reset(self):
        """Reset all metric states."""
        self.det_metric.reset()
        self.seg_metric.reset()
        self.cls_metric.reset()
        self.results = {}
        self.summary = {}

    def evaluate(self, preds, targets, task: str):
        """
        Update metric states based on the task.
        
        Args:
            preds: 
                - det: list of dicts, each with 'boxes', 'scores', 'labels'
                - seg: Tensor [B, C, H, W] logits or [B, H, W] preds
                - cls: Tensor [B, C] logits
            targets:
                - det: list of dicts, each with 'boxes' (Tensor[N,4]), 'labels' (Tensor[N])
                - seg: Tensor [B, H, W] ground-truth labels
                - cls: Tensor [B] ground-truth labels
            task: one of 'det', 'seg', 'cls'
        """
        if task == 'det':
            # preds and targets are lists of dicts for detection
            self.det_metric.update(preds, targets)
        elif task == 'seg':
            # Convert logits to predicted class labels if necessary
            if preds.ndim == 4:
                preds = torch.argmax(preds, dim=1)
            # preds and targets: [B, H, W]
            self.seg_metric.update(preds, remap_targets(targets))  # remap_targets: 1→0, 2→1, ..., 20→19, 255→-100 (ignore index for loss)targets)
        elif task == 'cls':
            # preds: [B, C], targets: [B]
            self.cls_metric.update(preds, targets)
        else:
            raise ValueError(f"Unknown task '{task}'")

    def compute(self, task: str):
        """
        Compute and return metrics:
            - mAP for detection
            - mIoU for segmentation
            - Top-1 accuracy for classification
        Returns:
            dict with keys 'mAP', 'mIoU', 'Top1'
        """
        # Detection: returns a dict, take 'map' (mean AP over IoU thresholds)
        if task == 'det':
            det_res = self.det_metric.compute()
            self.results['mAP'] = det_res.get('map', None)

        # Segmentation: IoU returns either single value or per-class, take mean
        elif task == 'seg':
            miou = self.seg_metric.compute()
            if torch.is_tensor(miou):
                miou = miou.item()
                print(miou)
            self.results['mIoU'] = miou

        # Classification accuracy
        elif task == 'cls':
            top1 = self.cls_metric.compute()
            if torch.is_tensor(top1):
                top1 = top1.item()
            self.results['Top1'] = top1
        return self.results

def remap_targets(targets):
    new = targets.clone().long()
    # 1…20 -> 0…19
    mask_fg = (new >= 1) & (new <= 20)
    new[mask_fg] = new[mask_fg] - 1
    new[new == 255] = -100
    return new