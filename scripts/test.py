import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import AllinOneModel
import numpy as np
import argparse
from dataset import get_multi_task_loader, get_multi_task_batch_loader, MiniCOCODataset, MiniVOCSegDataset, ImageNetDataset, collate_fn_coco
from utils import DETLoss, SEGLoss, CLSLoss, yolo_v1_NMS, Evaluator, encode_yolo_targets, convert_coco_to_xyxy
import time 
import os
import random
from torch.utils.data import DataLoader

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
class Tester():
    def __init__(self, args):
        self.args = args
        if args.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = AllinOneModel(nc_det=10, nc_seg=20, nc_cls=10, n_bbox=2).to(self.device)

        self.model.load_state_dict(torch.load(self.args.ckpt, map_location=self.device))
        self.model.eval()
        
        self.evaluator = Evaluator(num_seg_classes=20)
        self.evaluator.reset()

    def test(self, task):
        self.evaluator.reset()
        self.model.eval()
        
        if task == 'det':
            dataset_val = MiniCOCODataset('/'.join(self.args.det_root.split('/')[:-1])+'/val', '/'.join(self.args.det_ann.split('/')[:-1])+'/instances_val.json', train=False)
        elif task == 'seg':
            dataset_val = MiniVOCSegDataset(self.args.seg_root, split='val')
        elif task == 'cls':
            dataset_val = ImageNetDataset('/'.join(self.args.cls_root.split('/')[:-1])+'/val', train=False)
        torch.cuda.empty_cache()
        
        if task == 'det':
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=self.args.batchsize,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn_coco
            )
        else:
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=self.args.batchsize,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        with torch.no_grad():
            loop = tqdm(
                self.dataloader_val, total=len(self.dataloader_val), leave=False
            )
            for x, labels in loop:
                x = x.to(self.device)
                
                outs = self.model(x)
                outs = {k: v.cpu() for k, v in outs.items()}
                if task == 'det':
                    out_det = yolo_v1_NMS(outs['det'], conf_thres=0.0, iou_thres=0.5)
                    label_ = convert_coco_to_xyxy(labels)
                    # print('out_det', out_det)
                    # print('label_', label_)
                    self.evaluator.evaluate(out_det, label_, task='det')
                elif task == 'seg':
                    # print(outs['seg'].shape, labels.shape)
                    self.evaluator.evaluate(outs['seg'], labels, task='seg')
                elif task == 'cls':
                    self.evaluator.evaluate(outs['cls'], labels, task='cls')
                # self.evaluator.add_batch()
                
                loop.set_description("[Validation]")
            
            result = self.evaluator.compute(task)
            print(result)
            
        score = 0
        for k in result:
            print(f"{k}: {result[k]}")
            score += result[k]
        score /= len(result)

    def multi_task_validation(self, epoch):
        self.model.eval()
        val_loaders = {}

        # det
        dataset_det = MiniCOCODataset(self.args.det_root,
            self.args.det_ann,
            train=False
        )
        val_loaders['det'] = DataLoader(
            dataset_det, batch_size=self.args.batchsize, shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=collate_fn_coco
        )

        # seg
        dataset_seg = MiniVOCSegDataset(self.args.seg_root, split='val')
        val_loaders['seg'] = DataLoader(
            dataset_seg, batch_size=self.args.batchsize, shuffle=False,
            num_workers=2, pin_memory=True
        )

        # cls
        dataset_cls = ImageNetDataset(self.args.cls_root, train=False)
        val_loaders['cls'] = DataLoader(
            dataset_cls, batch_size=self.args.batchsize, shuffle=False,
            num_workers=2, pin_memory=True
        )

        with torch.no_grad():
            self.evaluator.reset()
            for task in ('det', 'seg', 'cls'):
                val_loader = val_loaders[task]
                loop = tqdm(val_loader, total=len(val_loader), leave=False, desc=f"[TEST-{task}]")
                for x, labels in loop:
                    x = x.cuda()
                    outs = self.model(x)
                    outs = {k: v.cpu() for k, v in outs.items()}
                    if task == 'det':
                        out_det = yolo_v1_NMS(outs['det'], conf_thres=0.3, iou_thres=0.5)
                        label_ = convert_coco_to_xyxy(labels)
                        self.evaluator.evaluate(out_det, label_, task='det')
                    elif task == 'seg':
                        self.evaluator.evaluate(outs['seg'], labels, task='seg')
                    elif task == 'cls':
                        self.evaluator.evaluate(outs['cls'], labels, task='cls')
                result = self.evaluator.compute(task)
            print("[Validation]")
            score = 0
            for k in result:
                print(f"{k}: {result[k]}")
                if k == 'det':
                    score += 10*result[k]
                else:
                    score += result[k]
            score /= len(result)

            self.save('all', epoch, score)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", type=int, default=8)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--model_name", type=str, default="AllinOne")
    parser.add_argument("--det_root", type=str, default="/ssd4/chingheng/DL2/data/mini-coco-det/images/test")
    parser.add_argument("--det_ann", type=str, default="/ssd4/chingheng/DL2/data/mini-coco-det/annotations/instances_test.json")
    parser.add_argument("--seg_root", default="/ssd4/chingheng/DL2/data/mini-voc-seg")
    parser.add_argument("--cls_root", default="/ssd4/chingheng/DL2/data/imagenette-160/test")
    parser.add_argument("--lambda_distill", type=float, default=100)
    
    parser.add_argument("--task", type=str, choices=['det', 'seg', 'cls', "all"])
    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--resume", type=str)
    parser.add_argument("--teacher", action='store_true', help="use teacher model")
    parser.add_argument("--cuda", action='store_true')

    args = parser.parse_args()

    tester = Tester(args)
    
    tester.test(args.task)

if __name__ == "__main__":
    start = time.time()
    seed_everything(777)

    main()

    end = time.time()
    print("The total training time is:", end - start)
