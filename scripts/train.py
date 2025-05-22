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
    
class MultiTaskTrainer:
    def __init__(self, args):
        self.args = args

        # student model
        self.model = AllinOneModel(nc_det=10, nc_seg=20, nc_cls=10, n_bbox=2).cuda()
        # self.model = DETModel(nc_det=80, n_bbox=2).cuda()

        # 1) 建 teacher model 並 freeze
        if args.teacher:
            self.teacher = {}
            for task in ('det', 'seg', 'cls'):
                self.teacher[task] = copy.deepcopy(self.model).eval().cuda()
                self.teacher[task].load_state_dict(torch.load(args.save_path + f"/{task}_best.pth"))
                self.teacher[task].eval()
                for p in self.teacher[task].parameters():
                    p.requires_grad = False
        # task-specific loss weights, initialize on GPU
        self.task_weights = torch.nn.Parameter(torch.ones(3, device='cuda'), requires_grad=True)
        
        # optimizer / scheduler
        # include task_weights parameter explicitly in optimizer
        self.optim = optim.AdamW(
            list(self.model.parameters()) + [self.task_weights],
            lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, args.epoch, eta_min=args.lr * 1e-2)

        # task‐specific losses
        self.det_loss_fn = DETLoss(S=32, B=2, C=10)
        self.seg_loss_fn = SEGLoss()
        self.cls_loss_fn = CLSLoss()
        self.distill_loss_fn = torch.nn.MSELoss()
        
        self.best_score = 0.

        # distillation weight
        self.lambda_distill = args.lambda_distill
        
        self.evaluator = Evaluator(num_seg_classes=20)
        self.evaluator.reset()
        
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def multi_task_train(self):
        self.train_loader = get_multi_task_batch_loader(self.args)
        for epoch in range(1, self.args.epoch + 1):
            self.model.train()
            loop = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch}")

            for batch in loop:
                self.optim.zero_grad()
                total_loss = 0.0
                total_main_loss = 0.0
                total_distill_loss = 0.0

                task_weights = torch.sigmoid(self.task_weights)
                task_weights = task_weights / task_weights.sum()
                # Det
                imgs_det, targets_det = batch['det']
                out_det_stu = self.model(imgs_det.cuda())["det"]
                loss_det = 100 * self.det_loss_fn(out_det_stu, encode_yolo_targets(targets_det))
                distill_det = 0.0
                for t in ("seg", "cls"):
                    with torch.no_grad():
                        out_tea = self.teacher[t](imgs_det.cuda())[t]
                    out_stu = self.model(imgs_det.cuda())[t]
                    distill_det += self.distill_loss_fn(out_stu, out_tea)
                distill_det /= 2
                total_loss += task_weights[0] * (loss_det + self.lambda_distill * distill_det)
                total_main_loss += loss_det.item()
                total_distill_loss += distill_det.item()

                # Seg
                imgs_seg, masks_seg = batch['seg']
                out_seg_stu = self.model(imgs_seg.cuda())["seg"]
                seg_out = F.interpolate(out_seg_stu, size=(masks_seg.shape[-1], masks_seg.shape[-2]), mode='bilinear', align_corners=False)
                loss_seg = self.seg_loss_fn(seg_out, masks_seg.cuda())
                distill_seg = 0.0
                for t in ("det", "cls"):
                    with torch.no_grad():
                        out_tea = self.teacher[t](imgs_seg.cuda())[t]
                    out_stu = self.model(imgs_seg.cuda())[t]
                    distill_seg += self.distill_loss_fn(out_stu, out_tea)
                distill_seg /= 2
                total_loss += task_weights[1] * (loss_seg + self.lambda_distill * distill_seg)
                total_main_loss += loss_seg.item()
                total_distill_loss += distill_seg.item()

                # Cls
                imgs_cls, labels_cls = batch['cls']
                out_cls_stu = self.model(imgs_cls.cuda())["cls"]
                loss_cls = self.cls_loss_fn(out_cls_stu, labels_cls.cuda())
                distill_cls = 0.0
                for t in ("det", "seg"):
                    with torch.no_grad():
                        out_tea = self.teacher[t](imgs_cls.cuda())[t]
                    out_stu = self.model(imgs_cls.cuda())[t]
                    distill_cls += self.distill_loss_fn(out_stu, out_tea)
                distill_cls /= 2
                total_loss += task_weights[2] * (loss_cls + self.lambda_distill * distill_cls)
                total_main_loss += loss_cls.item()
                total_distill_loss += distill_cls.item()

                # backward on all accumulated loss
                total_loss.backward()
                self.optim.step()

                loop.set_postfix({
                    'main_loss': total_main_loss / 3,
                    'distill': total_distill_loss / 3,
                    'total': total_loss.item() / 3,
                })

            self.scheduler.step()
            self.multi_task_validation(epoch)
            
    def train(self, train_type):
        if train_type == 'det':
            dataset = MiniCOCODataset(self.args.det_root, self.args.det_ann, train=True)
        elif train_type == 'seg':
            dataset = MiniVOCSegDataset(self.args.seg_root, split='train')
        elif train_type == 'cls':
            dataset = ImageNetDataset(self.args.cls_root, train=True)
        else:
            raise ValueError(f"Unknown train type {train_type}")
        
        if train_type == 'det':
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.args.batchsize,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn_coco
            )
        else:
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.args.batchsize,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
        
        for epoch in range(1, self.args.epoch+1):
            self.model.train()
            loop = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch}")

            for x, labels in loop:
                # print(x.shape)
                x = x.cuda()

                outs = self.model(x)     
                
                if train_type == 'det':
                    loss = self.det_loss_fn(outs['det'], encode_yolo_targets(labels))
                elif train_type == 'seg':
                    seg = F.interpolate(outs['seg'], size=(labels.shape[-1], labels.shape[-2]), mode='bilinear', align_corners=False)
                    loss = self.seg_loss_fn(seg, labels)
                elif train_type == 'cls':
                    loss = self.cls_loss_fn(outs['cls'], labels)

                # backward & step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loop.set_postfix({
                    'task': train_type,
                    'loss': loss.item(),
                })

            self.scheduler.step()
            if epoch >= 10:
                self.validation(train_type, epoch)

    def save(self, save_name, epoch, score=None):
        if epoch % 10 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{save_name}_{epoch}.pth")) 
        if score is not None:
            if score > self.best_score:
                self.best_score = score
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{save_name}_best.pth"))

    def validation(self, train_type, epoch):
        self.evaluator.reset()
        self.model.eval()
        
        if train_type == 'det':
            dataset_val = MiniCOCODataset('/'.join(self.args.det_root.split('/')[:-1])+'/val', '/'.join(self.args.det_ann.split('/')[:-1])+'/instances_val.json', train=False)
        elif train_type == 'seg':
            dataset_val = MiniVOCSegDataset(self.args.seg_root, split='val')
        elif train_type == 'cls':
            dataset_val = ImageNetDataset('/'.join(self.args.cls_root.split('/')[:-1])+'/val', train=False)
        torch.cuda.empty_cache()
        
        if train_type == 'det':
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
                x = x.to("cuda")
                
                outs = self.model(x)
                outs = {k: v.cpu() for k, v in outs.items()}
                if train_type == 'det':
                    out_det = yolo_v1_NMS(outs['det'])
                    label_ = convert_coco_to_xyxy(labels)
                    # print('out_det', out_det)
                    # print('label_', label_)
                    self.evaluator.evaluate(out_det, label_, task='det')
                elif train_type == 'seg':
                    # print(outs['seg'].shape, labels.shape)
                    self.evaluator.evaluate(outs['seg'], labels, task='seg')
                elif train_type == 'cls':
                    self.evaluator.evaluate(outs['cls'], labels, task='cls')
                # self.evaluator.add_batch()
                
                loop.set_description("[Validation]")
            
            result = self.evaluator.compute(train_type)
            print(result)
            
        score = 0
        for k in result:
            print(f"{k}: {result[k]}")
            score += result[k]
        score /= len(result)
        self.save(train_type, epoch, score)

    def multi_task_validation(self, epoch):
        self.model.eval()
        val_loaders = {}

        # det
        dataset_det = MiniCOCODataset(
            '/'.join(self.args.det_root.split('/')[:-1]) + '/val',
            '/'.join(self.args.det_ann.split('/')[:-1]) + '/instances_val.json',
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
        dataset_cls = ImageNetDataset('/'.join(self.args.cls_root.split('/')[:-1]) + '/val', train=False)
        val_loaders['cls'] = DataLoader(
            dataset_cls, batch_size=self.args.batchsize, shuffle=False,
            num_workers=2, pin_memory=True
        )

        with torch.no_grad():
            self.evaluator.reset()
            for task in ('det', 'seg', 'cls'):
                val_loader = val_loaders[task]
                loop = tqdm(val_loader, total=len(val_loader), leave=False, desc=f"[Val-{task}]")
                for x, labels in loop:
                    x = x.cuda()
                    outs = self.model(x)
                    outs = {k: v.cpu() for k, v in outs.items()}
                    if task == 'det':
                        out_det = yolo_v1_NMS(outs['det'])
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

    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--epoch_val", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--model_name", type=str, default="AllinOne")
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--det_root", type=str, default="/ssd4/chingheng/DL2/data/mini-coco-det/images/train")
    parser.add_argument("--det_ann", type=str, default="/ssd4/chingheng/DL2/data/mini-coco-det/annotations/instances_train.json")
    parser.add_argument("--seg_root", default="/ssd4/chingheng/DL2/data/mini-voc-seg")
    parser.add_argument("--cls_root", default="/ssd4/chingheng/DL2/data/imagenette-160/train")
    parser.add_argument("--lambda_distill", type=float, default=10)
    
    parser.add_argument("--train_type", type=str, choices=['det', 'seg', 'cls', "all"])

    parser.add_argument("--resume", type=str)
    parser.add_argument("--teacher", action='store_true', help="use teacher model")

    args = parser.parse_args()

    trainer = MultiTaskTrainer(args)
    
    if args.train_type == 'all':
        trainer.multi_task_train()
    else:
        trainer.train(args.train_type)
    print("Best score:", trainer.best_score)

if __name__ == "__main__":
    start = time.time()
    seed_everything(777)

    main()

    end = time.time()
    print("The total training time is:", end - start)
