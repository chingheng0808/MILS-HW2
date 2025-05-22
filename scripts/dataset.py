import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F
import random
import torch
import cv2
from torch.utils.data import DataLoader

######################################### COCO-Det #########################################
RESIZE_DET = 512
coco_detection_transform = A.Compose([
    A.LongestMaxSize(max_size=RESIZE_DET),
    A.PadIfNeeded(min_height=RESIZE_DET, min_width=RESIZE_DET, border_mode=0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
def collate_fn_coco(batch):
    """
    batch: list of tuples (img_tensor, target_dict)
    回傳:
      imgs: Tensor[B, C, H, W]
      targets: list of length B，裡面每個都是一個 dict
    """
    imgs, targets = zip(*batch)            # unzip
    imgs = torch.stack(imgs, dim=0)        # 堆疊成一個大 tensor
    return imgs, list(targets)
class MiniCOCODataset(Dataset):
    def __init__(self, root, ann_file, train=True):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.train = train
        if self.train:
            self.transforms = coco_detection_transform
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=RESIZE_DET),
                A.PadIfNeeded(min_height=RESIZE_DET, min_width=RESIZE_DET, border_mode=0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)

        # 使用 OpenCV + RGB 轉換，Albumentations 使用 numpy 格式
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for ann in anns:
            if ann.get("iscrowd", 0):  # optional: skip crowd
                continue
            boxes.append(ann["bbox"])  # [x, y, w, h]
            labels.append(self.cat2label[ann["category_id"]])

        # Albumentations transform
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                category_ids=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']

        # Convert to tensors
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        return img, target


######################################### VOC-Seg #########################################
class SegmentationAugment:
    def __init__(self, resize=(512, 512), hflip_prob=0.5, random_crop_size=256, train=True):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.random_crop_size = random_crop_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Resize
        image = F.resize(image, self.resize, interpolation=Image.BILINEAR)
        mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)  # don't interpolate masks (important!! => masks should be long type)
        
        if self.train:
            # Random horizontal flip
            if random.random() < self.hflip_prob:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.random_crop_size, self.random_crop_size))
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        # To tensor
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        mask = F.pil_to_tensor(mask).long().squeeze(0)

        return image, mask

class MiniVOCSegDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = root

        if split == 'train':
            split += '240'
        else:
            split += '60'
        split_file = os.path.join(root, 'ImageSets', 'Segmentation', split + '.txt')
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.images_dir = os.path.join(root, 'JPEGImages')
        self.masks_dir = os.path.join(root, 'SegmentationClass')
        self.train = split == 'train'
        self.transform = SegmentationAugment(train=self.train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_id + '.jpg')
        mask_path = os.path.join(self.masks_dir, img_id + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) 

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
######################################### ImageNet ########################################
LABEL_MAP = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}
class ImageNetDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        self.classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(144),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(160),
                transforms.CenterCrop(144),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
############################ Multi-Task Dataset ########################################

class MultiTaskLoader:
    def __init__(self, det_loader, seg_loader, cls_loader):
        self.loaders = [
            ("cls", cls_loader),
            ("seg", seg_loader),
            ("det", det_loader),
        ]
        # 總 batch 數 = 各 loader 的 batch 數總和
        self._len = sum(len(loader) for _, loader in self.loaders)

    def __len__(self):
        return self._len

    def __iter__(self):
        # 依序跑 cls, seg, det loader
        for task, loader in self.loaders:
            for batch in loader:
                if task == "det":
                    imgs, targets = batch
                    yield imgs.cuda(), "det", {"det": targets}
                elif task == "seg":
                    imgs, masks = batch
                    yield imgs.cuda(), "seg", {"seg": masks.cuda()}
                else:  # "cls"
                    imgs, labels = batch
                    yield imgs.cuda(), "cls", {"cls": labels.cuda()}

def get_multi_task_loader(args):
    det_ds = MiniCOCODataset(args.det_root, args.det_ann, train=True)
    det_loader = DataLoader(
        det_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_coco,
    )

    seg_ds = MiniVOCSegDataset(args.seg_root, split='train')
    seg_loader = DataLoader(
        seg_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cls_ds = ImageNetDataset(args.cls_root, train=True)
    cls_loader = DataLoader(
        cls_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return MultiTaskLoader(cls_loader=cls_loader, det_loader=det_loader, seg_loader=seg_loader)

class MultiTaskBatchLoader:
    """
    每次 __next__ 回傳 det/seg/cls 三個任務 batch 的組合，結構如下：
    {
        "det": (imgs_det, targets_det),
        "seg": (imgs_seg, masks_seg),
        "cls": (imgs_cls, labels_cls)
    }
    假設三個 loader 長度一致，否則取最短的長度。
    """
    def __init__(self, det_loader, seg_loader, cls_loader):
        self.det_loader = det_loader
        self.seg_loader = seg_loader
        self.cls_loader = cls_loader
        self.min_len = min(len(det_loader), len(seg_loader), len(cls_loader))
        self.iter_det = None
        self.iter_seg = None
        self.iter_cls = None

    def __len__(self):
        return self.min_len

    def __iter__(self):
        self.iter_det = iter(self.det_loader)
        self.iter_seg = iter(self.seg_loader)
        self.iter_cls = iter(self.cls_loader)
        for _ in range(self.min_len):
            batch = {}
            batch["det"] = next(self.iter_det)
            batch["seg"] = next(self.iter_seg)
            batch["cls"] = next(self.iter_cls)
            yield batch
def get_multi_task_batch_loader(args):
    det_ds = MiniCOCODataset(args.det_root, args.det_ann, train=True)
    det_loader = DataLoader(
        det_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_coco,
    )

    seg_ds = MiniVOCSegDataset(args.seg_root, split='train')
    seg_loader = DataLoader(
        seg_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cls_ds = ImageNetDataset(args.cls_root, train=True)
    cls_loader = DataLoader(
        cls_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return MultiTaskBatchLoader(det_loader=det_loader, seg_loader=seg_loader, cls_loader=cls_loader)
