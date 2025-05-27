import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import tarfile
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class VocDataSet(Dataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    """
    NCLASS = 21
    def __init__(self, root='data/VOCdevkit', split='train', transform=None, base_size=512, crop_size=320):
        super(VocDataSet).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size
        _voc_root = os.path.join(root, 'VOC2012')
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        assert split in ['train', 'val', 'trainval']
        _split_f = os.path.join(_splits_dir, split + '.txt')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert (len(self.images) == len(self.masks))
        print(f'Found {len(self.images)} images in the folder {_voc_root}')

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
    
    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.split == 'train':
            img, mask = self._sync_transform(img, mask)
        else:
            img, mask = self._val_sync_transform(img, mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)
    
    @property
    def classes(self):
        """Category Names"""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')
    
if __name__ == '__main__':
    # Download VOC2012 from torchvision

    root = 'data'
    voc_root = os.path.join(root, 'VOCdevkit')

    if not os.path.exists(voc_root):
        print(f"Downloading VOC2012 dataset to {voc_root}...")
        # Use a mirror URL if the official download fails
        # try:
            # datasets.VOCSegmentation(root=root, year='2012', image_set='train', download=True)
        # except Exception as e:
        # print(f"Official download failed: {e}")
        print("Trying alternative mirror...")
        
        # Define mirror URL
        mirror_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        
        import urllib.request
        
        # Create directories if they don't exist
        os.makedirs(root, exist_ok=True)
        
        # Download from mirror
        tar_path = os.path.join(root, "VOCtrainval_11-May-2012.tar")
        urllib.request.urlretrieve(mirror_url, tar_path)
        
        # Extract the tar file
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=root)
        
        # Clean up
        os.remove(tar_path)
        print("Download completed.")
    else:
        print(f"VOC2012 dataset already exists at {voc_root}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VocDataSet(split='train', transform=transform)
    image, mask, name = dataset[0]
    print(f'Image shape: {image.shape}, Mask shape: {mask.shape}, Name: {name}')
