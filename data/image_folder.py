"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# def make_dataset(dir, max_dataset_size=float("inf")):
#     images = []
#     assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

#     for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
#         for fname in fnames:
#             if is_image_file(fname):
#                 path = os.path.join(root, fname)
#                 images.append(path)
#     return images[:min(max_dataset_size, len(images))]

def make_dataset(dir, max_dataset_size=float("inf")):
    import random
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    n = len(images)
    if n == 0:
        return images

    # max_dataset_size가 무한대이거나 전체보다 크면 전체 반환
    if max_dataset_size == float("inf") or max_dataset_size >= n:
        return images

    k = int(max_dataset_size)

    # 재현성: 환경변수로 시드 설정 (예: DATASET_SAMPLE_SEED=42)
    seed_env = os.environ.get("DATASET_SAMPLE_SEED")
    rng = random.Random(int(seed_env)) if seed_env is not None else random

    # 무작위로 k개 샘플링(중복 없음)
    return rng.sample(images, k)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
