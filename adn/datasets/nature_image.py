import os
import os.path as path
import json
import torch
import numpy as np
# import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from random import choice
import SimpleITK as sitk
from torch.utils.data import Dataset
from ..utils import read_dir
import monai.transforms as transforms


class NatureImage(torch.utils.data.Dataset):
    def __init__(self, a_dir="data/train/nature_image/artifact", b_dir="data/train/nature_image/no_artifact",
        preprocess='resize_and_crop', random_flip=True, load_size=384, crop_size=256, crop_type="random"):
        super(NatureImage, self).__init__()

        self.a_files = sorted(read_dir(a_dir, predicate="file", recursive=True))
        self.b_files = sorted(read_dir(b_dir, predicate="file", recursive=True))
        self.transform = self.get_transform(preprocess)

    def __len__(self):
        return len(self.a_files)

    def normalize(self, data):
        data = data / 255.0
        data = data * 2.0 - 1.0
        return data

    def get_transform(self, preprocess, random_flip=True, load_size=[256, 256], crop_size=256, crop_type="random", angle=10, convert=True):
        """
        conduct data augmentation
        :param preprocess:
        :param params:
        :param convert:
        :return:
        transforms.Lambda(lambda img: __crop(img, pos, size)) reference：https://www.cnblogs.com/wanghui-garcia/p/11248416.html
        """
        transform_list = []
        if 'resize' in preprocess:
            transform_list.append(transforms.Resize(load_size))
        if 'crop' in preprocess:
            transform_list.append(transforms.RandSpatialCrop(crop_size, random_size=False))
        if 'rotate' in preprocess:
            transform_list.append(transforms.RandRotate(range_x=angle, keep_size=True))
        if 'resample' in preprocess:
            transform_list.append(transforms.Resample())
        if 'flip' in preprocess:
            transform_list.append(transforms.RandFlip(prob=0.5))
        if convert:
            transform_list.append((transforms.ToTensor()))
        return transforms.Compose(transform_list)

    def to_tensor(self, data):
        if data.ndim == 2: data = data[np.newaxis, ...]
        elif data.ndim == 3: data = data.transpose(2, 0, 1)

        data = self.normalize(data)
        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        data = self.denormalize(data)
        return data

    def denormalize(self, data):
        data = data * 0.5 + 0.5
        data = data * 255.0
        return data

    def get(self, a_file):
        data_name = path.basename(a_file)
        a = self.transform(sitk.GetArrayFromImage(sitk.ReadImage(a_file))[np.newaxis, ...].astype(np.float32))
        b = self.transform(sitk.GetArrayFromImage(sitk.ReadImage(choice(self.b_files)))[np.newaxis, ...].astype(np.float32))

        # a = np.array(a).astype(np.float32)
        # b = np.array(b).astype(np.float32)
        #
        # a = self.to_tensor(a)
        # b = self.to_tensor(b)

        return {"data_name": data_name, "artifact": a, "no_artifact": b}

    def __getitem__(self, index):
        a_file = self.a_files[index]
        return self.get(a_file)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from adn.datasets import NatureImage
    dataset = NatureImage()
    data = dataset[100]

    a = dataset.to_numpy(data["artifact"]).astype(np.uint8)
    b = dataset.to_numpy(data["no_artifact"]).astype(np.uint8)

    plt.ion()
    plt.figure(); plt.imshow(a)
    plt.figure(); plt.imshow(b)