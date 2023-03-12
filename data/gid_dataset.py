import os,glob
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GIDDATASET(data.Dataset):
    def __init__(self, root, ann_file='', transform=None, target_transform=None):
        super(GIDDATASET, self).__init__()

        self.data_path = os.path.join(root,'data')
        self.target_path = os.path.join(root,'labels')
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths = []
        self.image_paths += glob.glob(os.path.join(self.data_path, '*.tif'))

        self.labels = []
        self.labels += glob.glob(os.path.join(self.target_path, '*.tif'))

        # 保证图片路径的个数与标签个数相等
        assert len(self.image_paths) == len(self.labels)


    def _load_image(self, path)->Image:
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(256, 256, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def _load_target(self, path)->Image:
        try:
            im = Image.open(path)
        except:
            print("ERROR TARGET LOADED: ", path)
            random_img = np.random.rand(256, 256) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # images
        images = ToTensor().__call__(np.array(self._load_image(self.image_paths[index]).convert('RGB')))
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = ToTensor().__call__(np.array(self._load_target(self.labels[index]).convert('P')))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images.to(device=device), target.to(device=device)

    def __len__(self):
        return len(self.image_paths)
