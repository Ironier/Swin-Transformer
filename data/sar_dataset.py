import os,glob
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import warnings
from torchvision.transforms import functional as F

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import h5py

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SARDataset(data.Dataset):
    def __init__(self, root, ann_directory='', transform=None):
        super(SARDataset, self).__init__()

        #self.image_path = os.path.join(root,'JPEGImages')
        self.transform = transform
        self.ann_directory = ann_directory

        self.dir_path = os.path.join(root,ann_directory)
        self.sar_path = os.path.join(self.dir_path,'SAR')
        self.data_keys = [dir.split('.')[0][3:] for dir in os.listdir(os.path.join(self.dir_path,'opt_cloudy'))] #xx_xx

    def _load_image(self, path):
        im = np.array(Image.open(path))
        if(len(im.shape)==2):
            im = np.expand_dims(im,axis=-1)
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sar_images = None
        dirs = ['VH','VV']
        for path in [os.path.join(self.sar_path,dir) for dir in dirs]:
            im_path = os.path.join(path,'S1_'+self.data_keys[index]+'.png')
            if sar_images is None:
                sar_images = self._load_image(im_path)
            else:
                sar_images = np.concatenate((sar_images,self._load_image(im_path)),axis=-1)
        im_path = os.path.join(self.dir_path,'opt_cloudy','S2_'+self.data_keys[index]+'.png')
        images = self._load_image(im_path)
        if(self.ann_directory == 'test'):
            targets = np.zeros_like(images,dtype=np.float64)
        else:
            im_path = os.path.join(self.dir_path,'opt_clear','S2_'+self.data_keys[index]+'.png')
            targets = self._load_image(im_path)

        if self.transform is not None:
            transformed = self.transform(image=images,sar_images=sar_images,mask=targets)
            images = transformed['image']#B S C H W
            sar_images = transformed['sar_images']
            targets = transformed['mask']

        return images, sar_images, targets, index

    def __len__(self):
        return len(self.data_keys)
