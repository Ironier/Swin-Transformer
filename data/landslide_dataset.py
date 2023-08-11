import os,glob
import json
import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor
import torch
import warnings
from torchvision.transforms import functional as F

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import h5py

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LandslideDataset(data.Dataset):
    def __init__(self, root, ann_directory='', transform=None):
        super(LandslideDataset, self).__init__()

        #self.image_path = os.path.join(root,'JPEGImages')
        self.hdf5_path = os.path.join(root,'SegmentationClass')
        self.transform = transform
        #self.norm = F.normalize()
        #self.to_tensor = ToTensor() # !!!!!!!!该操作会将张量维度从(H,W,C)变为(C,H,W)

        self.image_paths = [os.path.join(root,ann_directory,_dir) for _dir in os.listdir(os.path.join(root,ann_directory))]
        self.labels = []


    def _load_hdf5(self, path, ignore_key=())->np.array: # H W C
        '''
            read hdf5 format file.
            path : path to hdf5 file. need '.hdf5' postfix.
            ignore_key : column to be ignored in data.
        '''
        file = h5py.File(path)
        data = None
        #print(file.keys())
        for key in file:
            if(key in ignore_key):
                continue
            val = np.array(file[key]) # H*W*[C]
            if(data is None):
                data = val # 256 256 {3,1,......}
            else:
                data = np.concatenate([data,val],axis=2) #在通道维度上拼接张量
        file.close()
        return data

    def _load_hdf5_in_seq(self, paths:list, ignore_key=())->np.array: # H W C
        '''
            read hdf5 format file.
            path : path to hdf5 file. need '.hdf5' postfix.
            ignore_key : column to be ignored in data.
        '''
        seq_all = None
        #print(file.keys())
        for file_name in paths:
            file = h5py.File(file_name)
            data = None
            #print(file.keys())
            for key in file:
                if(key in ignore_key):
                    continue
                val = np.expand_dims((file[key]),axis=-1)
                if(data is None):
                    data = val
                else:
                    data = np.concatenate([data,val],axis=-2) #拼接通道
            file.close()
            if(seq_all is None):
                seq_all = data
            else:
                seq_all = np.concatenate([seq_all,data],axis=-1) #拼接序列
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        input_list=[]
        target_list=[]
        #print('test',self.image_paths[index])
        with open(self.image_paths[index]) as f:
            flag = 0
            for line in f.readlines():
                line=line.strip('\n').strip('\t')
                if(line=='ORIGINAL:'):
                    flag=1
                    continue
                if(line=='FUTURE:'):
                    flag=2
                    continue
                if (flag==1):
                    input_list.append(os.path.join(self.hdf5_path,line+'.hdf5'))
                elif (flag==2):
                    target_list.append(os.path.join(self.hdf5_path,line+'.hdf5'))
        images = self._load_hdf5_in_seq(input_list,ignore_key=('seq')) #H W C S
        in_seq = self._load_hdf5_in_seq(input_list,ignore_key=('displace'))
        targets = self._load_hdf5_in_seq(target_list,ignore_key=('seq'))
        out_seq = self._load_hdf5_in_seq(target_list,ignore_key=('displace'))

        if self.transform is not None:
            H1,W1,C1,S1 = images.shape
            H2,W2,C2,S2 = targets.shape
            images = images.reshape(H1,W1,C1*S1)
            targets = targets.reshape(H2,W2,C2*S2)
            transformed = self.transform(image=images,mask=targets)
            images = transformed['image']#B S C H W
            targets = transformed['mask']
            images = images.reshape(S1,C1,H1,W1)
            targets = targets.reshape(S2,C2,H2,W2)
            #print(targets.min(),targets.max())
        grid_idx = int(self.image_paths[index].split('grid_')[1].rstrip('.txt'))
        return images, targets, grid_idx, in_seq, out_seq

    def __len__(self):
        return len(self.image_paths)
