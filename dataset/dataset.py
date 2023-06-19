import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
from stn.GeoTransformation import AffineTransformFromRange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def squeezeCondition(tensor_):
    TS = tensor_.size()
    if len(TS)==3 and TS[0]==3:
        tensor_ = tensor_[0]
    elif len(TS)==3 and TS[0]==1:
        tensor_ = tensor_.squeeze(0)
    return tensor_


def show_tensor(t, a, b, c):
    t = squeezeCondition(t)
    t = np.array(t.detach().cpu())
    t = t.astype(np.uint8)
    plt.subplot(a,b,c)
    plt.imshow(t, cmap='gray')


class MyDataset(Dataset):
    def __init__(self, path1, path2, geo_distortion):
        self.path1 = path1
        self.path2 = path2
        self.geo_distortion = geo_distortion
        self.device = device
        self.samples = self._init_data(path1, path2)

    def _init_data(self, path1, path2):
        image_types = ['jpg', 'jpeg', 'bmp', 'png']
        samples = []
        modal1_list = os.listdir(path1)
        modal2_list = os.listdir(path2)
        assert len(modal1_list) == len(modal2_list)
        for it in image_types:
            temp1 = glob.glob(os.path.join(path1, '*.{}'.format(it)))
            temp2 = glob.glob(os.path.join(path2, '*.{}'.format(it)))
            temp = [{'image1': img1, 'image2': img2} for img1, img2 in zip(temp1, temp2)]
            samples += temp
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path = self.samples[idx]
        name1 = os.path.split(data_path['image1'])[1]
        name2 = os.path.split(data_path['image2'])[1]
        assert name1 == name2
        image1 = cv2.imread(data_path['image1'], 0)  # Gray image
        image2 = cv2.imread(data_path['image2'], 0)
        assert image1.shape == image2.shape
        tensor1 = torch.as_tensor(image1, dtype=torch.float, device=self.device).unsqueeze(0)
        tensor2 = torch.as_tensor(image2, dtype=torch.float, device=self.device).unsqueeze(0)
        tensor2_warp, trans_matrix, inv_trans_matrix = AffineTransformFromRange(tensor2, self.geo_distortion)
        tensor2_warp = tensor2_warp.squeeze(0)
        trans_matrix = trans_matrix.squeeze(0)
        inv_trans_matrix = inv_trans_matrix.squeeze(0)

        # show_tensor(tensor1, 1, 3, 1)
        # show_tensor(tensor2, 1, 3, 2)
        # show_tensor(tensor2_warp, 1, 3, 3)
        # plt.show()  # show training samples
        tensor1 = tensor1 / 255.
        tensor2 = tensor2 / 255.
        tensor2_warp = tensor2_warp / 255.
        data = {
            'tensor1': tensor1,
            'tensor2': tensor2,
            'tensor2_warp': tensor2_warp,
            'trans_matrix': trans_matrix,
            'inv_trans_matrix': inv_trans_matrix
        }
        return data
