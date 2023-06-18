from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import numpy as np
from PIL import Image
from numpy import sin, cos, tan
import os
import random
from torch import linalg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inv_affine_matrix(A):
    # Calculate the inversed affine transformation matrix with 6 parameters.
    TA = A.size()
    B = torch.Tensor([[[0, 0, 1]]]).to(device)
    B = B.repeat(TA[0], 1, 1)
    A_ = torch.cat([A, B], dim=1)
    Inv = linalg.inv(A_)
    Inv = Inv[:, 0:2, :]
    return Inv

def get_affine_matrix(img_size, translation_pixel_x, translation_pixel_y,
                  scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y):
    # Calculate the affine transformation matrix and its inverse matrix
    # from the given parameters of translation, scaling, rotation and shearing.
    dx = translation_pixel_x * 2 / img_size
    dy = translation_pixel_y * 2 / img_size
    sx = scale_x
    sy = scale_y
    theta = rotate_angle * np.pi / 180
    faix = shear_angle_x * np.pi / 180
    faiy = shear_angle_y * np.pi / 180
    A = torch.tensor(np.float32(np.array([[
        (sx * (cos(theta) - sin(theta) * tan(faiy)), sx * (cos(theta) * tan(faix) - sin(theta)), dx),
        (sy * (sin(theta) + cos(theta) * tan(faiy)), sy * (sin(theta) * tan(faix) + cos(theta)), dy)]]))).to(device)
    A_inv = inv_affine_matrix(A)
    return A, A_inv

def get_random_number_from_range(range_):
    if range_[2]:
        range_ = np.array(range_)
        t = 0
        while range_[0]%1 or range_[1]%1 or range_[2]%1:
            range_ = range_ * 10
            t = t + 1
        range_ = np.array(range_, dtype=int)
        x_ = random.randrange(range_[0], range_[1], range_[2])
        if t:
            x_ = x_ * np.power(10., -t)
        return x_
    else:
        return range_[0]

def get_all_random_parameters_from_range(range_):
    range_translation_pixel_x = range_['range_translation_pixel_x']
    range_translation_pixel_y = range_['range_translation_pixel_y']
    range_scale_x = range_['range_scale_x']
    range_scale_y = range_['range_scale_y']
    range_rotate_angle = range_['range_rotate_angle']
    range_shear_angle_x = range_['range_shear_angle_x']
    range_shear_angle_y = range_['range_shear_angle_y']
    translation_x_equals_y = range_['translation_x_equals_y']
    scale_x_equals_y = range_['scale_x_equals_y']
    shear_x_equals_y = range_['shear_x_equals_y']
    translation_pixel_x = get_random_number_from_range(range_translation_pixel_x)
    translation_pixel_y = translation_pixel_x if translation_x_equals_y \
        else get_random_number_from_range(range_translation_pixel_y)
    scale_x = get_random_number_from_range(range_scale_x)
    scale_y = scale_x if scale_x_equals_y \
        else get_random_number_from_range(range_scale_y)
    rotate_angle = get_random_number_from_range(range_rotate_angle)
    shear_angle_x = get_random_number_from_range(range_shear_angle_x)
    shear_angle_y = shear_angle_x if shear_x_equals_y \
        else get_random_number_from_range(range_shear_angle_y)
    return translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y

def AffineTransform(tensor_, affine_matrix):
    TS = tensor_.size()
    TA = affine_matrix.size()
    if len(TS)==4:
        b = tensor_.size()[0]
    elif len(TS)==3 and TS[0]==1:
        b = 1
        tensor_ = tensor_.unsqueeze(0)
    elif len(TS)==2:
        b = 1
        tensor_ = tensor_.unsqueeze(0).unsqueeze(0)
    if len(TA)==2:
        affine_matrix = affine_matrix.repeat(b, 1, 1)
    grid = F.affine_grid(affine_matrix, tensor_.size())
    tensor_warp = F.grid_sample(tensor_, grid)
    return tensor_warp

def AffineTransformFromRange(tensor_, range_):
    translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y = \
        get_all_random_parameters_from_range(range_)
    assert tensor_.size()[-1] == tensor_.size()[-2]
    img_size = tensor_.size()[-1]
    A, A_inv = get_affine_matrix(img_size, translation_pixel_x, translation_pixel_y,
                  scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y)
    tensor_warp = AffineTransform(tensor_, A)
    return tensor_warp, A, A_inv

def affine(img, translation_pixel_x, translation_pixel_y, scale_x, scale_y, rotate_angle, shear_angle_x, shear_angle_y):
    img = img.unsqueeze(0)
    [b, c, w, h] = img.shape
    sx = 1/scale_x
    sy = 1/scale_y
    dx = move_x
    dy = move_y
    theta = rotate_angle*np.pi/180
    faix = shearx_angle*np.pi/180
    faiy = sheary_angle*np.pi/180
    A = torch.tensor(np.float32(np.array([[
        (sx * (cos(theta) - sin(theta) * tan(faiy)), sx * (cos(theta) * tan(faix) - sin(theta)), dx),
        (sy * (sin(theta) + cos(theta) * tan(faiy)), sy * (sin(theta) * tan(faix) + cos(theta)), dy)]]))).to(device)
    B = torch.cat([A, torch.Tensor([[[0, 0, 1]]]).to(device)], dim=1)
    Inv = linalg.inv(B)
    Inv = Inv[:, 0:2, :]
    # A = torch.tensor(np.float32(np.array([[(1, 1, 0),
    #                                        (0, 1.5, 0)]])))
    grid = F.affine_grid(A, img.size())
    img_flow = F.grid_sample(img, grid)
    matrix = A.reshape(6)
    matrix_inv = Inv.reshape(6)
    return img_flow, matrix, matrix_inv