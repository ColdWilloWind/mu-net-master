import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.dataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pylab as pl
from model.MUNet import MUNet_512
from losses.loss import Criterion
from stn.GeoTransformation import AffineTransform
import warnings
warnings.filterwarnings("ignore")

def adjust_learning_rate(optimizer):
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = learn_rate * (
                1.0 - float(group['step']) * float(batch_size) / (num_sample * float(train_epoch)))
    return

def show_loss_fig(Epoch, Loss):
    np_Epoch = np.array(Epoch.copy())
    np_Loss = np.array(Loss.copy())
    # np_Valid_Loss = np.array(Valid_Loss.copy())
    # np_Valid2_Loss = np.array(Valid2_Loss.copy())
    fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
    pl.plot(np_Epoch, np_Loss, 'g-')
    p2 = pl.plot(np_Epoch, np_Loss, 'r-', label=u'Train Loss')
    pl.legend()
    # 显示图例
    # p3 = pl.plot(np_Epoch, np_Valid_Loss, 'b-', label=u'OS Valid Loss')
    # pl.legend()
    # p4 = pl.plot(np_Epoch, np_Valid2_Loss, 'g-', label=u'Rocket Valid Loss')
    # pl.legend()
    pl.xlabel(u'epochs')
    pl.ylabel(u'loss')
    plt.title('Compare loss for different models in training')
    plt.savefig(save_loss_folder + 'train_results_loss.png')
    pl.show

def train():
    print('Using device ' + str(device) + ' for training!')
    dataset = MyDataset(train_modal1_folder, train_modal2_folder, range_geo_distortion)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = MUNet_512().to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate,
                           momentum=0.9, dampening=0.9,
                           weight_decay=weight_decay)
    Loss = []
    Epoch = []
    for epoch in range(train_epoch):
        loss_epoch = 0
        for idx, data in enumerate(dataloader):
            tensor1 = data['tensor1']
            # tensor2 = data['tensor2']
            tensor2_warp = data['tensor2_warp']
            gt_matrix = data['trans_matrix']
            # gt_inv_matrix = data['inv_trans_matrix']
            input_tensor = torch.cat([tensor1, tensor2_warp], dim=1)
            output_matrix = model(input_tensor)
            loss = Criterion(output_matrix, gt_matrix, tensor1, tensor2_warp,
                             supervised_criterion, alpha_,
                             unsupervised_criterion, similarity, descriptor, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer)
            loss_epoch = loss_epoch + loss.item()
        loss_epoch = loss_epoch / batch_size
        Epoch.append(epoch)
        Loss.append(loss_epoch)
        print('Epoch [{}/{}], LR [{:.5f}], Loss: {:.5f}'
              .format(epoch, train_epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss_epoch))
        show_loss_fig(Epoch=Epoch, Loss=Loss)
        save_model_path = os.path.join(save_model_folder,
                                         'MUNet' + '_{}_{}.pth').format(epoch, round(loss_epoch, 5))
        torch.save(model.state_dict(), save_model_path)

def input_tensor_preprocess(tensor_, scale):
    scale_factor = int(1/scale)
    Pooling = nn.AvgPool2d(scale_factor,stride=2,padding=0,ceil_mode=False)
    tensor_ = Pooling(tensor_)
    return tensor_

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ PATH Setting """
    train_modal1_folder = 'E:/Tang/DATA/Optical-Optical/time1/'  # Folder path for training image pairs for modal-1 and model-2.
    train_modal2_folder = 'E:/Tang/DATA/Optical-Optical/time2/'
    # Ensure that the paired images have the same name on both folders, and both folders have same number of images.
    save_model_folder = './saved/model/'  # Folder path for saved model for modal-1 and model-2.
    save_loss_folder = './saved/loss_fig/'  # Folder path for saved loss information

    """ Training Setting """
    learn_rate = 0.0001
    weight_decay = 1e-4
    batch_size = 8
    train_epoch = 100
    image_size = 512
    num_sample = 200
    supervised_criterion = True
    unsupervised_criterion = True
    similarity = 'NCC'
    descriptor = 'defult'
    mask = True
    alpha_ = 5  # balance between supervised and unsupervised criterion, which only works on both criterions set True.

    """ DATA Geometric Distortion Setting """
    # Set the parameter range for simulating geometric distortion.
    # [a, b, c]: a refers to min, b refers to max, and c refers to interval.
    range_translation_pixel_x = [-10, 10, 1]
    range_translation_pixel_y = [-10, 10, 1]
    range_scale_x = [0.9, 1.1, 0.05]
    range_scale_y = [0.9, 1.1, 0.05]
    range_rotate_angle = [-10, 10, 2]
    range_shear_angle_x = [0, 0, 0]
    range_shear_angle_y = [0, 0, 0]
    translation_x_equals_y = False
    scale_x_equals_y = True
    shear_x_equals_y = False
    range_geo_distortion = {
        'range_translation_pixel_x': range_translation_pixel_x,
        'range_translation_pixel_y': range_translation_pixel_y,
        'range_scale_x': range_scale_x,
        'range_scale_y': range_scale_y,
        'range_rotate_angle': range_rotate_angle,
        'range_shear_angle_x': range_shear_angle_x,
        'range_shear_angle_y': range_shear_angle_y,
        'translation_x_equals_y': translation_x_equals_y,
        'scale_x_equals_y': scale_x_equals_y,
        'shear_x_equals_y': shear_x_equals_y
    }
    train()
