import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from stn.GeoTransformation import AffineTransform, inv_affine_matrix
from losses.similarity import ComputeSimilarity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Criterion(predict, GT, tensor1, tensor2_warp, supervised_criterion,
              alpha_, unsupervised_criterion, similarity, descriptor, mask):
    loss = 0
    if supervised_criterion:
        loss = loss + alpha_*CriterionSupervised(predict, GT)
    if unsupervised_criterion:
        loss = loss + CriterionUnsupervised(tensor1, tensor2_warp, predict, similarity, descriptor, mask)
    return loss


def CriterionSupervised(predict, GT):
    mse_loss = nn.MSELoss(reduction='mean')
    loss = mse_loss(predict, GT)
    return loss

def CriterionUnsupervised(tensor1, tensor2_warp, predict, similarity='NCC', descriptor='defult', mask=True):
    tensor1_warp_predict = AffineTransform(tensor1, predict)
    predict_inv = inv_affine_matrix(predict)
    tensor2_predict = AffineTransform(tensor2_warp, predict_inv)
    similarity = ComputeSimilarity(tensor1, tensor2_predict, similarity, descriptor, mask)
    similarity_inv = ComputeSimilarity(tensor1_warp_predict, tensor2_warp, similarity, descriptor, mask)
    loss = (similarity + similarity_inv)*0.5
    return loss