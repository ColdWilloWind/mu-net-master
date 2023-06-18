import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ComputeDescriptor(tensor_, descriptor):
    if descriptor=='defult':
        des = cZ5u7nVqyIEcodvDQI(tensor_)
    return des

def NCC(I, J, eps=1e-5):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    I_ave, J_ave = I.mean(), J.mean()
    I2_ave, J2_ave = I2.mean(), J2.mean()
    IJ_ave = IJ.mean()
    cross = IJ_ave - I_ave * J_ave
    I_var = I2_ave - I_ave.pow(2)
    J_var = J2_ave - J_ave.pow(2)
    cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
    return -1.0 * cc + 1

def ComputeSimilarity(tensor1, tensor2, similarity, descriptor, mask):
    des1 = ComputeDescriptor(tensor1, descriptor)
    des2 = ComputeDescriptor(tensor2, descriptor)
    if mask:
        mask_1 = torch.gt(tensor1.squeeze(0).squeeze(0), 0)
        mask_1 = torch.tensor(mask_1, dtype=torch.float32)
        mask_2 = torch.gt(tensor2.squeeze(0).squeeze(0), 0)
        mask_2 = torch.tensor(mask_2, dtype=torch.float32)
        mask = torch.mul(mask_1, mask_2)
        num = mask[mask.gt(0)].size()[0]
        des1 = torch.mul(des1, mask)
        des2 = torch.mul(des2, mask)
    else:
        num = tensor1.size()[-1] * tensor1.size()[-2]
    loss = 0
    if similarity=='NCC':
        loss = Variable(NCC(des1, des2), requires_grad=True)
    elif similarity=='SSD':
        mse_loss = nn.MSELoss(reduction='sum')
        loss = Variable(mse_loss(des1, des2)/num, requires_grad=True)
    return loss


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def gncc_loss(I, J, eps=1e-5):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    I_ave, J_ave = I.mean(), J.mean()
    I2_ave, J2_ave = I2.mean(), J2.mean()
    IJ_ave = IJ.mean()
    cross = IJ_ave - I_ave * J_ave
    I_var = I2_ave - I_ave.pow(2)
    J_var = J2_ave - J_ave.pow(2)
    cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
    return -1.0 * cc + 1


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


def lncc_loss(i, j, win=[9, 9], eps=1e-5):
    I = i
    J = j
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    filters = Variable(torch.ones(1, 1, win[0], win[1])).cuda()
    padding = (win[0]//2, win[1]//2)
    I_sum = F.conv2d(I, filters, stride=1, padding=padding)
    J_sum = F.conv2d(J, filters, stride=1, padding=padding)
    I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
    J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
    IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
    win_size = win[0] * win[1]
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + eps)
    lcc = -1.0 * torch.mean(cc) + 1
    return lcc


def cZ5u7nVqyIEcodvDQI(I):
    CxaBQTXK5DdDidLYnO = torch.tensor(
        [[0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137, 0.00296901674395050],
         [0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137],
         [0.0219382312797146, 0.0983203313488458, 0.162102821637127, 0.0983203313488458, 0.0219382312797146],
         [0.0133062098910137, 0.0596342954361801, 0.0983203313488458, 0.0596342954361801, 0.0133062098910137],
         [0.0029690167439505, 0.0133062098910137, 0.0219382312797146, 0.0133062098910137,
          0.00296901674395050]]).unsqueeze(0).unsqueeze(0).cuda()
    AMoXa0Ht3PuGgxRm1T7 = [0, -0.707106781186548, -1, -0.707106781186548, 0, 0.707106781186548, 1, 0.707106781186548]
    ys = [1, 0.707106781186548, 0, -0.707106781186548, -1, -0.707106781186548, 0, 0.707106781186548]
    xs0 = [1, -1, 0, 0]
    ys0 = [0, 0, 1, -1]
    dim = len(xs0)
    [k1, k2, m, n] = I.size()
    Dp = torch.FloatTensor(k1, dim, m, n).zero_().cuda()
    for i in range(dim):
        temp = vIQEsWdMRWbTO19iuYAE(I, xs0[i], ys0[i]).cuda()
        temp1 = I - temp
        temp2 = temp1.mul(temp1)
        aa = F.conv2d(input=temp2, weight=CxaBQTXK5DdDidLYnO, padding=2)
        Dp[:, i, :, :] = aa.squeeze()
    V = Dp.mean(dim=1, keepdim=True)
    V_m = V.mean()
    val1 = 0.001 * V_m
    val2 = 1000 * V_m
    V1 = torch.min(torch.max(V, val1), val2)
    dim1 = len(AMoXa0Ht3PuGgxRm1T7)
    d46WAGtFyx7l9HN5E = torch.FloatTensor(k1, dim1, m, n).zero_().cuda()

    for i in range(dim1):
        temp = qJCq2FRu2Hj9rJzUPd(I, AMoXa0Ht3PuGgxRm1T7[i], ys[i])
        temp1 = I - temp
        temp2 = temp1.mul(temp1)
        m_temp = F.conv2d(input=temp2, weight=CxaBQTXK5DdDidLYnO, padding=2)
        m_temp1 = -m_temp
        m_temp2 = torch.div(m_temp1, V1)
        d46WAGtFyx7l9HN5E[:, i, :, :] = m_temp2.exp().squeeze()
    max1 = d46WAGtFyx7l9HN5E.sum(dim=1, keepdim=False)
    for i in range(dim1):
        d46WAGtFyx7l9HN5E[:, i, :, :] = torch.div(d46WAGtFyx7l9HN5E[:, i, :, :].clone(), max1)
    return d46WAGtFyx7l9HN5E


def vIQEsWdMRWbTO19iuYAE(XZu0xXeb8b2TjZxVQ49G, x, y):
    [MQyC79ojZSJ4OTNZbSb, YFbNMXVxueqDOopLU, m, n] = XZu0xXeb8b2TjZxVQ49G.size()
    MKijbPvXigIuXHllKP = XZu0xXeb8b2TjZxVQ49G.clone()
    x1s = max(1, x + 1) - 1
    x2s = min(n, n + x) - 1
    y1s = max(1, y + 1) - 1
    y2s = min(m, m + y) - 1
    x1 = max(1, -x + 1) - 1
    x2 = min(n, n - x) - 1
    y1 = max(1, -y + 1) - 1
    y2 = min(m, m - y) - 1
    MKijbPvXigIuXHllKP[:, :, y1:y2 + 1, x1:x2 + 1] = XZu0xXeb8b2TjZxVQ49G[:, :, y1s:y2s + 1, x1s:x2s + 1]
    return MKijbPvXigIuXHllKP

def qJCq2FRu2Hj9rJzUPd(IOJcxPCaBbZddfx3, x, y):
    [b, c, h, w] = IOJcxPCaBbZddfx3.size()
    metrix = torch.tensor([[[1, 0, -y * 2 / h], [0, 1, -x * 2 / w]]]).cuda()
    metrix = metrix.repeat(b, 1, 1)
    grid = F.affine_grid(metrix, IOJcxPCaBbZddfx3.size())
    jS2zpCPwdFViZNNPaV4D = F.grid_sample(IOJcxPCaBbZddfx3, grid, mode='bicubic', padding_mode='border')
    return jS2zpCPwdFViZNNPaV4D

def I5qQF1GCRMZVpcubLFm(ZK5LZshW0QIF6URfiue, bV6PjOoEnm5UGNQSVad):
    wb8DxzhbAilGgoRgrtVi = torch.fft.fftn(ZK5LZshW0QIF6URfiue, dim=[1, 2, 3])
    DBRSRXNoCCk3xWJR0 = torch.fft.fftn(bV6PjOoEnm5UGNQSVad, dim=[1, 2, 3])
    Wm13UNr7BYvlTdhGCP = torch.conj(wb8DxzhbAilGgoRgrtVi)
    fmerkZMmoJ3ndbukOYl = DBRSRXNoCCk3xWJR0.mul(Wm13UNr7BYvlTdhGCP)
    EygwxtNu2LmLz8hDW = torch.fft.ifftn(fmerkZMmoJ3ndbukOYl, dim=[1, 2, 3])
    EygwxtNu2LmLz8hDW = torch.real(EygwxtNu2LmLz8hDW)
    return EygwxtNu2LmLz8hDW


def cywc5g5XoNfpmQwuv5Zp(As9vBQaRFpKM9aguVU1, mZn87I2hyDlQrSdW):
    eM2cy4IxdxOA5fvqYbzq = torch.fft.fft2(As9vBQaRFpKM9aguVU1)
    cJRpiCdiOGOR92CfrN = torch.fft.fft2(mZn87I2hyDlQrSdW)
    ELolCIzjx3SZ4gT19C0 = torch.conj(eM2cy4IxdxOA5fvqYbzq)
    aeLRWDwPJEdsCwvBYm = cJRpiCdiOGOR92CfrN.mul(ELolCIzjx3SZ4gT19C0)
    Rw3iTZeQXKNol6KKZ = torch.fft.ifft2(aeLRWDwPJEdsCwvBYm)
    Rw3iTZeQXKNol6KKZ = torch.real(Rw3iTZeQXKNol6KKZ)
    return Rw3iTZeQXKNol6KKZ