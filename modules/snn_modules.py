import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
import os
from utils import Logger
from modules.optimizations import VariationalHidDropout2d, weight_spectral_norm
sys.path.append('../')


class SNNFuncStatic(nn.Module):
    def __init__(self, network_s_list, network_x_list, vth, fb_num=1):
        super(SNNFuncStatic, self).__init__()
        self.network_s_list = network_s_list
        self.network_x_list = network_x_list
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num
        self.DBReLU = DBReLU(self.vth)
        channels = 256
        self.upsample1_0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample0_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def snn_forward(self, x_list, time_step, output_type='normal', input_type='constant'):
        pass

    def equivalent_func_impl(self, a, x, x_n):
        a[0] = self.DBReLU((self.network_s_list[0][-1](a[0]) + self.network_x_list[0](x)) / self.vth)
        for i in range(len(self.network_s_list[0]) - self.fb_num):
            a[0] = self.DBReLU((self.network_s_list[0][i](a[0])) / self.vth)
        a[1] = self.DBReLU((self.network_s_list[1][-1](a[1]) + self.network_x_list[1](x_n)) / self.vth)
        for i in range(len(self.network_s_list[1]) - self.fb_num):
            a[1] = self.DBReLU((self.network_s_list[1][i](a[1])) / self.vth)

        return a

    def equivalent_func(self, a, x_list):
        x = x_list[0]
        x_n = x_list[1]
        return self.equivalent_func_impl(a, x, x_n)

    def forward(self, x_list, time_step):
        return self.snn_forward(x_list, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].copy(target.network_s_list[i][j])

        self.network_x_list[0].copy(target.network_x_list[0])
        self.network_x_list[1].copy(target.network_x_list[1])

    def set_bn_mode_s(self, mode='train'):
        if mode == 'train':
            for i in range(len(self.network_s_list)):
                for j in range(len(self.network_s_list[i])):
                    if self.network_s_list[i][j].BN:
                        self.network_s_list[i][j].bn.train()
        else:
            for i in range(len(self.network_s_list)):
                for j in range(len(self.network_s_list[i])):
                    if self.network_s_list[i][j].BN:
                        self.network_s_list[i][j].bn.eval()

    def save_bn_statistics(self):
        self.network_x_list[0].save_bn_statistics()
        self.network_x_list[1].save_bn_statistics()
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].save_bn_statistics()

    def restore_bn_statistics(self):
        self.network_x_list[0].restore_bn_statistics()
        self.network_x_list[1].restore_bn_statistics()
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].restore_bn_statistics()

    def save_bn_statistics_x(self):
        self.network_x_list[0].save_bn_statistics()
        self.network_x_list[1].save_bn_statistics()

    def restore_bn_statistics_x(self):
        self.network_x_list[0].restore_bn_statistics()
        self.network_x_list[1].restore_bn_statistics()


class SNNLIFFuncStatic(SNNFuncStatic):

    def __init__(self, network_s_list, network_x_list, vth, leaky, fb_num=1):
        super(SNNLIFFuncStatic, self).__init__(network_s_list, network_x_list, vth, fb_num)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x_list, time_step, output_type='normal', input_type='constant'):
        x1 = self.network_x_list[0](x_list[0])
        x1_n = self.network_x_list[1](x_list[1])
        # first layer fuse
        x1_r = x1 + self.upsample1_0(x1_n)
        x1_n_r = self.downsample0_1(x1) + x1_n
        u_list = []
        u_n_list = []
        s_list = []
        s_n_list = []
        u1 = x1_r
        s1 = (u1 >= self.vth).float()
        u1 = u1 - self.vth * s1
        u1 = u1 * self.leaky
        u1_n = x1_n_r
        s1_n = (u1_n >= self.vth).float()
        u1_n = u1_n - self.vth * s1_n
        u1_n = u1_n * self.leaky
        
        u_list.append(u1)
        s_list.append(s1)
        u_n_list.append(u1_n)
        s_n_list.append(s1_n)

        for i in range(len(self.network_s_list[0]) - 1):
            ui = self.network_s_list[0][i](s_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            ui = ui * self.leaky
            u_list.append(ui)
            s_list.append(si)

        for i in range(len(self.network_s_list[1]) - 1):
            ui = self.network_s_list[1][i](s_n_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            ui = ui * self.leaky
            u_n_list.append(ui)
            s_n_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]
        af_n = s_n_list[0]
        al_n = s_n_list[-self.fb_num]
        for t in range(time_step - 1):
            u_list[0] = u_list[0] + self.network_s_list[0][-1](s_list[-1]) + x1
            u_n_list[0] = u_n_list[0] + self.network_s_list[1][-1](s_n_list[-1]) + x1_n
            u1_r = u_list[0] + self.upsample1_0(u_n_list[0])
            u1_n_r = self.downsample0_1(u_list[0]) + u_n_list[0]

            u_list[0] = u1_r
            u_n_list[0] = u1_n_r

            s_list[0] = (u_list[0] >= self.vth).float()
            u_list[0] = u_list[0] - self.vth * s_list[0]
            u_list[0] = u_list[0] * self.leaky

            s_n_list[0] = (u_n_list[0] >= self.vth).float()
            u_n_list[0] = u_n_list[0] - self.vth * s_n_list[0]
            u_n_list[0] = u_n_list[0] * self.leaky

            for i in range(len(self.network_s_list[0]) - 1):
                u_list[i + 1] = u_list[i + 1] + self.network_s_list[0][i](s_list[i])
                s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]
                u_list[i + 1] = u_list[i + 1] * self.leaky

            for i in range(len(self.network_s_list[1]) - 1):
                u_n_list[i + 1] = u_n_list[i + 1] + self.network_s_list[1][i](s_n_list[i])
                s_n_list[i + 1] = (u_n_list[i + 1] >= self.vth).float()
                u_n_list[i + 1] = u_n_list[i + 1] - self.vth * s_n_list[i + 1]
                u_n_list[i + 1] = u_n_list[i + 1] * self.leaky
        
        weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
        if output_type == 'normal':
            return [af/weighted, af_n/weighted], [al/weighted, al_n/weighted]


class SNNFuncNeuro(nn.Module):
    def __init__(self, network_s_list, network_x_list, vth, fb_num=1):
        super(SNNFuncNeuro, self).__init__()
        self.network_s_list = network_s_list
        self.network_x_list = network_x_list
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num
        self.DBReLU = DBReLU(self.vth)
        channels = 32
        self.upsample1_0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample0_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def snn_forward(self, x_list, time_step, output_type='normal', input_type='constant'):
        pass

    def equivalent_func_impl(self, a, x, x_n):
        a[0] = self.DBReLU((self.network_s_list[0][-1](a[0]) + self.network_x_list[0](x)) / self.vth)
        for i in range(len(self.network_s_list[0]) - self.fb_num):
            a[0] = self.DBReLU((self.network_s_list[0][i](a[0])) / self.vth)
        a[1] = self.DBReLU((self.network_s_list[1][-1](a[1]) + self.network_x_list[1](x_n)) / self.vth)
        for i in range(len(self.network_s_list[1]) - self.fb_num):
            a[1] = self.DBReLU((self.network_s_list[1][i](a[1])) / self.vth)

        return a

    def equivalent_func(self, a, x_list):
        x = x_list[0]
        x_n = x_list[1]
        return self.equivalent_func_impl(a, x, x_n)

    def forward(self, x_list, time_step):
        return self.snn_forward(x_list, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].copy(target.network_s_list[i][j])

        self.network_x_list[0].copy(target.network_x_list[0])
        self.network_x_list[1].copy(target.network_x_list[1])

    def set_bn_mode_s(self, mode='train'):
        if mode == 'train':
            for i in range(len(self.network_s_list)):
                for j in range(len(self.network_s_list[i])):
                    if self.network_s_list[i][j].BN:
                        self.network_s_list[i][j].bn.train()
        else:
            for i in range(len(self.network_s_list)):
                for j in range(len(self.network_s_list[i])):
                    if self.network_s_list[i][j].BN:
                        self.network_s_list[i][j].bn.eval()

    def save_bn_statistics(self):
        self.network_x_list[0].save_bn_statistics()
        self.network_x_list[1].save_bn_statistics()
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].save_bn_statistics()

    def restore_bn_statistics(self):
        self.network_x_list[0].restore_bn_statistics()
        self.network_x_list[1].restore_bn_statistics()
        for i in range(len(self.network_s_list)):
            for j in range(len(self.network_s_list[i])):
                self.network_s_list[i][j].restore_bn_statistics()

    def save_bn_statistics_x(self):
        self.network_x_list[0].save_bn_statistics()
        self.network_x_list[1].save_bn_statistics()

    def restore_bn_statistics_x(self):
        self.network_x_list[0].restore_bn_statistics()
        self.network_x_list[1].restore_bn_statistics()


class SNNLIFFuncNeuro(SNNFuncNeuro):
    def __init__(self, network_s_list, network_x_list, vth, leaky, fb_num=1):
        super(SNNLIFFuncNeuro, self).__init__(network_s_list, network_x_list, vth, fb_num)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x_list, time_step, output_type='normal', input_type='constant'):
        if self.network_x_list[0].BN and self.network_x_list[0].bn.training:
            with torch.no_grad():
                leaky_ = 1.
                x_mean = x_list[0][time_step - 1]
                for t in range(time_step - 1):
                    leaky_ *= self.leaky
                    x_mean += x_list[0][time_step - 2 - t] * leaky_
                x_mean /= (1 - leaky_ * self.leaky) / (1 - self.leaky)
                x_mean = self.network_x_list[0].forward_linear(x_mean)
                if len(x_mean.shape) == 4:
                    mean = torch.mean(x_mean, dim=(0, 2, 3), keepdim=True)
                    var = torch.var(x_mean, dim=(0, 2, 3), keepdim=True)
                else:
                    mean = torch.mean(x_mean, dim=0, keepdim=True)
                    var = torch.var(x_mean, dim=0, keepdim=True)
                var = torch.sqrt(var + 1e-8)
            x1 = self.network_x_list[0](x_list[0][0], BN_mean_var=[mean, var])
        else:
            x1 = self.network_x_list[0](x_list[0][0])
        if self.network_x_list[1].BN and self.network_x_list[1].bn.training:
            with torch.no_grad():
                leaky_ = 1.
                x_mean_n = x_list[1][time_step - 1]
                for t in range(time_step - 1):
                    leaky_ *= self.leaky
                    x_mean_n += x_list[1][time_step - 2 - t] * leaky_
                x_mean_n /= (1 - leaky_ * self.leaky) / (1 - self.leaky)
                x_mean_n = self.network_x_list[1].forward_linear(x_mean_n)
                if len(x_mean_n.shape) == 4:
                    mean_n = torch.mean(x_mean_n, dim=(0, 2, 3), keepdim=True)
                    var_n = torch.var(x_mean_n, dim=(0, 2, 3), keepdim=True)
                else:
                    mean_n = torch.mean(x_mean_n, dim=0, keepdim=True)
                    var_n = torch.var(x_mean_n, dim=0, keepdim=True)
                var_n = torch.sqrt(var_n + 1e-8)
            x1_n = self.network_x_list[1](x_list[1][0], BN_mean_var=[mean_n, var_n])
        else:
            x1_n = self.network_x_list[1](x_list[1][0])
        # first layer fuse
        x1_r = x1 + self.upsample1_0(x1_n)
        x1_n_r = self.downsample0_1(x1) + x1_n
        u_list = []
        u_n_list = []
        s_list = []
        s_n_list = []
        u1 = x1_r
        s1 = (u1 >= self.vth).float()
        u1 = u1 - self.vth * s1
        u1 = u1 * self.leaky
        u1_n = x1_n_r
        s1_n = (u1_n >= self.vth).float()
        u1_n = u1_n - self.vth * s1_n
        u1_n = u1_n * self.leaky

        u_list.append(u1)
        s_list.append(s1)
        u_n_list.append(u1_n)
        s_n_list.append(s1_n)

        for i in range(len(self.network_s_list[0]) - 1):
            ui = self.network_s_list[0][i](s_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            ui = ui * self.leaky
            u_list.append(ui)
            s_list.append(si)

        for i in range(len(self.network_s_list[1]) - 1):
            ui = self.network_s_list[1][i](s_n_list[-1])
            si = (ui >= self.vth).float()
            ui = ui - self.vth * si
            ui = ui * self.leaky
            u_n_list.append(ui)
            s_n_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]
        af_n = s_n_list[0]
        al_n = s_n_list[-self.fb_num]
        for t in range(time_step - 1):
            if self.network_x_list[0].BN and self.network_x_list[0].bn.training:
                u_list[0] = u_list[0] + self.network_s_list[0][-1](s_list[-1]) + self.network_x_list[0](
                    x_list[0][t + 1], BN_mean_var=[mean, var])
            else:
                u_list[0] = u_list[0] + self.network_s_list[0][-1](s_list[-1]) + self.network_x_list[0](
                    x_list[0][t + 1])

            if self.network_x_list[1].BN and self.network_x_list[1].bn.training:
                u_n_list[0] = u_n_list[0] + self.network_s_list[1][-1](s_n_list[-1]) + self.network_x_list[1](
                    x_list[1][t + 1], BN_mean_var=[mean, var])
            else:
                u_n_list[0] = u_n_list[0] + self.network_s_list[1][-1](s_n_list[-1]) + self.network_x_list[1](
                    x_list[1][t + 1])

            u1_r = u_list[0] + self.upsample1_0(u_n_list[0])
            u1_n_r = self.downsample0_1(u_list[0]) + u_n_list[0]

            u_list[0] = u1_r
            u_n_list[0] = u1_n_r

            s_list[0] = (u_list[0] >= self.vth).float()
            u_list[0] = u_list[0] - self.vth * s_list[0]
            u_list[0] = u_list[0] * self.leaky

            s_n_list[0] = (u_n_list[0] >= self.vth).float()
            u_n_list[0] = u_n_list[0] - self.vth * s_n_list[0]
            u_n_list[0] = u_n_list[0] * self.leaky

            for i in range(len(self.network_s_list[0]) - 1):
                u_list[i + 1] = u_list[i + 1] + self.network_s_list[0][i](s_list[i])
                s_list[i + 1] = (u_list[i + 1] >= self.vth).float()
                u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]
                u_list[i + 1] = u_list[i + 1] * self.leaky

            for i in range(len(self.network_s_list[1]) - 1):
                u_n_list[i + 1] = u_n_list[i + 1] + self.network_s_list[1][i](s_n_list[i])
                s_n_list[i + 1] = (u_n_list[i + 1] >= self.vth).float()
                u_n_list[i + 1] = u_n_list[i + 1] - self.vth * s_n_list[i + 1]
                u_n_list[i + 1] = u_n_list[i + 1] * self.leaky

        weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
        if output_type == 'normal':
            return [af / weighted, af_n / weighted], [al / weighted, al_n / weighted]


class SNNConv(nn.Module):

    def __init__(self, d_in, d_out, kernel_size, bias=True, BN=False, stride=1, padding=None, pooling=False, dropout=0.0):
        super(SNNConv, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(d_in, d_out, kernel_size, stride, padding, bias=bias)
        self.BN = BN
        self.pooling = pooling
        if self.BN:
            self.bn = nn.BatchNorm2d(d_out)
        if self.pooling:
            self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.drop = VariationalHidDropout2d(dropout, spatial=False)

        

        self._initialize_weights()

    def forward(self, x, BN_mean_var=None):
        x = self.conv(x)
        if self.BN:
            if BN_mean_var == None:
                x = self.bn(x)
            else:
                x = (x - BN_mean_var[0]) / BN_mean_var[1] * self.bn.weight.reshape(1, -1, 1, 1) + self.bn.bias.reshape(1, -1, 1, 1)
        if self.pooling:
            x = self.pool(x)
        return self.drop(x)

    def forward_linear(self, x):
        return self.conv(x)

    def _wnorm(self, norm_range=1.):
        self.conv, self.conv_fn = weight_spectral_norm(self.conv, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'conv_fn' in self.__dict__:
            self.conv_fn.reset(self.conv)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.conv
        m.weight.data.uniform_(-1, 1)
        for i in range(m.out_channels):
            m.weight.data[i] /= torch.norm(m.weight.data[i])
        if m.bias is not None:
            m.bias.data.zero_()

        if self.BN:
            m = self.bn
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def copy(self, target):
        self.conv.weight.data = target.conv.weight.data.clone()
        if self.conv.bias is not None:
            self.conv.bias.data = target.conv.bias.data.clone()

        if self.BN:
            self.bn.weight.data = target.bn.weight.data.clone()
            self.bn.bias.data = target.bn.bias.data.clone()
            self.bn.running_mean.data = target.bn.running_mean.data.clone()
            self.bn.running_var.data = target.bn.running_var.data.clone()

        self.drop.mask = target.drop.mask.clone()

    def save_bn_statistics(self):
        if self.BN:
            self.bn_running_mean = self.bn.running_mean
            self.bn_running_var = self.bn.running_var

    def restore_bn_statistics(self):
        if self.BN:
            self.bn.running_mean.data = self.bn_running_mean.data.clone()
            self.bn.running_var.data = self.bn_running_var.data.clone()


class SNNConvTranspose(nn.Module):

    def __init__(self, d_in, d_out, kernel_size=3, bias=False, BN=False, stride=2, padding=1, output_padding=1, dropout=0.0):
        super(SNNConvTranspose, self).__init__()
        self.convT = nn.ConvTranspose2d(d_in, d_out, kernel_size, stride, padding, output_padding, bias=bias)
        self.drop = VariationalHidDropout2d(dropout, spatial=False)
        self.BN = BN
        if self.BN:
            self.bn = nn.BatchNorm2d(d_out)

        self._initialize_weights()

    def forward(self, x, BN_mean_var=None):
        x = self.convT(x)
        if self.BN:
            if BN_mean_var == None:
                x = self.bn(x)
            else:
                x = (x - BN_mean_var[0]) / BN_mean_var[1] * self.bn.weight.reshape(1, -1, 1, 1) + self.bn.bias.reshape(1, -1, 1, 1)
        return self.drop(x)

    def _wnorm(self, norm_range=1.):
        self.convT, self.convT_fn = weight_spectral_norm(self.convT, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'convT_fn' in self.__dict__:
            self.convT_fn.reset(self.convT)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.convT
        m.weight.data.uniform_(-1, 1)
        for i in range(m.out_channels):
            m.weight.data[:, i] /= torch.norm(m.weight.data[:, i])
        if m.bias is not None:
            m.bias.data.zero_()

        if self.BN:
            m = self.bn
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def copy(self, target):
        self.convT.weight.data = target.convT.weight.data.clone()
        if self.convT.bias is not None:
            self.convT.bias.data = target.convT.bias.data.clone()

        if self.BN:
            self.bn.weight.data = target.bn.weight.data.clone()
            self.bn.bias.data = target.bn.bias.data.clone()
            self.bn.running_mean.data = target.bn.running_mean.data.clone()
            self.bn.running_var.data = target.bn.running_var.data.clone()

        self.drop.mask = target.drop.mask.clone()

    def save_bn_statistics(self):
        if self.BN:
            self.bn_running_mean = self.bn.running_mean
            self.bn_running_var = self.bn.running_var

    def restore_bn_statistics(self):
        if self.BN:
            self.bn.running_mean.data = self.bn_running_mean.data.clone()
            self.bn.running_var.data = self.bn_running_var.data.clone()


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class DBReLU(nn.Module):
    def __init__(self, v_th):
        super(DBReLU, self).__init__()
        self.v_th = v_th

    def forward(self, inputs):
        out = torch.clamp(inputs / self.v_th, 0, 1)
        return out