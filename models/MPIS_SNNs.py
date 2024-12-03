import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy

sys.path.append("../")
from modules.MPIS_module import StaticModule, NeuroModule
from modules.snn_modules import SNNLIFFuncStatic, SNNLIFFuncNeuro, SNNConv, SNNConvTranspose, DBReLU

logger = logging.getLogger(__name__)

class MPIS_static(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(MPIS_static, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_x, bias=True, BN=True, stride=1,
                                 padding=self.padding_x, dropout=self.dropout, pooling=self.pooling_x)
        self.network_x_n = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_x, bias=True, BN=True, stride=1,
                                   padding=self.padding_x, dropout=self.dropout, pooling=self.pooling_x)

        self.network_s_1 = SNNConv(self.c_s1, self.c_s2, self.kernel_size_s, bias=True, BN=True, stride=1,
                                   padding=self.padding_x, pooling=False, dropout=self.dropout)
        self.network_s_2 = SNNConv(self.c_s2, self.c_s3, self.kernel_size_s, bias=True, BN=True, stride=2,
                                   pooling=False, dropout=self.dropout)

        self.network_s_n_1 = SNNConv(self.c_s1, self.c_s2, self.kernel_size_s, bias=True, BN=True, stride=1,
                                     padding=self.padding_x, pooling=False, dropout=self.dropout)
        self.network_s_n_2 = SNNConv(self.c_s2, self.c_s3, self.kernel_size_s, bias=True, BN=True, stride=2,
                                     pooling=False, dropout=self.dropout)

        self.Trans = SNNConvTranspose(self.c_s3, self.c_s1, bias=False, dropout=self.dropout, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.Trans_n = SNNConvTranspose(self.c_s3, self.c_s1, bias=False, dropout=self.dropout, kernel_size=3, stride=2,
                                        padding=1, output_padding=1)

        self.s_list = nn.ModuleList([self.network_s_1, self.network_s_2, self.Trans])
        self.s_list_n = nn.ModuleList([self.network_s_n_1, self.network_s_n_2, self.Trans_n])

        self.snn_func = SNNLIFFuncStatic(
            nn.ModuleList([self.s_list, self.s_list_n]),
            nn.ModuleList([self.network_x, self.network_x_n]), vth=self.vth, leaky=self.leaky)

        self.snn_func_copy = copy.deepcopy(self.snn_func)

        for param in self.snn_func_copy.parameters():
            param.requires_grad_(False)

        self.snn_ide_conv = StaticModule(self.snn_func, self.snn_func_copy)

        self.downsamp = nn.Sequential(nn.Conv2d(self.c_s3, self.c_s3, 3, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(self.c_s3, momentum=0.1),
                                      DBReLU(self.vth))
        self.incre_modules = nn.Sequential(nn.Conv2d(self.c_s3, self.c_s3, 1, 1, bias=False),
                                           nn.BatchNorm2d(self.c_s3, eps=1e-05, momentum=0.1),
                                           DBReLU(self.vth)
                                           )
        self.downsample_init = nn.Sequential(
            nn.Conv2d(self.c_in, self.c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.c_hidden, momentum=0.1),
            DBReLU(self.vth),
            nn.Conv2d(self.c_hidden, self.c_hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.c_hidden, momentum=0.1),
            DBReLU(self.vth)
        )

        self.classifier = nn.Linear(self.c_s3 * self.h_hidden * self.w_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.c_s1 = cfg['MODEL']['c_s1']
        self.c_s2 = cfg['MODEL']['c_s2']
        self.c_s3 = cfg['MODEL']['c_s3']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
        self.pooling_x = cfg['MODEL']['pooling_x'] if 'pooling_x' in cfg['MODEL'].keys() else False
        self.kernel_size_s = cfg['MODEL']['kernel_size_s']
        self.threshold = cfg['MODEL']['threshold']
        self.time_step = cfg['MODEL']['time_step']
        self.vth = cfg['MODEL']['vth']
        self.dropout = cfg['MODEL']['dropout'] if 'dropout' in cfg['MODEL'].keys() else 0.0
        self.leaky = cfg['MODEL']['leaky'] if 'leaky' in cfg['MODEL'].keys() else None
        if 'OPTIM' in cfg.keys() and 'solver' in cfg['OPTIM'].keys():
            self.solver = cfg['OPTIM']['solver']
        else:
            self.solver = 'broy'

    def _forward(self, x, **kwargs):
        threshold = kwargs.get('threshold', self.threshold)
        time_step = kwargs.get('time_step', self.time_step)
        input_type = kwargs.get('input_type', 'constant')

        leaky = kwargs.get('leaky', self.leaky)
        dev = x.device

        if input_type == 'constant':
            B, C, H, W = x.size()
        else:
            B, C, H, W, _ = x.size()
        H *= 2
        W *= 2
        x1 = torch.zeros([B, self.c_s1, H // self.stride_x, W // self.stride_x]).to(x.device)
        x1_n = torch.zeros([B, self.c_s1, H // self.stride_x // self.stride_x, W // self.stride_x // self.stride_x]).to(
            x.device)
        self.snn_func.network_x_list[0]._reset(x1)
        self.Trans._reset(x1)
        self.snn_func.network_x_list[1]._reset(x1_n)
        self.Trans_n._reset(x1_n)
        x1 = torch.zeros([B, self.c_s2, H // self.stride_x, W // self.stride_x]).to(x.device)
        self.network_s_1._reset(x1)
        x1 = torch.zeros([B, self.c_s3, H // (self.stride_x * 2), W // (self.stride_x * 2)]).to(x.device)
        self.network_s_2._reset(x1)
        x1 = torch.zeros([B, self.c_s2, H // (self.stride_x * 2), W // (self.stride_x * 2)]).to(x.device)
        self.network_s_n_1._reset(x1)
        x1 = torch.zeros([B, self.c_s3, H // (self.stride_x * 2 * 2), W // (self.stride_x * 2 * 2)]).to(x.device)
        self.network_s_n_2._reset(x1)
        z = self.snn_ide_conv(x, time_step=time_step, threshold=threshold, input_type=input_type,
                              solver_type=self.solver, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        x = self.downsample_init(x)
        z = self._forward(x, **kwargs)
        z = self.downsamp(z[0]) + z[1]
        z = self.incre_modules(z)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y

class MPIS_neuro(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(MPIS_neuro, self).__init__()
        self.parse_cfg(cfg)

        self.network_x = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, BN=True, stride=1, padding=self.padding_x, dropout=self.dropout, pooling=self.pooling_x)
        self.network_x_n = SNNConv(self.c_in, self.c_hidden, self.kernel_size_x, bias=True, BN=True, stride=1, padding=self.padding_x, dropout=self.dropout, pooling=self.pooling_x)

        self.network_s_1 = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_s, bias=True, BN=True, stride=2, padding=self.padding_x, pooling=False, dropout=self.dropout)
        self.network_s_2 = SNNConv(self.c_s1, self.c_s2, self.kernel_size_s, bias=True, BN=True, stride=1, pooling=False, dropout=self.dropout)

        self.network_s_n_1 = SNNConv(self.c_hidden, self.c_s1, self.kernel_size_s, bias=True, BN=True, stride=2, padding=self.padding_x, pooling=False, dropout=self.dropout)
        self.network_s_n_2 = SNNConv(self.c_s1, self.c_s2, self.kernel_size_s, bias=True, BN=True, stride=1, pooling=False, dropout=self.dropout)

        self.Trans = SNNConvTranspose(self.c_s2, self.c_hidden, bias=False, dropout=self.dropout, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Trans_n = SNNConvTranspose(self.c_s2, self.c_hidden, bias=False, dropout=self.dropout, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.s_list = nn.ModuleList([self.network_s_1, self.network_s_2, self.Trans])
        self.s_list_n = nn.ModuleList([self.network_s_n_1, self.network_s_n_2, self.Trans_n])

        self.snn_func = SNNLIFFuncNeuro(
                nn.ModuleList([self.s_list, self.s_list_n]),
                nn.ModuleList([self.network_x, self.network_x_n]), vth=self.vth, leaky=self.leaky)

        self.snn_func_copy = copy.deepcopy(self.snn_func)

        for param in self.snn_func_copy.parameters():
            param.requires_grad_(False)

        self.snn_ide_conv = NeuroModule(self.snn_func, self.snn_func_copy)

        self.downsamp = nn.Sequential(nn.Conv2d(self.c_s2, self.c_s2, 3, stride=2, padding=1, bias=True),
                                      nn.BatchNorm2d(self.c_s2, momentum=0.1),
                                      DBReLU(self.vth))
        self.incre_modules = nn.Sequential(nn.Conv2d(self.c_s2, self.c_s2,1,1,bias=False),
                                            nn.BatchNorm2d(self.c_s2, eps=1e-05, momentum=0.1),
                                            DBReLU(self.vth)
                                            )
        self.classifier = nn.Linear(self.c_s2 * self.h_hidden * self.w_hidden, self.num_classes, bias=True)

    def parse_cfg(self, cfg):
        self.c_in = cfg['MODEL']['c_in']
        self.c_hidden = cfg['MODEL']['c_hidden']
        self.c_s1 = cfg['MODEL']['c_s1']
        self.c_s2 = cfg['MODEL']['c_s2']
        self.h_hidden = cfg['MODEL']['h_hidden']
        self.w_hidden = cfg['MODEL']['w_hidden']
        self.num_classes = cfg['MODEL']['num_classes']
        self.kernel_size_x = cfg['MODEL']['kernel_size_x']
        self.stride_x = cfg['MODEL']['stride_x']
        self.padding_x = cfg['MODEL']['padding_x']
        self.pooling_x = cfg['MODEL']['pooling_x'] if 'pooling_x' in cfg['MODEL'].keys() else False
        self.kernel_size_s = cfg['MODEL']['kernel_size_s']
        self.threshold = cfg['MODEL']['threshold']
        self.time_step = cfg['MODEL']['time_step']
        self.vth = cfg['MODEL']['vth']
        self.dropout = cfg['MODEL']['dropout'] if 'dropout' in cfg['MODEL'].keys() else 0.0
        self.leaky = cfg['MODEL']['leaky'] if 'leaky' in cfg['MODEL'].keys() else None
        if 'OPTIM' in cfg.keys() and 'solver' in cfg['OPTIM'].keys():
            self.solver = cfg['OPTIM']['solver']
        else:
            self.solver = 'broy'

    def _forward(self, x, **kwargs):
        threshold = kwargs.get('threshold', self.threshold)
        time_step = kwargs.get('time_step', self.time_step)
        input_type = kwargs.get('input_type', 'spike')
        leaky = kwargs.get('leaky', self.leaky)
        dev = x.device

        if input_type == 'constant':
            B, C, H, W = x.size()
        else:
            B, C, H, W, _ = x.size()
        x1 = torch.zeros([B, self.c_hidden, H, W]).to(x.device)
        x1_n = torch.zeros([B, self.c_hidden, H//self.stride_x, W//self.stride_x]).to(x.device)
        self.snn_func.network_x_list[0]._reset(x1)
        self.Trans._reset(x1)
        self.snn_func.network_x_list[1]._reset(x1_n)
        self.Trans_n._reset(x1_n)

        x1 = torch.zeros([B, self.c_s1, H//self.stride_x, W//self.stride_x]).to(x.device)
        self.network_s_1._reset(x1)
        x1 = torch.zeros([B, self.c_s2, H // (self.stride_x), W // (self.stride_x)]).to(x.device)
        self.network_s_2._reset(x1)

        x1 = torch.zeros([B, self.c_s1, H // (self.stride_x * 2), W // (self.stride_x * 2)]).to(x.device)
        self.network_s_n_1._reset(x1)
        x1 = torch.zeros([B, self.c_s2, H // (self.stride_x * 2), W // (self.stride_x * 2)]).to(x.device)
        self.network_s_n_2._reset(x1)

        z = self.snn_ide_conv(x, time_step=time_step, threshold=threshold, input_type=input_type, solver_type=self.solver, leaky=leaky)

        return z

    def forward(self, x, **kwargs):
        B = x.size(0)
        z = self._forward(x, **kwargs)
        z = self.downsamp(z[0]) + z[1]
        z = self.incre_modules(z)
        z = z.reshape(B, -1)
        y = self.classifier(z)

        return y

