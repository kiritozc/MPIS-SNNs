import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import pickle
import sys
import os
import time
import copy
from modules.broyden import broyden, analyze_broyden
from collections import OrderedDict
from modules.deq2d import DEQFunc2d

import logging
logger = logging.getLogger(__name__)


class StaticModule(nn.Module):
    def __init__(self, snn_func, snn_func_copy):
        super(StaticModule, self).__init__()
        self.snn_func = snn_func
        self.snn_func_copy = snn_func_copy

    def forward(self, u, **kwargs):
        time_step = kwargs.get('time_step', 30)
        threshold = kwargs.get('threshold', 30)
        input_type = kwargs.get('input_type', 'constant')
        solver_type = kwargs.get('solver_type', 'broy')
        leaky = kwargs.get('leaky', None)
        get_all_rate = kwargs.get('get_all_rate', False)

        with torch.no_grad():
            u_n = torch.zeros([u.shape[0], u.shape[1], u.shape[2]//2, u.shape[3]//2]).cuda()
                
            u_list = [u, u_n]
            if get_all_rate:
                r_list = self.snn_func.snn_forward(u_list, time_step, output_type='all_rate', input_type=input_type)
                return r_list

            self.snn_func.set_bn_mode_s('eval')
            self.snn_func.save_bn_statistics()

            z1_f, z1_out = self.snn_func.snn_forward(u_list, time_step, input_type=input_type)
            if self.training:
                self.snn_func.set_bn_mode_s('train')
                self.snn_func.restore_bn_statistics_x()

        u_n = torch.zeros([u.shape[0], u.shape[1], u.shape[2] // 2, u.shape[3] // 2]).cuda()
        u_list = [u, u_n]
        z1_out_ = self.snn_func.equivalent_func(z1_out, u_list)
        z1_out = z1_out_

        if self.training:
            self.snn_func_copy.copy(self.snn_func)
            self.snn_func_copy.set_bn_mode_s('eval')

            cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_out_]
            z1_out_ = DEQFunc2d.list2vec(z1_out_)
            if threshold > 0:
                z1_out_ = self.Backward.apply(self.snn_func_copy, z1_out_, u, cutoffs, threshold, solver_type)
            z1_out_ = DEQFunc2d.vec2list(z1_out_, cutoffs)
            z1_out = self.Replace.apply(z1_out_, z1_out)
            self.snn_func_copy.set_bn_mode_s('train')

        return z1_out

    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)

    class Backward(Function):

        @staticmethod
        def forward(ctx, snn_func_copy, z1, u, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.snn_func = snn_func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):

            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1, = ctx.saved_tensors
            u = ctx.u
            args = ctx.args
            sizes, threshold, solver_type = args[-3:]

            snn_func = ctx.snn_func
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = u.clone().detach()

            def infer_from_vec(z, u):
                
                B = z.shape[0]
                z_in = DEQFunc2d.vec2list(z, sizes)
                return DEQFunc2d.list2vec(snn_func.equivalent_func(z_in, u) - z_in)

            with torch.enable_grad():
                y = infer_from_vec(z1_temp, u_temp)

            def g(x):
                y.backward(x, retain_graph=True)   
                res = z1_temp.grad.clone().detach() + grad
                z1_temp.grad.zero_()
                return res

            if solver_type == 'broy':
                eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
                dl_df_est = torch.zeros_like(grad)

                result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
                dl_df_est = result_info['result']
                nstep = result_info['nstep']
                lowest_step = result_info['lowest_step']
            else:
                dl_df_est = grad
                for i in range(threshold):
                    dl_df_est = (dl_df_est + g(dl_df_est)) / 2.
            
            if threshold > 30:
                torch.cuda.empty_cache()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            print("dl_df_est:", dl_df_est)
            return (None, dl_df_est, None, *grad_args)


class NeuroModule(nn.Module):

    def __init__(self, snn_func, snn_func_copy):
        super(NeuroModule, self).__init__()
        self.snn_func = snn_func
        self.snn_func_copy = snn_func_copy

    def forward(self, u, **kwargs):
        time_step = kwargs.get('time_step', 30)
        threshold = kwargs.get('threshold', 30)
        input_type = kwargs.get('input_type', 'spike')
        solver_type = kwargs.get('solver_type', 'broy')
        leaky = kwargs.get('leaky', None)
        get_all_rate = kwargs.get('get_all_rate', False)

        with torch.no_grad():
            if len(u.size()) == 3:
                u = u.permute(2, 0, 1)
            else:
                u = u.permute(4, 0, 1, 2, 3)
            u_n = torch.zeros([u.shape[0], u.shape[1], u.shape[2], u.shape[3] // 2, u.shape[4] // 2]).cuda()
            u_list = [u, u_n]
            if get_all_rate:
                r_list = self.snn_func.snn_forward(u_list, time_step, output_type='all_rate', input_type=input_type)
                return r_list

            self.snn_func.set_bn_mode_s('eval')
            self.snn_func.save_bn_statistics()

            z1_f, z1_out = self.snn_func.snn_forward(u_list, time_step, input_type=input_type)
            if self.training:
                self.snn_func.set_bn_mode_s('train')
                self.snn_func.restore_bn_statistics_x()

        if leaky == None:
            u = torch.mean(u, dim=0)
        else:
            leaky_ = 1.
            u_ = u[time_step - 1]
            for i in range(time_step):
                leaky_ *= leaky
                u_ += u[time_step - 2 - i] * leaky_
            u_ /= (1 - leaky_ * leaky) / (1 - leaky)
            u = u_
        u_n = torch.zeros([u.shape[0], u.shape[1], u.shape[2] // 2, u.shape[3] // 2]).cuda()
        u_list = [u, u_n]
        z1_out_ = self.snn_func.equivalent_func(z1_out, u_list)
        z1_out = z1_out_

        if self.training:
            self.snn_func_copy.copy(self.snn_func)
            self.snn_func_copy.set_bn_mode_s('eval')

            cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_out_]
            z1_out_ = DEQFunc2d.list2vec(z1_out_)
            if threshold > 0:
                z1_out_ = self.Backward.apply(self.snn_func_copy, z1_out_, u, cutoffs, threshold, solver_type)

            z1_out_ = DEQFunc2d.vec2list(z1_out_, cutoffs)

            z1_out = self.Replace.apply(z1_out_, z1_out)

            self.snn_func_copy.set_bn_mode_s('train')

        return z1_out

    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)

    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass.

        """

        @staticmethod
        def forward(ctx, snn_func_copy, z1, u, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.snn_func = snn_func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            # torch.cuda.empty_cache()

            # grad should have dimension (bsz x d_model x seq_len) to be consistent with the solver
            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1, = ctx.saved_tensors
            u = ctx.u
            args = ctx.args
            sizes, threshold, solver_type = args[-3:]

            snn_func = ctx.snn_func
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = u.clone().detach()

            def infer_from_vec(z, u):
                # change the dimension of z
                B = z.shape[0]
                z_in = DEQFunc2d.vec2list(z, sizes)
                return DEQFunc2d.list2vec(snn_func.equivalent_func(z_in, u, f_type='last') - z_in)

            with torch.enable_grad():
                y = infer_from_vec(z1_temp, u_temp)

            def g(x):
                y.backward(x, retain_graph=True)  # Retain for future calls to g
                res = z1_temp.grad.clone().detach() + grad
                z1_temp.grad.zero_()
                return res

            if solver_type == 'broy':
                eps = 2e-10 * np.sqrt(bsz * seq_len * d_model)
                dl_df_est = torch.zeros_like(grad)

                result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
                dl_df_est = result_info['result']
                nstep = result_info['nstep']
                lowest_step = result_info['lowest_step']
            else:
                dl_df_est = grad
                for i in range(threshold):
                    dl_df_est = (dl_df_est + g(dl_df_est)) / 2.

            if threshold > 30:
                torch.cuda.empty_cache()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            print("dl_df_est:", dl_df_est)
            return (None, dl_df_est, None, *grad_args)

