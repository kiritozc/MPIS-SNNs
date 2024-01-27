

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    
    
    
    
    while alpha1 > amin:       
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    
    
    
    
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    


def matvec(part_Us, part_VTs, x):
    
    
    
    
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     


def broyden(g, x0, threshold, eps, ls=False, name="unknown"):
    bsz, total_hsize, n_elem = x0.size()
    dev = x0.device
    
    x_est = x0           
    gx = g(x_est)        
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(threshold, 27)
    
    
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(dev)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(dev)
    update = -gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]
    
    
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep
    
    while new_objective >= eps and nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        try:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item())   
        except:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item() + 1e-9)
        new_trace.append(new2_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            break
        if new_objective < 3*eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            
            break
        if new_objective > init_objective * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:(nstep-1)], VTs[:,:(nstep-1)]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)
        
    Us, VTs = None, None
    return {"result": lowest_xest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "new_trace": new_trace,
            "eps": eps,
            "threshold": threshold}


def analyze_broyden(res_info, err=None, judge=True, name='forward', training=True, save_err=True):
    """
    For debugging use only :-)
    """
    res_est = res_info['result']
    nstep = res_info['nstep']
    diff = res_info['diff']
    diff_detail = res_info['diff_detail']
    prot_break = res_info['prot_break']
    trace = res_info['trace']
    eps = res_info['eps']
    threshold = res_info['threshold']
    if judge:
        return nstep >= threshold or (nstep == 0 and (diff != diff or diff > eps)) or prot_break or torch.isnan(res_est).any()
    
    assert (err is not None), "Must provide err information when not in judgment mode"
    prefix, color = ('', 'red') if name == 'forward' else ('back_', 'blue')
    eval_prefix = '' if training else 'eval_'
    
    
    if torch.isnan(res_est).any():
        msg = colored(f"WARNING: nan found in Broyden's {name} result. Diff: {diff}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}nan.pkl', 'wb'))
        return (1, msg, res_info)
        
    
    if nstep == 0 and (diff != diff or diff > eps):
        msg = colored(f"WARNING: Bad Broyden's method {name}. Why?? Diff: {diff}. STOP.", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}badbroyden.pkl', 'wb'))
        return (2, msg, res_info)
        
    
    if prot_break and np.random.uniform(0,1) < 0.05:
        msg = colored(f"WARNING: Hit Protective Break in {name}. Diff: {diff}. Total Iter: {len(trace)}", color)
        print(msg)
        if save_err: pickle.dump(err, open(f'{prefix}{eval_prefix}prot_break.pkl', 'wb'))
        return (3, msg, res_info)
        
    return (-1, '', res_info)
