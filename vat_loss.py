# VAT "https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py"

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-12
    return d


# from train_trans.py

def structure_loss(pred, mask):
    delta = 1e-9
    pred_sig = torch.sigmoid(pred) # P pred_sig
    mask_sig = torch.sigmoid(mask) # Q mask_sig
    lds_1 = torch.mean(mask_sig*(torch.log(mask_sig+delta)-torch.log(pred_sig+delta)))
    lds_0 = torch.mean((1.0-mask_sig)*(torch.log(1.0-mask_sig+delta)-torch.log(1.0-pred_sig+delta)))

    return lds_1 + lds_0

"""
def structure_loss(mask, pred):
    #flatten to transfer to distribution
    flatten = torch.nn.Flatten()
    pred_flat = flatten(pred)

    pred_dist = F.softmax(pred_flat)
    mask_dist = F.softmax(flatten(mask))

    lds = F.kl_div(pred_dist.log(), mask_dist, None, None, 'batchmean')  # log_target=True)
    return lds
"""

class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        #print(x.max(), x.min())
        with torch.no_grad():
            # normal vat
            # pred = F.softmax(model(x), dim=1)

            # changed vat for this
            # ---- forward ----
            pred_lateral_map_4, pred_lateral_map_3, pred_lateral_map_2 = model(x)

        # prepare random unit tensor
        #d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.rand(x.shape).sub(0.5)#.cuda()
        d = Variable(_l2_normalize(d)).cuda()

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                lateral_map_4_hat, lateral_map_3_hat, lateral_map_2_hat = model(x + self.xi * d)
                # logp_hat = F.log_softmax(pred_hat, dim=1)
                # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                # adv_distance.backward()
                # ---- loss function ----
                #adv_loss4 = structure_loss(lateral_map_4_hat, pred_lateral_map_4)  # BiFusion map
                #adv_loss3 = structure_loss(lateral_map_3_hat, pred_lateral_map_3)  # Transformer map
                adv_loss2 = structure_loss(lateral_map_2_hat, pred_lateral_map_2)  # Joint map
                #adv_loss = 0.5 * adv_loss2 + 0.2 * adv_loss3 + 0.3 * adv_loss4
                adv_loss = adv_loss2
                adv_loss.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps * (torch.sum(x ** 2, dim=[1, 2, 3], keepdim=True) ** 0.5)
            lateral_map_4_hat, lateral_map_3_hat, lateral_map_2_hat = model(x + r_adv)
            # logp_hat = F.log_softmax(pred_hat, dim=1)
            #adv_loss4 = structure_loss(lateral_map_4_hat, pred_lateral_map_4)  # BiFusion map
            #adv_loss3 = structure_loss(lateral_map_3_hat, pred_lateral_map_3)  # Transformer map
            adv_loss2 = structure_loss(lateral_map_2_hat, pred_lateral_map_2)  # Joint map
            #adv_loss = 0.5 * adv_loss2 + 0.2 * adv_loss3 + 0.3 * adv_loss4
            lds = adv_loss2
            # lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class MAVATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(MAVATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, gt):
        #print(x.max(), x.min())
        #mask = gt.cuda() # sekkai nomi setsudou
        mask = (-1.0 * (gt - 1.0)).cuda(); #haikei nomi setsudou
        with torch.no_grad():
            # normal vat
            # pred = F.softmax(model(x), dim=1)

            # changed vat for this
            # ---- forward ----
            pred_lateral_map_4, pred_lateral_map_3, pred_lateral_map_2 = model(x)

        # prepare random unit tensor
        #d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.rand(x.shape).sub(0.5).cuda() * mask#.cuda()
        d = Variable(_l2_normalize(d)).cuda() * mask

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                lateral_map_4_hat, lateral_map_3_hat, lateral_map_2_hat = model(x + self.xi * d * mask)
                # logp_hat = F.log_softmax(pred_hat, dim=1)
                # adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                # adv_distance.backward()
                # ---- loss function ----
                #adv_loss4 = structure_loss(lateral_map_4_hat, pred_lateral_map_4)  # BiFusion map
                #adv_loss3 = structure_loss(lateral_map_3_hat, pred_lateral_map_3)  # Transformer map
                adv_loss2 = structure_loss(lateral_map_2_hat, pred_lateral_map_2)  # Joint map
                #adv_loss = 0.5 * adv_loss2 + 0.2 * adv_loss3 + 0.3 * adv_loss4
                adv_loss = adv_loss2
                adv_loss.backward()
                d = _l2_normalize(d.grad) * mask
                model.zero_grad()

            # calc LDS
            r_adv = (d * mask) * self.eps * (torch.sum(x ** 2, dim=[1, 2, 3], keepdim=True) ** 0.5)
            lateral_map_4_hat, lateral_map_3_hat, lateral_map_2_hat = model(x + r_adv)
            # logp_hat = F.log_softmax(pred_hat, dim=1)
            #adv_loss4 = structure_loss(lateral_map_4_hat, pred_lateral_map_4)  # BiFusion map
            #adv_loss3 = structure_loss(lateral_map_3_hat, pred_lateral_map_3)  # Transformer map
            adv_loss2 = structure_loss(lateral_map_2_hat, pred_lateral_map_2)  # Joint map
            #adv_loss = 0.5 * adv_loss2 + 0.2 * adv_loss3 + 0.3 * adv_loss4
            lds = adv_loss2
            # lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
