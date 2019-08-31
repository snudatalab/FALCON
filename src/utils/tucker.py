"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/tucker.py
 - Contain source code related to tucker decomposition.
 - Code is got from https://github.com/CasvandenBogaard/VBMF/blob/master/VBMF.py

Version: 1.0

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

from __future__ import division

import torch
import torch.nn as nn

import tensorly as tl
import tensorly.decomposition as decomp
from tensorly.tucker_tensor import tucker_to_tensor

import numpy as np
from scipy.optimize import minimize_scalar

# tucker
tl.set_backend('pytorch')

__all__ = ['EVBMF']

class Tucker2DecomposedConv(nn.Module):
    """
    The most basic convolutional module with tucker-2 decomposition

    Reference
        - Compression of Deep Convolutional Neural Networks
          for Fast and Low Power Mobile Applications
    """

    def __init__(self, conv_layer, ranks=None, hooi=False, multiplier=1):
        """
        Constructor for Tucker2DecomposedConv Layer

        @param conv_layer: Original layer to decompose
        @param ranks: Projection rank (default None)
            If None, run EVBMF to calculate rank
        @param hooi: whether using HOOI to initialize layer weight (default F)
            If true, it is same as tucker decomposition
            Else, it initializes randomly
        """

        super(Tucker2DecomposedConv, self).__init__()

        # get device
        device = conv_layer.weight.device

        # get weight and bias
        weight = conv_layer.weight.data
        bias = conv_layer.bias
        if bias is not None:
            bias = bias.data

        out_channels, in_channels, _, _ = weight.shape

        if ranks is None:
            # run EVBMF and get estimated ranks
            unfold_0 = tl.base.unfold(weight, 0)
            unfold_1 = tl.base.unfold(weight, 1)
            _, diag_0, _, _ = EVBMF(unfold_0)
            _, diag_1, _, _ = EVBMF(unfold_1)
            ranks = [diag_0.shape[0], diag_1.shape[1]]
            out_rank, in_rank = ranks

            # print('Projection Ranks: [%d, %d]' % (in_rank, out_rank))
        else:
            if isinstance(ranks, int):
                in_rank, out_rank = ranks
            else:
                out_rank, in_rank = ranks
        # print('pre_ranks:', end='')
        # print(ranks)
        if multiplier != 1:
            in_rank_new = int(float(in_rank) * multiplier)
            out_rank_new = int(float(out_rank) * multiplier)
            if in_rank_new <= in_channels:
                in_rank = in_rank_new
            else:
                in_rank = in_channels
            if out_rank_new <= out_channels:
                out_rank = out_rank_new
            else:
                out_rank = out_channels
            if in_rank == 0:
                in_rank += 1
            if out_rank == 0:
                out_rank += 1
            ranks = [out_rank, in_rank]
        # print('[in_rank, out_rank]: [%d, %d]' % (in_rank, out_rank))
        # print('ranks:', end='')
        # print(ranks)

        # initialize layers
        self.in_channel_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=conv_layer.dilation,
            bias=False).to(device)

        self.core_layer = nn.Conv2d(
            in_channels=in_rank,
            out_channels=out_rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            bias=False).to(device)

        self.out_channel_layer = nn.Conv2d(
            in_channels=out_rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=conv_layer.dilation,
            bias=conv_layer.bias is not None).to(device)

        if hooi:
            # use traditional tucker2 decomposition

            core, [out_channel_factor,
                   in_channel_factor] = decomp.partial_tucker(
                       weight, modes=[0, 1], ranks=ranks, init='svd')

            # assign bias
            if self.out_channel_layer.bias is not None:
                self.out_channel_layer.bias.data = conv_layer.bias.data

            # assign weights
            self.in_channel_layer.weight.data = torch.transpose(in_channel_factor, 1, 0).unsqueeze(-1).unsqueeze(-1)
            self.out_channel_layer.weight.data = out_channel_factor.unsqueeze(-1).unsqueeze(-1)
            self.core_layer.weight.data = core

    def forward(self, x):
        """
        Run forward propagation
        """
        x = self.in_channel_layer(x)
        x = self.core_layer(x)
        x = self.out_channel_layer(x)

        return x

    def recover(self):
        """
        Recover original tensor from decomposed tensor

        @return: 4D weight tensor with original layer's shape
        """

        # get core
        core = self.core_layer.weight.data

        # get factor
        out_factor = self.out_channel_layer.weight.data.squeeze()

        in_factor = self.in_channel_layer.weight.data.squeeze()
        in_factor = torch.transpose(in_factor, 1, 0)

        # recover
        recovered = tucker_to_tensor(core, [out_factor, in_factor])

        return recovered


# VBMF
def EVBMF(Y, sigma2=None, H=None):
    """
    Implementation of the analytical solution to
    Empirical Variational Bayes Matrix Factorization.

    This function can be used to calculate the analytical solution to empirical VBMF.
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."

    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.

    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.

    sigma2 : int or None (default=None)
        Variance of the noise on Y.

    H : int or None (default = None)
        Maximum rank of the factorized matrices.

    Returns
    -------
    U : numpy-array
        Left-singular vectors.

    S : numpy-array
        Diagonal matrix of singular values.

    V : numpy-array
        Right-singular vectors.

    post : dictionary
        Dictionary containing the computed posterior values.

    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.

    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.
    """
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = np.linalg.svd(Y.cpu())
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.
    if H < L:
        residual = np.sum(np.sum(Y**2) - np.sum(s**2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (np.sum(s**2) + residual) / (L * M)
        lower_bound = np.max([s[eH_ub + 1]**2 / (M * xubar), np.mean(s[eH_ub + 1:]**2) / M])

        scale = 1.  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2_opt = minimize_scalar(
            EVBsigma2,
            args=(L, M, s, residual, xubar),
            bounds=[lower_bound, upper_bound],
            method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))
    pos = np.sum(s > threshold)

    # Formula (15) from [2]
    d = np.multiply(
        s[:pos] / 2, 1 - np.divide((L + M) * sigma2, s[:pos]**2) +
        np.sqrt((1 - np.divide((L + M) * sigma2, s[:pos]**2))**2 - 4 * L * M * sigma2**2 / s[:pos]**4))

    # Computation of the posterior
    post = {}
    post['ma'] = np.zeros(H)
    post['mb'] = np.zeros(H)
    post['sa2'] = np.zeros(H)
    post['sb2'] = np.zeros(H)
    post['cacb'] = np.zeros(H)

    tau = np.multiply(d, s[:pos]) / (M * sigma2)
    delta = np.multiply(
        np.sqrt(np.divide(M * d, L * s[:pos])), 1 + alpha / tau)

    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2 * delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos]) / (L * M))
    post['sigma2'] = sigma2
    post['F'] = 0.5 * (
        L * M * np.log(2 * np.pi * sigma2) + (residual + np.sum(s**2)) / sigma2 + np.sum(M * np.log(tau + 1) + L * np.log(tau / alpha + 1) - M * tau))

    return U[:, :pos], np.diag(d), V[:, :pos], post


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    """
    For case when sigma2 is unspecified.
    function for EVBMF function (minimization objective function)

    @param sigma2: current value (do not pass arg explicitly)
    @param L: width of target factorization matrix
    @param M: height of target factorization matrix
    @param s: singular values
    @param residual: residual value (squared difference of target matrix and singular value)
    @param xubar: scaled taubar

    @return: value of calculated sigma2 from given args
    """
    H = len(s)

    alpha = L / M
    x = s**2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau = lambda x, alpha: 0.5 * (x - (1 + alpha) + np.sqrt((x - (1 + alpha))**2. - 4 * alpha))
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1 + 1, z1)))
    term4 = alpha * np.sum(np.log(tau_z1 / alpha + 1))

    obj = term1 + term2 + term3 + term4 + residual / (M * sigma2) + (L - H) * np.log(sigma2)

    return obj

