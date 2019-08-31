"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/optimizer_option.py
 - Contain source code for choosing optimizer.

Version: 1.0

"""

import sys
import torch.optim as optim


def get_optimizer(net, lr, optimizer='SGD',  weight_decay=1e-4, momentum=0.9):
    """
    Get optimizer accroding to arguments.
    :param net: the model
    :param lr: learing rate
    :param optimizer: choose an optimizer - SGD/Adagrad/Adam/RMSprop
    :param weight_decay: weight decay rate
    :param momentum: momentum of the optimizer
    """
    if optimizer == 'SGD':
        return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                         lr=lr,
                         weight_decay=weight_decay,
                         momentum=momentum)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=lr,
                             weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=lr,
                          weight_decay=weight_decay,
                          )
    elif optimizer == 'RMSprop':
        return optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=lr,
                             weight_decay=weight_decay,
                             momentum=momentum)
    else:
        sys.exit('Wrong Instruction! '
                 'The optimizer must be one of SGD/Adagrad/Adam/RMSprop.')
