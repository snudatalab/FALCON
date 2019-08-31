"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/lr_decay.py
 - Contain source code for updating learning rate.

Version: 1.0

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""


def adjust_lr(lr, lrd=10, log=None):
    """
    Update learnign rate.
    :param lr: original learning rate.
    """
    lr = lr / lrd
    print("learning rate change to %f" % lr)
    if log != None:
        log.write(("learning rate change to %f\n" % lr))
    return lr

