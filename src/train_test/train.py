"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: train_test/train_test.py
 - Contain training code for execution for model.

Version: 1.0

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

from train_test.validation import validation
from utils.optimizer_option import get_optimizer
from utils.load_data import load_cifar10, load_cifar100, load_svhn, load_mnist, load_tiny_imagenet
from utils.lr_decay import adjust_lr


def train(net,
          lr,
          log=None,
          optimizer_option='SGD',
          data='cifar100',
          epochs=350,
          batch_size=128,
          is_train=True,
          net_st=None,
          beta=0.0,
          lrd=10):
    """
    Train the model.
    :param net: model to be trained
    :param lr: learning rate
    :param optimizer_option: optimizer type
    :param data: datasets used to train
    :param epochs: number of training epochs
    :param batch_size: batch size
    :param is_train: whether it is a training process
    :param net_st: uncompressed model
    :param beta: transfer parameter
    """

    net.train()
    if net_st != None:
        net_st.eval()

    if data == 'cifar10':
        trainloader = load_cifar10(is_train, batch_size)
        valloader = load_cifar10(False, batch_size)
    elif data == 'cifar100':
        trainloader = load_cifar100(is_train, batch_size)
        valloader = load_cifar100(False, batch_size)
    elif data == 'svhn':
        trainloader = load_svhn(is_train, batch_size)
        valloader = load_svhn(False, batch_size)
    elif data == 'mnist':
        trainloader = load_mnist(is_train, batch_size)
    elif data == 'tinyimagenet':
        trainloader, valloader = load_tiny_imagenet(is_train, batch_size)
    else:
        exit()

    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = get_optimizer(net, lr, optimizer_option)

    start_time = time.time()
    last_time = 0

    best_acc = 0
    best_param = net.state_dict()

    iteration = 0
    for epoch in range(epochs):
        print("****************** EPOCH = %d ******************" % epoch)
        if log != None:
            log.write("****************** EPOCH = %d ******************\n" % epoch)

        total = 0
        correct = 0
        loss_sum = 0

        # change learning rate
        if epoch == 150 or epoch == 250:
            lr = adjust_lr(lr, lrd=lrd, log=log)
            optimizer = get_optimizer(net, lr, optimizer_option)

        for i, data in enumerate(trainloader, 0):
            iteration += 1

            # foward
            inputs, labels = data
            inputs_V, labels_V = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs, outputs_conv = net(inputs_V)
            loss = criterion(outputs, labels_V)
            if net_st != None:
                outputs_st, outputs_st_conv = net_st(inputs_V)
                # loss += beta * transfer_loss(outputs_conv, outputs_st_conv)
                for i in range(len(outputs_st_conv)):
                    # print("!!!!! %d" % i)
                    if i != (len(outputs_st_conv)-1):
                        loss += beta / 50 * criterion_mse(outputs_conv[i], outputs_st_conv[i].detach())
                    else:
                        loss += beta * criterion_mse(outputs_conv[i], outputs_st_conv[i].detach())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(F.softmax(outputs, -1), 1)
            total += labels_V.size(0)
            correct += (predicted == labels_V).sum()
            loss_sum += loss

            if iteration % 100 == 99:
                now_time = time.time()
                print('accuracy: %f %%; loss: %f; time: %ds'
                      % ((float(100) * float(correct) / float(total)), loss, (now_time - last_time)))
                if log != None:
                    log.write('accuracy: %f %%; loss: %f; time: %ds\n'
                              % ((float(100) * float(correct) / float(total)), loss, (now_time - last_time)))

                total = 0
                correct = 0
                loss_sum = 0
                last_time = now_time

        # validation
        if data == 'tinyimagenet':
            if epoch % 10 == 9:
                net.eval()
                val_acc = validation(net, valloader, log)
                net.train()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_param = net.state_dict()
        else:
            if epoch % 10 == 9:
                best_param = net.state_dict()
                net.eval()
                validation(net, valloader, log)
                net.train()

    print('Finished Training. It took %ds in total' % (time.time() - start_time))
    if log != None:
        log.write('Finished Training. It took %ds in total\n' % (time.time() - start_time))
    return best_param
