"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/load_data.py
 - Contain source code for loading data.

Version: 1.0

"""
import sys
sys.path.append('../')
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os, shutil, random

# CIFAR10 data
def load_cifar10(is_train=True, batch_size=128):
    """
    Load cifar-10 datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if is_train:
        # dataset
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
        # valset = torchvision.datasets.CIFAR10(root='./data',
        #                                       train=True,
        #                                       download=True,
        #                                       transform=transform_train)
        #
        # # split validation set from training set
        # num_train = len(trainset)
        # valid_size = 0.2
        # indices = list(range(num_train))
        # split = int(np.floor(valid_size * num_train))
        #
        # random_seed = 5
        # np.random.seed(random_seed)
        # np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        # dataloader
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # sampler=train_sampler,
                                                  num_workers=2,
                                                  shuffle=True)
        # valloader = torch.utils.data.DataLoader(valset,
        #                                         batch_size=batch_size,
        #                                         sampler=valid_sampler,
        #                                         num_workers=2)
        return trainloader #, valloader

    else:
        testset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=False,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader


# CIFAR100 data
def load_cifar100(is_train=True, batch_size=128):
    """
    Load cifar-100 datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if is_train:
        # dataset
        trainset = torchvision.datasets.CIFAR100(root='./data',
                                                 train=True,
                                                 download=False,
                                                 transform=transform_train)
        # valset = torchvision.datasets.CIFAR100(root='./data',
        #                                        train=True,
        #                                        download=True,
        #                                        transform=transform_train)
        #
        # # split validation set from training set
        # num_train = len(trainset)
        # valid_size = 0.2
        # indices = list(range(num_train))
        # split = int(np.floor(valid_size * num_train))
        #
        # random_seed = 5
        # np.random.seed(random_seed)
        # np.random.shuffle(indices)
        #
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        # dataloader
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # sampler=train_sampler,
                                                  num_workers=2,
                                                  shuffle=True)
        # valloader = torch.utils.data.DataLoader(valset,
        #                                         batch_size=batch_size,
        #                                         sampler=valid_sampler,
        #                                         num_workers=2)
        return trainloader #, valloader

    else:
        testset = torchvision.datasets.CIFAR100(root='./data',
                                                train=False,
                                                download=False,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader


# SVHN data
def load_svhn(is_train=True, batch_size=128):
    """
    Load SVHN datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    if is_train:
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        return trainloader

    else:
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        return testloader


# MNIST data
def load_mnist(is_train=True, batch_size=128):
    """
    Load MNIST datasets.
    :param is_train: if true, load train_test/val data; else load test data.
    :param batch_size: batch_size of train_test data
    """
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    if is_train:
        trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        return train_loader

    else:
        testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        return test_loader


# tiny imagenet 200
def load_tiny_imagenet(is_train=True, batch_size=128):
    """
        Load tiny imagenet datasets.
        :param is_train: if true, load train_test/val data; else load test data.
        :param batch_size: batch_size of train_test data
    """
    # path
    data_path = '../../../datasets/tiny-imagenet-200/'

    # transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        # path
        train_dir = data_path + 'train/'
        val_dir = data_path + 'val/'

        # transforms
        transforms_train = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transforms_val = transforms.Compose([transforms.ToTensor(), normalize])

        # dataset
        train_set = datasets.ImageFolder(train_dir, transforms_train)
        val_set = datasets.ImageFolder(val_dir, transforms_val)

        # dataloader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)

        return train_loader, val_loader

    else:
        # path
        test_dir = data_path + 'test/'

        # transforms
        transforms_test = transforms.Compose([transforms.ToTensor(), normalize])

        # set
        test_set = datasets.ImageFolder(test_dir, transforms_test)

        # loader
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return test_loader


def preprocessing_val():
    """
        Preprocess tiny imagenet validation datasets.
        Only used for im-preprocessing tiny imagenet.
    """
    # move val data to corresponding label folders
    val_base_path = '../../../datasets/tiny-imagenet-200/val/'
    with open(val_base_path + 'val_annotations.txt', 'r') as val_anno:
        for line in val_anno:
            words = line.split('\t')
            image_name = words[0]
            label_name = words[1]
            print('image name: ' + image_name + ', label name: ' + label_name)
            src_path = val_base_path + 'images/' + image_name
            dest_path = val_base_path + label_name + '/'
            print(src_path)
            print(dest_path)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.move(src_path, dest_path)


def preprocessing_test():
    """
        Preprocess tiny imagenet test datasets.
        Only used for im-preprocessing tiny imagenet.
        Since tiny imagenet didn't provide labels for test dataset,
        we divided val dataset into two part - half as val, half as test.
    """
    # create folders for test data
    data_base_path = '../../../datasets/tiny-imagenet-200/'
    val_path = data_base_path + 'val/'
    test_path = data_base_path + 'test/'
    print(val_path)
    print(test_path)
    with open(data_base_path + 'wnids.txt', 'r') as wnids:
        i = 0
        for line in wnids:
            i += 1
            label = line.split('\n')[0]
            print('%d. label: ' % i + label)
            val_label_path = val_path + label + '/'
            test_label_path = test_path + label + '/'
            if os.path.exists(val_label_path):
                if not os.path.exists(test_label_path):
                    os.mkdir(test_label_path)
                # random choose images
                path = os.listdir(val_label_path)
                sample = random.sample(path, 25)
                print(len(sample))
                print(sample)
                for name in sample:
                    shutil.move(val_label_path + name, test_label_path)


# imagenet
def load_imagenet(is_train=True, batch_size=128):
    """
        Load tiny imagenet datasets.
        :param is_train: if true, load train_test/val data; else load test data.
        :param batch_size: batch_size of train_test data
    """
    # path
    data_path = '../../../../../../data/quanchun/ImageNet_data/'

    # transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        # path
        train_dir = data_path + 'train/'
        val_dir = data_path + 'val/'

        # transforms
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # dataset
        train_set = datasets.ImageFolder(train_dir, transforms_train)
        val_set = datasets.ImageFolder(val_dir, transforms_val)

        # print('Number of classes: %d' % len(train_set.classes))
        # print(train_set.classes)

        # dataloader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)

        return train_loader, val_loader

    else:
        # path
        val_dir = data_path + 'val/'

        # transforms
        transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # dataset
        val_set = datasets.ImageFolder(val_dir, transforms_val)

        # dataloader
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=2)

        return val_loader
