"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: models/vgg.py
 - Contain VGG model.

Version: 1.0
"""

import torch.nn as nn
from models.falcon import EHPdecompose
from models.dsconv import DepthwiseSeparableConv
from utils.tucker import Tucker2DecomposedConv
from models.mobileconvv2 import Block as Block_MobileConvV2
from models.shuffleunit import ShuffleUnit
from models.shuffleunitv2 import ShuffleUnitV2


class VGG(nn.Module):
    """
    Description: VGG class.
    """

    # configures of different models
    cfgs_VGG16 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512, 2)]

    cfgs_VGG19 = [(3, 64), (64, 64, 2),
                  (64, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512, 2)]

    cfgs_VGG_en = [(3, 64), (64, 64), (64, 64, 2),
                  (64, 128), (128, 128), (128, 128, 2),
                  (128, 256), (256, 256), (256, 256), (256, 256), (256, 256), (256, 256, 2),
                  (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512, 2),
                  (512, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512, 2)]

    def __init__(self, num_classes=10, which='VGG16'):
        """
        Initialize VGG Model as argument configurations.
        :param num_classes: number of classification labels
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        """

        super(VGG, self).__init__()
        self.conv = nn.Sequential()

        self.layers = self._make_layers(which)

        self.avgPooling = nn.AvgPool2d(2, 2)

        output_size = 512
        self.fc = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, which):
        """
        Make standard-conv Model layers.
        :param which: choose a model architecture from VGG16/VGG19/MobileNet
        """

        layers = []
        if which == 'VGG16':
            self.cfgs = self.cfgs_VGG16
        elif which == 'VGG19':
            self.cfgs = self.cfgs_VGG19
        else:
            pass

        for cfg in self.cfgs:
            # if len(cfg) == 3:
            #     layers.append(nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=cfg[2], padding=0))
            # else:
            #     layers.append(nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(cfg[1]))
            layers.append(nn.ReLU())
            if len(cfg) == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Run forward propagation"""
        out_conv = self.conv(x)
        out_conv = self.layers(out_conv)
        # for i in range(len(self.layers)):
        #     out_conv = self.layers[i](out_conv)
        #     if isinstance(self.layers[i], EHPdecompose):
        #         if self.layers[i].pw.in_channels != self.layers[i].pw.out_channels:
        #             # print(self.layers[i])
        #             out_list.append(out_conv)
        #     elif isinstance(self.layers[i], nn.Conv2d):
        #         if self.layers[i].in_channels != self.layers[i].out_channels:
        #             # print(self.layers[i])
        #             out_list.append(out_conv)
        #     else:
        #         pass
        if out_conv.size(2) != 1:
            out_conv = self.avgPooling(out_conv)
        out = out_conv.view(out_conv.size(0), -1)
        out = self.fc(out)
        return out, out_conv

    def falcon(self, rank, init=True, alpha=1.0, bn=False, relu=False, groups=1):
        """
        Replace standard convolution by FALCON
        :param rank: rank of EHP
        :param init: whether initialize FALCON with EHP decomposition tensors
        :param bn: whether add batch normalization after FALCON
        :param relu: whether add ReLU function after FALCON
        :param groups: number of groups for pointwise convolution
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                compress = EHPdecompose(self.layers[i], rank, init, alpha=alpha, bn=bn, relu=relu, groups=groups)
                self.layers[i] = compress
            if isinstance(self.layers[i], nn.BatchNorm2d):
                device = self.layers[i].weight.device
                self.layers[i] = nn.BatchNorm2d(self.layers[i].num_features).to(device)

    def dsc(self):
        """
        Replace standard convolution by depthwise separable convolution
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                if shape[1] > 3:
                    compress = DepthwiseSeparableConv(self.layers[i])
                    self.layers[i] = compress

    def compress_tucker(self, loss_type='l2', rank=None, tucker_init=True, multiplier=1):
        """
        Compress standard convolution via tucker decomposition
        :param distill: whether use distillation
        :param loss_type: distillation loss
        :param rank: projection rank (use evbmf if None)
        :param tucker_init: whether initialize with tucker decomposition tensors
        :param multiplier: a parameter for adjusting estimated rank
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                rank = None

                # if shape[0] != 3:
                # compress
                compress = Tucker2DecomposedConv(
                    self.layers[i],
                    rank,
                    tucker_init,
                    multiplier=multiplier
                )
                self.layers[i] = compress
            if isinstance(self.layers[i], nn.BatchNorm2d):
                device = self.layers[i].weight.device
                self.layers[i] = nn.BatchNorm2d(self.layers[i].num_features).to(device)

    def mobileconvv2(self, expansion):
        """
        Replace standard convolution by MobileConvV2
        :param expansion: (t) increase the number of input channels M into tM
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                if shape[1] > 3:
                    compress = Block_MobileConvV2(self.layers[i].in_channels,
                                                  self.layers[i].out_channels,
                                                  expansion, 1)
                    self.layers[i] = compress
        layers = []
        for i in range(len(self.layers)):
            if (isinstance(self.layers[i], nn.BatchNorm2d) and isinstance(self.layers[i-1], Block_MobileConvV2)) \
                    or (isinstance(self.layers[i], nn.ReLU) and isinstance(self.layers[i-2], Block_MobileConvV2)):
                pass
            else:
                layers.append(self.layers[i])
        self.layers = nn.Sequential(*layers)

    def shuffleunit(self, groups=2, alpha=1):
        """
        Replace standard convolution by ShuffleUnit
        :param groups: number of groups for group convolution
        :param alpha: width multiplier
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                if shape[1] == 3:
                    self.layers[i] = nn.Conv2d(3, self.layers[i].out_channels, kernel_size=3, padding=1)
                else:
                    combine = 'concat' if self.layers[i].in_channels != self.layers[i].out_channels else 'add'
                    compress = ShuffleUnit(int(self.layers[i].in_channels * alpha),
                                           int(self.layers[i].out_channels * alpha),
                                           grouped_conv=True,
                                           groups=self.groups,
                                           combine=combine)
                    self.layers[i] = compress
        layers = []
        for i in range(len(self.layers)):
            if (isinstance(self.layers[i], nn.BatchNorm2d) and isinstance(self.layers[i - 1], ShuffleUnit)) \
                    or (isinstance(self.layers[i], nn.ReLU) and isinstance(self.layers[i - 2], ShuffleUnit)):
                pass
            else:
                layers.append(self.layers[i])
        self.layers = nn.Sequential(*layers)

    def shuffleunitv2(self):
        """
        Replace standard convolution by ShuffleUnitV2
        """
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                # print(self.layers[i])
                shape = self.layers[i].weight.shape
                if shape[1] == 3:
                    self.layers[i] = nn.Conv2d(3, self.layers[i].out_channels, kernel_size=3, padding=1)
                else:
                    compress = ShuffleUnitV2(int(self.layers[i].in_channels),
                                            int(self.layers[i].out_channels),
                                            stride=self.layers[i].stride)
                    self.layers[i] = compress
        layers = []
        for i in range(len(self.layers)):
            if (isinstance(self.layers[i], nn.BatchNorm2d) and isinstance(self.layers[i - 1], ShuffleUnit)) \
                    or (isinstance(self.layers[i], nn.ReLU) and isinstance(self.layers[i - 2], ShuffleUnit)):
                pass
            else:
                layers.append(self.layers[i])
        self.layers = nn.Sequential(*layers)
