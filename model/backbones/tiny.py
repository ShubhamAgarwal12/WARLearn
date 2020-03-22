# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BasicBlock
from model import Upsample


class Tiny(torch.nn.Module):

    def __init__(self):
        super(Tiny, self).__init__()
        self.layer1 = BasicBlock(3, 16, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer2 = BasicBlock(16, 32, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer3 = BasicBlock(32, 64, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer4 = BasicBlock(64, 128, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer5 = BasicBlock(128, 256, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer6 = BasicBlock(256, 512, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')

        self.layer7 = BasicBlock(512, 1024, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer8 = BasicBlock(1024, 256, 1, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer9 = BasicBlock(256, 512, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.layer10 = BasicBlock(512, 256, 1, 1, 0)  # small

        self.layer11 = BasicBlock(256, 128, 1, 1, 1,
                                  batch_norm=True, activation='leakyrelu')
        self.layer12 = Upsample(2)
        self.layer13 = BasicBlock(128, 256, 3, 1, 1,
                                  batch_norm=True, activation='leakyrelu')
        self.layer14 = BasicBlock(256, 256, 1, 1, 0)  # medium

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool(x)

        x = self.layer3(x)
        x = self.maxpool(x)

        x = self.layer4(x)
        x = self.maxpool(x)

        x = self.layer5(x)
        x = self.maxpool(x)

        x = self.layer6(x)
        x = self.maxpool(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        small_output = self.layer10(x)  # small

        x = self.layer11(small_output)
        x = self.layer12(x)  # upsample
        x = self.layer13(x)
        medium_output = self.layer14(x)  # medium

        return small_output, medium_output