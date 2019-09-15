import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class A3CActor(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden=64):
        super(A3CActor, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.action_mean = nn.Linear(hidden, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.value_head = nn.Linear(hidden, 1)

        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std, self.value_head]
        self.module_list_old = [None] *len(self.module_list_current)
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = deepcopy(self.module_list_current[i])

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            action_mean = self.module_list_old[2](x)
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)

            value = self.module_list_old[4](x)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))

            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

            value = self.value_head(x)

        return action_mean, action_log_std, action_std, value
