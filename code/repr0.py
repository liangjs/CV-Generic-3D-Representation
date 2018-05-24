from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision import transforms, utils, models
from torch.optim import lr_scheduler
import os
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

import warnings
warnings.filterwarnings("ignore")

from repr0loader import *


class ReprNet(nn.Module):

    def __init__(self):
        super(ReprNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 7, 1)
        self.conv2 = nn.Conv2d(20, 40, 5, 1)
        self.conv3 = nn.Conv2d(40, 80, 4, 1)
        self.conv4 = nn.Conv2d(80, 160, 4, 2)
        self.fc1 = nn.Linear(160, 500)

    def forward(self, x):
        # x.size() = [?, 3, 101, 101]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        # x.size() = [?, 160, 1, 1]
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        return x


class CompdNet(nn.Module):

    def __init__(self):
        super(CompdNet, self).__init__()
        self.rnet = ReprNet()
        self.fc = nn.Linear(1000, 500)

    def forward(self, x0):
        # x0.size() = [?, 6, 101, 101]
        x, y = torch.split(x0, 3, 1)
        x = self.rnet(x)
        y = self.rnet(y)
        x0 = torch.cat((x, y), 1)
        x0 = F.relu(self.fc(x0))
        return x0


class LearnNet(nn.Module):
    
    def __init__(self):
        super(LearnNet, self).__init__()
        self.cnet = CompdNet()
        self.fc_pose = nn.Linear(500, 3)
        self.fc_match = nn.Linear(500, 2)

    def forward(self, x):
        x = self.cnet(x)
        pose = self.fc_pose(x)
        match = self.fc_match(x)
        return (pose, match)


class FPoseLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ans = input.clone()
        t = input > 1
        ans[t] = 1 + torch.log(input[t])
        return ans

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        ans = grad_output.clone()
        t = input > 1
        ans[t] = 1.0 / input[t] * grad_output[t]
        return ans


fposeloss = FPoseLoss.apply

def joint_loss(my_pose, my_match, true_pose, true_match):
    global fposeloss
    factor = 1
    if true_match.sum() >= 1: # only train camera pose on matched data
        pose_loss = F.mse_loss(my_pose[true_match], true_pose[true_match], reduce=False)
        pose_loss = pose_loss.sum(1)
        pose_loss = fposeloss(pose_loss)
        pose_loss = pose_loss.mean()
    else:
        pose_loss = 0
    match_loss = F.cross_entropy(my_match, true_match)
    return pose_loss + factor * match_loss


if __name__ == "__main__":

    net = LearnNet()
    # net.load_state_dict(torch.load('.pkl'))
    # print('load parameters')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("device:", device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.1)

    trainloader = TrainLoader(shuffleTar=False, keepTar=True)

    output_period = 100
    acc_loss = torch.zeros(1)

    for i, data in enumerate(trainloader):
        #print("train counter: ", i)

        inputs, answers = data
        true_pose, true_match = answers
        inputs = inputs.to(device)
        true_pose = true_pose.to(device)
        true_match = true_match.to(device)

        optimizer.zero_grad()

        my_pose, my_match = net(inputs)

        loss = joint_loss(my_pose, my_match, true_pose, true_match)
        loss.backward()

        optimizer.step()
        scheduler.step()

        acc_loss += loss
        if (i+1) % output_period == 0:
            print('step:', i)
            print('acc_loss:', acc_loss.tolist()[0] / output_period)
            acc_loss = torch.zeros(1)

        if (i+1) % 10000 == 0:
            torch.save(net.state_dict(), 'repr0-%d.pkl' % i)
            print('saved repr0-%d.pkl' % i)
