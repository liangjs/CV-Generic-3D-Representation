from __future__ import print_function
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import random

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
        if t.sum() >= 1:
            ans[t] = 1 + torch.log(input[t])
        return ans

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        ans = grad_output.clone()
        t = input > 1
        if t.sum() >= 1:
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


def validate(net, dataset, num):
    global device
    net.train(False)

    sel = list(range(len(dataset)))
    random.shuffle(sel)
    sel = sel[:num]
    counter = 0
    counter_pose = 0

    acc_pose = 0
    acc_match = 0
    pose_std_ = pose_std.to(device)
    pose_mean_ = pose_mean.to(device)

    for idx in sel:
        data = dataset[idx]
        inputs, answers = data
        true_pose, true_match = answers
        if true_match == -1:
            continue

        inputs = inputs.to(device)
        true_pose = true_pose.to(device)

        counter += 1

        my_pose, my_match = net(torch.stack((inputs,)))
        if true_match == 1:
            counter_pose += 1
            my_pose = my_pose[0] * pose_std_ + pose_mean_
            my_pose = my_pose[1:]
            true_pose = true_pose[1:]
            # three cosines theorem
            cs = torch.cos((my_pose - true_pose) / 180 * math.pi)
            acc_pose += float(torch.abs(torch.acos(cs[0] * cs[1])) * 180 / math.pi)
        my_match = my_match[0]
        if my_match[true_match] > my_match[1 ^ true_match]:
            acc_match += 1.0

        if counter >= num:
            break

    net.train(True)

    if counter_pose > 0:
        acc_pose /= counter_pose
    if counter > 0:
        acc_match /= counter
    return (acc_pose, acc_match)


if __name__ == "__main__":

    net = LearnNet()
    net.load_state_dict(torch.load('repr0-40000.pkl'))
    print('load parameters')
    sys.stdout.flush()

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("device:", device)
    sys.stdout.flush()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60000, gamma=0.1)

    testdataset = TestDataset()

    print('acc_pose:%.2f, acc_match:%.2f' % validate(net, testdataset, 1000))
    sys.stdout.flush()

    counter = 0

    for epoch in range(100):
        print('epoch:', epoch)

        trainloader = TrainLoader(shuffleTar=True, keepTar=False)

        output_period = 100
        acc_loss = torch.zeros(1).to(device)
        valid_period = 1000
        save_period = 10000

        for data in trainloader:
            counter += 1
            #print("train counter:", counter)

            inputs, answers = data
            true_pose, true_match = answers
            valid = true_match >= 0
            inputs = inputs[valid].to(device)
            true_pose = true_pose[valid].to(device)
            true_match = true_match[valid].to(device)

            optimizer.zero_grad()

            my_pose, my_match = net(inputs)

            loss = joint_loss(my_pose, my_match, true_pose, true_match)
            loss.backward()

            optimizer.step()
            scheduler.step()

            acc_loss += loss
            if counter % output_period == 0:
                print('step:%d, acc_loss:%.2f' % (counter, acc_loss / output_period))
                acc_loss = torch.zeros(1).to(device)

            if counter % valid_period == 0:
                acc_pose, acc_match = validate(net, testdataset, 1000)
                print('acc_pose:%.2f, acc_match:%.2f' % (acc_pose, acc_match))

            if counter % save_period == 0:
                torch.save(net.state_dict(), 'repr0-%d.pkl' % counter)
                print('saved repr0-%d.pkl' % counter)

            sys.stdout.flush()
