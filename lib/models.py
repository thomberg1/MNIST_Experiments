from torch import nn
import torch

###################################################################################################################

# http://yann.lecun.com/exdb/lenet/index.html

class LeNet(nn.Module):
    def __init__(self, D_in=(1,28,28), H=84, D_out=10, dropout=0.5, initialize=None, preload=None):
        super(LeNet, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.conv1 = nn.Sequential(
            nn.Conv2d(D_in[0], 6, (5,5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, (5,5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
    
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16*5*5, 120, bias=False),
            nn.BatchNorm1d(120, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(120, self.H, bias=False),
            nn.BatchNorm1d(self.H, affine=True),
            nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if not self.initialize is None:
            self.initialize(self)

        if self.preload is not None:
            self.load_state_dict(torch.load(self.preload))


    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, 16 * 5 * 5)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

###################################################################################################################

# http://yann.lecun.com/exdb/lenet/index.html
# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html

class LeNet_5(nn.Module):
    def __init__(self, D_in=(1,28,28), H=120, D_out=10, dropout=0.5, initialize=None, preload=None):
        super(LeNet_5, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.conv1 = nn.Sequential(
            nn.Conv2d(D_in[0], 6, (5,5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, (5,5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16*5*5, H, bias=False),
            nn.BatchNorm1d(H, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if not self.initialize is None:
            self.initialize(self)

        if self.preload is not None:
            self.load_state_dict(torch.load(self.preload))

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

###################################################################################################################

# http://yann.lecun.com/exdb/lenet/index.html

class LeNet_5_3(nn.Module):
    def __init__(self, D_in=(1,28,28), H=256, D_out=10, dropout=0.5, initialize=None, preload=None):
        super(LeNet_5_3, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.conv1 = nn.Sequential(
            nn.Conv2d(D_in[0], 32, (3,3), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  32, (3,3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64*5*5, H, bias=False),
            nn.BatchNorm1d(H, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if not self.initialize is None:
            self.initialize(self)

        if self.preload is not None:
            self.load_state_dict(torch.load(self.preload))

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])
        
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, 64 * 5 * 5)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

class LeNet_5_3_argumented(nn.Module):
    def __init__(self, D_in=(1,28,28), H=256, D_out=10, dropout=0.5, initialize=None, preload=None):
        super(LeNet_5_3_argumented, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.cnn = LeNet_5_3(D_in=D_in, H=H, D_out=D_out, dropout=dropout, initialize=initialize, preload=self.preload)

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        x = self.cnn(x)

        return x


###################################################################################################################

class LeNet_5_AVG(nn.Module):
    def __init__(self, D_in=(1,28,28), H=256, D_out=10, dropout=0.5, initialize=None, preload=None):
        super(LeNet_5_AVG, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.conv1 = nn.Sequential(
            nn.Conv2d(D_in[0], 32, kernel_size=(5,5), bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(5,5), bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, H, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(H, affine=True),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if not self.initialize is None:
            self.initialize(self)

        if self.preload is not None:
            self.load_state_dict(torch.load(self.preload))

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avgpool(x)

        x = x.view(-1, self.H * 1 * 1)
        x = self.fc1(x)

        return x

###################################################################################################################

from torch import nn
from .classifier import GridGenerator,GridSampler

class SpatialTransformer_bilinear_interpolation(nn.Module):
    def __init__(self, D_in=(1,28,28), initialize=None):
        super(SpatialTransformer_bilinear_interpolation, self).__init__()
        self.D_in = D_in
        self.H = 128
        self.initialize = initialize

        self.grid_generator = GridGenerator(D_in[1], D_in[2])
        self.sampler = GridSampler()

        self.conv_loc_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), bias=False),
            nn.BatchNorm2d(8, affine=True),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(10, affine=True),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc_loc_net = nn.Sequential(
            nn.Conv2d(10, self.H, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(self.H, affine=True),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.reg_loc_net = nn.Sequential(
            nn.Linear(self.H, 3 * 2)
        )

        if not self.initialize is None:
            self.initialize(self)

        # Initialize the weights/bias with identity transformation
        self.reg_loc_net[0].weight.data.fill_(0)
        self.reg_loc_net[0].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        xs = self.conv_loc_net(x)

        xs = self.fc_loc_net(xs)

        xs = self.avgpool(xs)

        xs = xs.view(-1, self.H * 1 * 1)

        theta = self.reg_loc_net(xs)

        theta = theta.view(-1, 2, 3)

        grid_out = self.grid_generator(theta)

        x = x.transpose(1,2).transpose(2,3)
        x = self.sampler(x, grid_out)
        x = x.transpose(3,2).transpose(2,1)

        return x

###################################################################################################################

class MNIST_STN(nn.Module):
    def __init__(self, D_in=(1,28,28), H=256, D_out=10, dropout=0.25, initialize=None, preload=None):
        super(MNIST_STN, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.cnn = LeNet_5_3(D_in=D_in, H=H, D_out=D_out, dropout=dropout, initialize=initialize, preload=self.preload)

        self.st = SpatialTransformer_bilinear_interpolation(D_in=D_in, initialize=initialize)

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        x = self.st(x)

        x = self.cnn(x)

        return x

###################################################################################################################

from .classifier.tps import grid_sample, TPSGridGen
import numpy as np
import itertools

class SpatialTransformer_thin_plate_spline(nn.Module):
    def __init__(self, D_in=(1,28,28), initialize=None):
        super(SpatialTransformer_thin_plate_spline, self).__init__()
        self.D_in = D_in
        self.H = 256
        self.initialize = initialize

        r1 = 0.9
        r2 = 0.9
        self.grid_height = 3
        self.grid_width = 3
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (self.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (self.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)
        target_control_points = torch.cat([X, Y], dim = 1)

        self.grid_generator = TPSGridGen(D_in[1], D_in[2], target_control_points)

        self.conv_loc_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), bias=False),
            nn.BatchNorm2d(8, affine=True),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 10, kernel_size=(5, 5), bias=False),
            nn.BatchNorm2d(10, affine=True),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc_loc_net = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv2d(10, self.H, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(self.H, affine=True),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.reg_loc_net = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(self.H, self.grid_height * self.grid_width * 2)
        )

        if not self.initialize is None:
            self.initialize(self)

        # Initialize the weights/bias with identity transformation
        self.reg_loc_net[0].weight.data.fill_(0)
        self.reg_loc_net[0].bias.data.copy_(target_control_points.view(-1))

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])
        xs = self.conv_loc_net(x)
        xs = self.fc_loc_net(xs)
        xs = self.avgpool(xs)
        xs = xs.view(-1, self.H * 1 * 1)
        theta = self.reg_loc_net(xs)
        theta = theta.view(x.size(0), -1, 2)

        grid_out = self.grid_generator(theta)
        grid_out = grid_out.view(x.size(0), self.D_in[1], self.D_in[2], 2)

        x = grid_sample(x, grid_out)
        return x

###################################################################################################################

class MNIST_TPN(nn.Module):
    def __init__(self, D_in=(1,28,28), H=256, D_out=10, dropout=0.25, initialize=None, preload=None):
        super(MNIST_TPN, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.droput = dropout
        self.initialize = initialize
        self.preload = preload

        self.cnn = LeNet_5_3(D_in=D_in, H=H, D_out=D_out, dropout=dropout, initialize=initialize, preload=self.preload)

        self.st = SpatialTransformer_thin_plate_spline(D_in=D_in, initialize=initialize)

    def forward(self, x):
        x = x.view(-1, self.D_in[0], self.D_in[1], self.D_in[2])

        x = self.st(x)

        x = self.cnn(x)

        return x

###################################################################################################################