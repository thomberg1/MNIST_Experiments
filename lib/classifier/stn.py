import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np

from ._ext import my_lib

from cffi import FFI
ffi = FFI()

class GridGenerator(Module):
    def __init__(self, height, width):
        super(GridGenerator, self).__init__()
        self.height, self.width = height, width

        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))


    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()

        output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

class GridSamplerFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")

        output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            output = output.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())

        if not grad_output.is_cuda:
            my_lib.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2

class GridSampler(Module):
    def __init__(self):
        super(GridSampler, self).__init__()
        self.f = GridSamplerFunction()

    def forward(self, input1, input2):
        return self.f(input1, input2)