# coding: utf-8

import math
import numpy as np

import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable

#####################################################################################################################

import collections
import types
import dill
import inspect

class HYPERPARAMETERS(collections.OrderedDict):
    """
    Class to make it easier to access hyper parameters by either dictionary or attribute syntax.
    """
    def __init__(self, dictionary={}):
        super(HYPERPARAMETERS, self).__init__(dictionary)
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return self
    def __setstate__(self, d):
        self = d
    @staticmethod
    def load(path):
        h = None
        with open(path, 'rb') as in_strm:
            h = dill.load(in_strm)
        return h
    @staticmethod
    def dump(h, path):
        with open(path, 'wb') as out_strm:
            dill.dump(h, out_strm)
    def __repr__(self):
        fmt_str = '{' + '\n'
        for k, v in self.items():
            if '__class__' in k:
                continue
            if isinstance(v, types.LambdaType):        # function or lambda
                if v.__name__ in '<lambda>':
                    try:
                        fmt_str +=  inspect.getsource(v)
                    except:
                        fmt_str += "    " + "'{}'".format(k).ljust(32) +  ": '" + str(v) + "' ,\n"
                else:
                    fmt_str += "    " + "'{}'".format(k).ljust(32) +  ': ' + v.__name__ + ' ,\n'
            elif isinstance(v, type):                  # class
                fmt_str += "    " + "'{}'".format(k).ljust(32) +  ': ' + v.__name__ + ' ,\n'
            else:                                      # everything else
                if isinstance(v, str):
                    fmt_str += "    " + "'{}'".format(k).ljust(32) +  ": '" + str(v) + "' ,\n"
                else:
                    fmt_str += "    " + "'{}'".format(k).ljust(32) +  ': ' + str(v) + ' ,\n'
        fmt_str += '}\n'
        return fmt_str

#####################################################################################################################

class Metric(object):
    """
    Class to track runtime statistics easier. Inspired by History Variables that not only store the current value,
    but also the values previously assigned. (see https://rosettacode.org/wiki/History_variables)
    """
    def __init__(self, metrics):
        self.metrics = [m[0] for m in metrics]
        self.init_vals = { m[0] : m[1] for m in metrics}
        self.values = {}
        for name in self.metrics:
            self.values[name] = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name in self.metrics:
            self.values[name].append(value)

    def __getattr__(self, attr):
        if attr in self.metrics and not len(self.values[attr]):
            val = self.init_vals[attr]
        else:
            val = self.__dict__[attr]
        return val

    def values(self, metric):
        return self.values[metric]

    def state_dict(self):
        state = {}
        for m in self.metrics:
            state[m] = self.values[m]
        return state

    def load_state_dict(self, state_dict):
        for m in state_dict:
            self.values[m] = state_dict[m]

#####################################################################################################################

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

#####################################################################################################################

def mini_batch(X, Y, n, shuffle=False):
    X_iterable, Y_iterable = X, Y
    l = len(X)
    for ndx in range(0, l, n):
        if shuffle:
            idx_data = np.random.permutation(min(ndx + n, l) - ndx)
            yield X_iterable[ndx:min(ndx + n, l)][idx_data], Y_iterable[ndx:min(ndx + n, l)][idx_data]
        else:
            yield X_iterable[ndx:min(ndx + n, l)], Y_iterable[ndx:min(ndx + n, l)]

#####################################################################################################################

def exp_decay (epoch):
    return  0.5  * 1 / (1  + epoch * 0.5 )

def exp_decay1(epoch):
    lr = 0.2
    lr_g = 0.05
    lr_b = 0.75
    return  lr * (1  + epoch * lr_g )**-lr_b

def step_decay(epoch):
    lr = 0.2
    lr_drop = 0.78         # factor / percent to drop
    lr_epochs_drop = 16.0  # no of epochs after which to drop lr
    lr_min = 0.01
    return max( lr * math.pow(lr_drop, math.floor((1 + epoch) / lr_epochs_drop)), lr_min)


def step_decay1(epoch): return max( math.pow(0.78, math.floor((1 + epoch) / 5.5)), 0.01)
def step_decay2(epoch): return max( math.pow(0.78, math.floor((1 + epoch) / 12.0)), 0.01)
def step_decay3(epoch): return max( math.pow(0.78, math.floor((1 + epoch) / 18.0)), 0.01)
#####################################################################################################################

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, async=False):
    if torch.cuda.is_available():
        x = x.cuda(async=async)
    return Variable(x)

#######################################################################################################################
# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

def torch_weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)

#######################################################################################################################
# https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

import signal

class DelayedKeyboardInterrupt(object):
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

#######################################################################################################################