"""
Pytorch implementation of the basic LSTM cell, code taken from
From: https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import math
import numpy as np


class SimpleRANCell(nn.Module):

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Implementation of the Recurrent Additive Network cell https://arxiv.org/pdf/1705.07393.pdf.
        This cell implements the version described in equations 2. on page 2.
        """

        super(SimpleRANCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 2 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 2 * hidden_size))
        if use_bias:
            self.bias_h = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize with standard init from PyTorch source 
        or from recurrent batchnorm paper https://arxiv.org/abs/1603.09025.
        """
        print("init RANCell")
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) 
        if self.use_bias:
            init.constant(self.bias_h.data, val=0)


    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: Initial hidden cell state, with size (batch, hidden_size).

        Returns:
            c_1: Tensors containing the next cell state.
        """
        c_0 = hx
        batch_size = c_0.size(0)
        bias_batch = (self.bias_h.unsqueeze(0)
                        .expand(batch_size, *self.bias_h.size()))
        
        wh_c = torch.addmm(bias_batch, c_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i = torch.split(wh_c + wi,
                           split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(i) * input_  + torch.sigmoid(f) * torch.tanh(c_0)
        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SimpleRAN(nn.Module):

    """A module that runs multiple steps of RAN."""

    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False,
                 dropout=0, **kwargs):
        super(SimpleRAN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.batch_first_out = self.batch_first
        self.dropout = dropout
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = SimpleRANCell(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_,  hx, length):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(c_next)
            c_next = c_next*mask + hx * (1 - mask)
            output.append(c_next)
            hx = c_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None, length=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
        if input_.is_cuda:
            device = input_.get_device()
            length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(num_layers, batch_size, self.hidden_size).zero_())
        #TODO this is because the init_hidden in lm.py gives a tuple for the LSTM
        elif isinstance(hx, tuple):
            hx = hx[0]
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            c0 = hx[layer]
            cell = self.get_cell(layer)
            layer_output, layer_c_n = SimpleRAN._forward_rnn(
                cell=cell, input_=input_,  hx=c0, length=length)
            # Don't apply dropout to last layer (PyTorch default)
            if layer < self.num_layers -1:
                input_ = self.dropout_layer(layer_output)
            else:
                input_ = layer_output
            c_n.append(layer_c_n)
        output = layer_output
        c_n = torch.stack(c_n, 0)
        if self.batch_first_out:
            output = output.transpose(1, 0)
        return output, c_n

