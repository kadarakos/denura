"""
Implements Hierachical Multiscale LSTM from 
Hierarchical Multiscale Recurrent Networks https://arxiv.org/pdf/1609.01704.pdf
"""
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from util import mask_time
from lstm import LSTMCell
from util import st_hard_sigmoid, size_splits, copy_op
import numpy as np


class BottomHMLSTMCell(nn.Module):

    """
    A MultiScale LSTM cell.
    https://arxiv.org/pdf/1609.01704.pdf
    """

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Bottom HMLSTMCell does not take boundary variable from lower layer.

        input_size: int, size of the input layer
        hidden_size: int, size of the hidden layer
        bottom: bool, bottom layer in the stack always takes input
                so need to know at init.
        """

        super(BottomHMLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # bottom to hidden 
        self.weight_bh = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size + 1))
        # hidden to hidden
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size + 1))
        # top to hidden 
        self.weight_th = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size + 1))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size + 1))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()


    def reset_parameters(self):
        """
        Initialize with standard init from PyTorch source 
        or from recurrent batchnorm paper https://arxiv.org/abs/1603.09025.
        """
        # for weight in self.parameters():
        #    weight.data.uniform_(-stdv, stdv)
        print('init BottomHMLSTMCell')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_bottom, input_top, hx, z_tm1):
        """
        Args:
            input_bottom: A (batch, input_size) tensor containing input
                features from the lower layer at the current timestep.
            input_top: A (batch, input_size) tensor containing input
                features from the higher layer from the previous timestep. 
            z_tm1: A (batch, ) tensor containing boundary variables 
                from the current layer at the previous timestep. 
            hx: A tuple (h_0, c_0), 
                which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        # Bottom layer always takes input so z_lm1 is always 1 for all samples.
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        # Recurrent
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        # Bottom-up with no mask for lowest layer
        wb = torch.mm(input_bottom, self.weight_bh)
        #Top-down, mask where z = 0
        wt = z_tm1.unsqueeze(1) * torch.mm(input_top, self.weight_th)
        f, i, o, g, z = size_splits(wh_b + wb + wt,
                                    split_sizes=[self.hidden_size, self.hidden_size, 
                                                 self.hidden_size, self.hidden_size, 1],
                                                 dim=1)
        # Using z_tm1 as a mask either runs UPDATE or FLUSH
        c_1 = (1 - z_tm1).unsqueeze(1) * torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        # TODO implement slope annealing trick
        slope = 0.5
        z_mask = st_hard_sigmoid(z, slope).squeeze()
        return h_1, c_1, z_mask

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class TopHMLSTMCell(nn.Module):

    """
    A MultiScale LSTM cell.
    https://arxiv.org/pdf/1609.01704.pdf
    """

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Top HMLSTMCell does not output boundary variable.

        input_size: int, size of the input layer
        hidden_size: int, size of the hidden layer
        bottom: bool, bottom layer in the stack always takes input
                so need to know at init.
        """

        super(TopHMLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # bottom to hidden 
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        # hidden to hidden
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()


    def reset_parameters(self):
        """
        Initialize with standard init from PyTorch source 
        or from recurrent batchnorm paper https://arxiv.org/abs/1603.09025.
        """
        # for weight in self.parameters():
        #    weight.data.uniform_(-stdv, stdv)
        print('init  TopHMLSTMCell')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx, z_lm1):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features from the lower layer at the current timestep.
            z_tm1: A (batch, ) tensor containing boundary variables 
                from the current layer at the previous timestep. 
            hx: A tuple (h_0, c_0), 
                which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        # Bottom layer always takes input so z_lm1 is always 1 for all samples.
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        # Recurrent
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        # Don't worry about the bottom-up mask. 
        # When the mask is on, COPY is ran the current activation is dropped anyways
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 split_size=self.hidden_size, dim=1)
        # Run normal LSTM UPDATE
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        # Run COPY, which in case of the top-most layer is just the 
        c_1 = z_lm1.unsqueeze(1) * c_1 + (1 - z_lm1).unsqueeze(1) * c_0
        h_1 = z_lm1.unsqueeze(1) * h_1 + (1 - z_lm1).unsqueeze(1) * h_0
        # Doesn't return z
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class HMLSTMCell(nn.Module):

    """
    General "middle" HMLSTM cell.
    https://arxiv.org/pdf/1609.01704.pdf
    """

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        

        input_size: int, size of the input layer
        hidden_size: int, size of the hidden layer
        bottom: bool, bottom layer in the stack always takes input
                so need to know at init.
        """

        super(HMLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # bottom to hidden 
        self.weight_bh = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size + 1))
        # hidden to hidden
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size + 1))
        # top to hidden 
        self.weight_th = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size + 1))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size + 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initialize with standard init from PyTorch source 
        or from recurrent batchnorm paper https://arxiv.org/abs/1603.09025.
        """
        # for weight in self.parameters():
        #    weight.data.uniform_(-stdv, stdv)
        print('init HMLSTMCell')
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_bottom, input_top, hx,
                z_tm1=None, z_lm1=None):
        """
        Args:
            input_bottom: A (batch, input_size) tensor containing input
                features from the lower layer at the current timestep.
            input_top: A (batch, input_size) tensor containing input
                features from the higher layer from the previous timestep. 
            z_lm1: A (batch, ) tensor containing boundary variables 
                from the lower layer at the current timestep. 
            z_tm1: A (batch, ) tensor containing boundary variables 
                from the current layer at the previous timestep. 
            hx: A tuple (h_0, c_0), 
                which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        # Bottom layer always takes input so z_lm1 is always 1 for all samples.
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        # Recurrent
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        # Bottom-up, mask where z = 0
        wb = torch.mm(input_bottom, self.weight_bh)
        # Top-down, mask where z = 0
        wt = torch.mm(input_top, self.weight_th)
        # Lowest layer doen"t get z_lm1
        if z_lm1 is not None:
            wb = z_lm1.unsqueeze(1) * wb
        # Highest layer doesn"t get z_tm1
        if z_tm1 is not None:
            wt = z_tm1.unsqueeze(1) * wt
            mask = (1 - z_tm1).unsqueeze(1)
        # New Boundary variable z
        f, i, o, g, z = size_splits(wh_b + wb + wt,
                                    split_sizes=[self.hidden_size, self.hidden_size, 
                                                 self.hidden_size, self.hidden_size, 1],
                                                 dim=1)
        if z_tm1 is not None:
            past = mask * torch.sigmoid(f) * c_0 
        else: 
            past = torch.sigmoid(f) * c_0 
        if z_tm1 is not None:
            write = z_tm1 + z_lm1 == 0
        else:
            write = z_lm1 == 0
        c_1 = past + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        # TODO implement slope annealing trick
        slope = 0.2
        z_mask = st_hard_sigmoid(z, slope).squeeze()
        return h_1, c_1, z_mask

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class HMLSTM(nn.Module):

    """A module that runs multiple steps of a MultiScaleLSTM stack."""

    def __init__(self, input_size, hidden_size, num_layers=3,
                 use_bias=True, batch_first=False, batch_first_out=False,
                 dropout=0, **kwargs):
        super(HMLSTM, self).__init__()
        print("Running HMLSTM")
        assert num_layers >= 2, "Number of layers must be >= 2."
        self.input_size = input_size
        self.hidden_size = hidden_size
        # TODO have output_size not depend on hidden_size
        self.output_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.batch_first_out = self.batch_first
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_matrix = nn.Linear(self.hidden_size, self.output_size)
        self.gate_vector = nn.Linear(self.num_layers * self.hidden_size, self.num_layers)  
        # Parameters of the gated output module
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            # Mark lowest layer as "bottom"
            if layer == 0:
                cell = BottomHMLSTMCell(input_size=layer_input_size,
                                  hidden_size=hidden_size, use_bias=use_bias)
            elif layer == num_layers - 1: 
                cell = TopHMLSTMCell(input_size=layer_input_size,
                                     hidden_size=hidden_size, use_bias=use_bias)
            #TODO This HMLSTMCell is bad
            else:
                cell = HMLSTMCell(input_size=layer_input_size,
                                  hidden_size=hidden_size, **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        # Top layer is vanilla LSTM layer
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def update_state(self, H, h,  C, c, l, Z=None, z=None):
        C_ = C.clone()
        H_ = H.clone()
        C_[l] = C[l] * 0 + c
        H_[l] = H[l] * 0 + h
        if Z is not None:
            Z_ = Z.clone()
            Z_[l] = Z[l] * 0 + z
            return H_, C_, Z_
        else:
            return H_, C_


    def forward(self, input_, hx=None, length=None, pred_boundaries=False, show_z=False):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if pred_boundaries:
            assert batch_size == 1, "Can only predict boundaries on a single example."
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
        if input_.is_cuda:
            device = input_.get_device()
            length = length.cuda(device)
        if hx is None:     
            # Array to hold data h_t for all layers
            Ht = [Variable(input_.data.new(batch_size, self.hidden_size).zero_()) for x in range(self.num_layers)]
            # Array to hold data c_t for all layers
            C = [Variable(input_.data.new(batch_size, self.hidden_size).zero_()) for x in range(self.num_layers)]
            # Array to hold data for boundary variable z_t for all layers
            # Top layer doesn't have boundary detector
            Z = [Variable(input_.data.new(batch_size).zero_()) for x in range(self.num_layers - 1)]
            hx = [Ht, C, Z]
        Ht, C, Z = map(list, hx)
        if pred_boundaries:
            boundaries = np.zeros((self.num_layers - 1, max_time))
            gates = []
        # Vector of ones and zeros
        max_time = input_.size(0)
        output = []
        for t in range(max_time):
            for l in range(self.num_layers):
                cell = self.get_cell(l)
                h_tm1, c_tm1 = Ht[l], C[l]       # previous step from layer l
                hx = (h_tm1, c_tm1)
                if l == 0:
                    top = Ht[1]
                    bottom = input_[t]  # bottom for lowest layer is the embedding
                    z_tm1 = Z[0]
                    h_next, c_next, z_next = cell(input_bottom=bottom, input_top=top, 
                                                  z_tm1=z_tm1, hx=hx)
                    h_next, c_next, z_next = mask_time(t, length, 
                                                       [h_next, c_next, z_next],
                                                       [hx[0], hx[1], z_tm1])
                    # Ht, C, Z = self.update_state(Ht, h_next, C, c_next, l, Z, z_next)
                    # print(Ht)
                    Ht[l], C[l], Z[l] = h_next, c_next, z_next
                #TODO handle general case 
                elif l > 0 and l < self.num_layers - 1:
                    top = Ht[l + 1]      # previous step from higher layer
                    bottom = Ht[l -1]    # Current step from lower layer 
                    z_tm1 = Z[l]
                    z_lm1 = Z[l - 1]
                    h_next, c_next, z_next = cell(input_bottom=bottom, 
                                                  input_top=top, z_tm1=z_tm1, 
                                                  z_lm1=z_lm1, hx=hx)
                    h_next, c_next = copy_op(h_tm1, c_tm1, h_next, c_next, z_lm1, z_tm1)
                    h_next, c_next = mask_time(t, length, h_next, c_next, hx[0], hx[1])
                    Ht, C, Z = self.update_state(Ht, h_next, C, c_next, l, Z, z_next)
                else:
                    bottom = Ht[l - 1]   # input is just the activation of the penultimate layer
                    z_lm1 = Z[l - 1]
                    # print(torch.sum(z_lm1))
                    h_next, c_next = cell(input_=bottom, z_lm1=z_lm1, hx=hx)
                    h_next, c_next = mask_time(t, length, [h_next, c_next], 
                                              [hx[0], hx[1]])
                    Ht[l], C[l] = h_next, c_next
                    #Ht, C = self.update_state(Ht, h_next, C, c_next, l)
                if pred_boundaries and l < self.num_layers - 1:
                    boundaries[l, t] = z_next.data.cpu().numpy()[0]
            # Gated output layer from equations 11 and 12
            Ht_ = torch.stack(Ht)
            Ht_hat = self.output_matrix(Ht_.view(-1, self.hidden_size))
            g = functional.sigmoid(self.gate_vector(Ht_.view(batch_size, -1)))
            gated = g.view(-1).unsqueeze(1) * Ht_hat
            gated = gated.view(self.num_layers, batch_size, -1)
            out = gated.sum(0)
            output.append(out)
            if pred_boundaries:
                gates.append(g.data.cpu().numpy()[0])
            if show_z:
                print(torch.sum(Z))
        output = torch.stack(output, 0)
        if self.batch_first_out:
            output = output.transpose(1, 0)
        if pred_boundaries:
            return boundaries, gates
        else:
            return output, (Ht, C, Z)

