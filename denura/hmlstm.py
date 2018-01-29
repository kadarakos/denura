import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from util import mask_time
from lstm import LSTMCell
from util import st_hard_sigmoid, size_splits, copy_op


class HMLSTMCell(nn.Module):

    """
    A MultiScale LSTM cell.
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
        print("init MultiScaleLSTMCell")
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
                z_tm1, z_lm1, training):
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
        wb = z_lm1.unsqueeze(1) * torch.mm(input_bottom, self.weight_bh)
        # Top-down, mask where z = 0
        wt = z_tm1.unsqueeze(1) * torch.mm(input_top, self.weight_th)
        # New Boundary variable z
        f, i, o, g, z = size_splits(wh_b + wb + wt,
                                    split_sizes=[self.hidden_size, self.hidden_size, 
                                                 self.hidden_size, self.hidden_size, 1],
                                                 dim=1)
        mask = (1 - z_tm1)
        c_1 = mask.unsqueeze(1) * torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        # TODO implement slope annealing trick
        slope = 1.0
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
        print("Running  MultiScaleLSTM")
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
                cell = HMLSTMCell(input_size=layer_input_size,
                                  hidden_size=hidden_size, **kwargs)
            elif layer == num_layers - 1: 
                cell = LSTMCell(input_size=layer_input_size,
                                hidden_size=hidden_size, **kwargs)
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


    def forward(self, input_, training, length=None, hx=None,
                pred_boundaries=False):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if pred_boundaries:
            assert batch_size == 1, "Can only predict boundaries on a single example."
            training = False
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
        if input_.is_cuda:
            device = input_.get_device()
            length = length.cuda(device)
        if hx is None:     
            # Array to hold data h_t for all layers
            Ht = Variable(input_.data.new(self.num_layers, batch_size, self.hidden_size).zero_())
            # Array to hold data c_t for all layers
            C = Variable(input_.data.new(self.num_layers, batch_size, self.hidden_size).zero_())
            # Array to hold data for boundary variable z_t for all layers
            # Top layer doesn't have boundary detector
            Z = Variable(input_.data.new(self.num_layers - 1, batch_size).zero_())
            hx = (Ht, C, Z)
        Ht, C, Z = hx
        if pred_boundaries:
            boundaries = np.zeros((self.num_layers - 1, max_time))
        # Vector of ones and zeros
        zeros = Variable(input_.data.new(batch_size).zero_())
        ones = Variable(input_.data.new(batch_size).zero_() + 1)
        max_time = input_.size(0)
        output = []
        for t in range(max_time):
            for l in range(self.num_layers):
                cell = self.get_cell(l)
                h_tm1, c_tm1 = Ht[l], C[l]       # previous step from layer l
                hx = (h_tm1, c_tm1)
                # Run lowest layer, special because it takes the word-embeddings 
                # and always has z_lm1 boundary variable = 1
                if l == 0:
                    top = Ht[1]
                    bottom = input_[t]
                    z_tm1 = Z[0]
                    z_lm1 = ones
                    h_next, c_next, z_next = cell(input_bottom=bottom, 
                                                  input_top=top, z_tm1=z_tm1, 
                                                  z_lm1=z_lm1, hx=hx, training=training)
                    h_next, c_next = copy_op(h_tm1, c_tm1, z_tm1, z_lm1, h_next, c_next, ones)
                    h_next, c_next = mask_time(t, length, h_next, c_next, hx[0], hx[1])
                # Replace with current state
                elif l > 0 and l < self.num_layers - 1:
                    top = Ht[l + 1]      # previous step from higher layer
                    bottom = Ht[l -1]    # Current step from lower layer 
                    z_tm1 = Z[l]
                    z_lm1 = Z[l - 1]
                    h_next, c_next, z_next = cell(input_bottom=bottom, 
                                                  input_top=top, z_tm1=z_tm1, 
                                                  z_lm1=z_lm1, hx=hx, training=training)
                    h_next, c_next = copy_op(h_tm1, c_tm1, z_tm1, z_lm1, h_next, c_next, ones)
                    h_next, c_next = mask_time(t, length, h_next, c_next, hx[0], hx[1])
                # Run top layer vanilla LSTM
                # The top layer doesn't take boundary variables as input because
                # it either runs UPDATE or COPY and has no top-down connection.
                else:
                    inp = Ht[l - 1]   # input is just the activation of the penultimate layer
                    z_tm1 = zeros  # top layer never detects boundary
                    h_next, c_next = cell(input_=inp, hx=hx)
                    h_next, c_next = copy_op(h_tm1, c_tm1, z_tm1, z_lm1, h_next, c_next, ones)
                    h_next, c_next = mask_time(t, length, h_next, c_next, hx[0], hx[1])
                    # Gated output layer from equations 11 and 12
                    Ht_hat = self.output_matrix(Ht.view(-1, self.hidden_size))
                    g = functional.sigmoid(self.gate_vector(Ht.view(batch_size, -1)))
                    gated = g.view(-1).unsqueeze(1) * Ht_hat
                    gated = gated.view(self.num_layers, batch_size, -1)
                    out = gated.sum(0)
                    output.append(out)
                # Update H and C arrays, but not in-place, because of autograd 
                C_ = C.clone()
                Ht_ = Ht.clone()
                C_[l] = C[l] * 0 + c_next
                Ht_[l] = Ht[l] * 0 + h_next
                C = C_
                Ht = Ht_
                if l < self.num_layers - 1:
                    Z_ = Z.clone()
                    Z_[l] = Z[l] * 0 + z_next
                    Z = Z_
                if pred_boundaries and l < self.num_layers - 1:
                    boundaries[l, t] = z_next.data.cpu().numpy()[0]
        output = torch.stack(output, 0)
        if self.batch_first_out:
            output = output.transpose(1, 0)
        if pred_boundaries:
            return boundaries
        else:
            return output, (Ht, C, Z)

