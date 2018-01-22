import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init


class TopDownLSTMCell(nn.Module):

    """A LSTM cell with top-down connections."""

    def __init__(self, input_size, hidden_size,
                 use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(TopDownLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        # bottom to hidden 
        self.weight_bh = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        # hidden to hidden
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        # top to hidden 
        self.weight_th = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize with standard init from PyTorch source 
        or from recurrent batchnorm paper https://arxiv.org/abs/1603.09025.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.use_bias:
            init.constant(self.bias.data, val=0)


    def forward(self, input_bottom, input_top, hx):
        """
        Args:
            input_bottom: A (batch, input_size) tensor containing input
                features from the lower layer at the current timestep.
            input_top: A (batch, input_size) tensor containing input
                features from the higher layer from the previous timestep. 
            hx: A tuple (h_0, c_0), 
                which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wb = torch.mm(input_bottom, self.weight_bh)
        wt = torch.mm(input_top, self.weight_th)
        f, i, o, g = torch.split(wh_b + wb + wt,
                                 split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class TodDownLSTM(nn.Module):

    """A module that runs multiple steps of TopDownLSTM."""

    def __init__(self, input_size, hidden_size, num_layers=3,
                 use_bias=True, batch_first=False, batch_first_out=False,
                 dropout=0, **kwargs):
        super(MultiLevelLSTM, self).__init__()
        print("Running custom MultilevelLSTM")
        assert num_layers >= 2, "Number of layers must be >= 2."
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.batch_first_out = batch_first_out
        self.dropout = dropout
        for layer in range(num_layers-1):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = TopDownLSTMCell(input_size=layer_input_size,
                              hidden_size=hidden_size, 
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        # Top layer is vanilla LSTM layer
        cell = LSTMCell(input_size=hidden_size,
                        hidden_size=hidden_size, 
                        **kwargs)
        setattr(self, 'cell_{}'.format(num_layers-1), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()


    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
        if input_.is_cuda:
            device = input_.get_device()
            length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        
        # Array to hold data h_t for all layers
        Ht = Variable(input_.data.new(batch_size, self.num_layers, self.hidden_size).zero_())
        # Array to hold data c_t for all layers
        C = Variable(input_.data.new(batch_size, self.num_layers, self.hidden_size).zero_())
        # Array to hold data h_t for all layers
        h_n = []
        c_n = []
        layer_output = None
        max_time = input_.size(0)
        output = []
        max_time = input_.size(0)
        output = []
        for t in range(max_time):
            for l in range(self.num_layers):
                # Run lowest layer, special because it takes the word-embeddings
                cell = self.get_cell(l)
                hx = (Ht[:, l], C[:, l])       # previous step from layer l
                if l == 0:
                    top = Ht[:, 1]
                    bottom = input_[t]
                    h_next, c_next = cell(input_bottom=bottom, 
                                          input_top=top,
                                          hx=(Ht[:, 0], C[:, 0]))
                    mask = (t < length).float().unsqueeze(1).expand_as(h_next)
                    h_next = h_next*mask + hx[0]*(1 - mask)
                    c_next = c_next*mask + hx[1]*(1 - mask)
                # Replace with current state
                elif l > 0 and l < self.num_layers - 1:
                    top = Ht[:, l + 1]      # previous step from higher layer
                    bottom = Ht[:, l -1]    # Current step from lower layer 
                    h_next, c_next = cell(input_bottom=bottom, 
                                          input_top=top,
                                          hx=hx)
                    mask = (t < length).float().unsqueeze(1).expand_as(h_next)
                    h_next = h_next*mask + hx[0]*(1 - mask)
                    c_next = c_next*mask + hx[1]*(1 - mask)
                # Run top layer vanilla LSTM
                else:
                    inp = Ht[:, l - 1]   # input is just the activation of the penultimate layer
                    h_next, c_next = cell(input_=inp, hx=hx)
                    mask = (t < length).float().unsqueeze(1).expand_as(h_next)
                    h_next = h_next*mask + hx[0]*(1 - mask)
                    c_next = c_next*mask + hx[1]*(1 - mask)
                    output.append(h_next)
                # Update H and C arrays, but not in-place, because of autograd 
                C_ = C.clone()
                Ht_ = Ht.clone()
                C_[:, l] = C[:, l] * 0 + c_next
                Ht_[:, l] = Ht[:, l] * 0 + h_next
                C = C_
                Ht = Ht_
        output = torch.stack(output, 0)
        if self.batch_first_out:
            output = output.transpose(1, 0)
        # FIXME return interediate hidden states
        return output, (None, None)

