import torch.nn.functional as F
import torch

def mask_time(t, length, new, past):
    """
    Returns new state if t < length else copies old state over.
    new: iterable containing new states [h, c, z, ..]
    old: iterable containing past states [h_tm1, c_tm1, z_tm1, ..]
    """   
    for n, p in zip(new, past):
        mask = (t < length).float()
        if len(n.size()) > 1:
            mask = mask.unsqueeze(1).expand_as(n)
        yield n * mask + p * (1 - mask)

def hard_sigmoid(x, slope=0.2):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (slope * x) + 0.5
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x


def st_hard_sigmoid(x, slope=0.2):
    """
    Straight-through estimator with hard_sigmoid and thresholding.
    """
    #TODO implement sampling variant (from Bernoulli)
    x = hard_sigmoid(x, slope)
    x_hard = torch.round(x)
    x = (x_hard - x).detach() + x
    return x


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))

def copy_op(h_tm1, c_tm1, h_next, c_next, z_lm1=None, z_tm1=None):
    """COPY op for MultiscaleLSTM. Keep or replace old state."""
    # write = torch.min(ones, z_tm1 + z_lm1).unsqueeze(1)
    # Highest layer doesnt have z_tm1
    if z_tm1 is not None:
        write = z_tm1 + z_lm1 == 0
    else:
        write = z_lm1 == 0
    write = write.unsqueeze(1).float()
    c_next = (1 - write) * c_tm1 + write * c_next
    h_next = (1 - write) * h_tm1 + write * h_next
    return h_next, c_next 
