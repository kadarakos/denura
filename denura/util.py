def mask_time(t, length, h, c, h_, c_):
    """Returns new state if t < length else copies old state over."""
    mask = (t < length).float().unsqueeze(1).expand_as(h)
    h_next = h * mask + h_ * (1 - mask)
    c_next = c * mask + c_ * (1 - mask)
    return h_next, c_next

