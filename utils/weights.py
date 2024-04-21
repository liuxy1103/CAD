import numpy as np

def weight_binary_ratio(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 20*alpha
    if label.max() == label.min(): # uniform weights for single-label volume
        weight_factor = 1.0
        weight = np.ones_like(label, np.float32)
    else:
        label = (label!=0).astype(int)
        if mask is None:
            weight_factor = float(label.sum()) / np.prod(label.shape)
        else:
            weight_factor = float((label*mask).sum()) / mask.sum()
        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor > 0.5:
            weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
        else:
            weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

        if mask is not None:
            weight = weight*mask

    return weight.astype(np.float32)
