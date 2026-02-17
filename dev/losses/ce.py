import torch.nn.functional as F


class CE:
    def __init__(self, **kwargs): # needed for signature uniformity among the loss variants
        return
    
    def __call__(self, action_preds, action_targs, **kwargs):
        return F.cross_entropy(action_preds, action_targs)