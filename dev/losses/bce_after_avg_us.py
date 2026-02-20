import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from math import sqrt
from dev.losses.ce_after_avg_us import CEAfterAvgUS

class BCEAfterAvgUS(CEAfterAvgUS):

    def __init__(self, reg_lambda=1.0):
        super().__init__(reg_lambda=reg_lambda)

    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        
        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        ce = F.cross_entropy(action_preds, action_targs)
        reg = self.reg_lambda * F.binary_cross_entropy_with_logits(gaze_preds, gaze_targs)

        return ce + reg