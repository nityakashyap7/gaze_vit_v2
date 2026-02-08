import torch
from utils.hook import AttentionExtractor
from GABRIL_utils.utils import load_dataset
from utils.patch_gaze_masks import patch_gaze_masks
import torch.nn.functional as F
from einops import rearrange

class Trainer:
    def __init__(self, config):
        pass

    def _ce_only(self, action_preds, action_targs, **kwargs):
        return F.cross_entropy(action_preds, action_targs)

    def _ce_plus_gaze_reg(self, action_preds, action_targs, gaze_preds, gaze_targs, reg_lambda=1.0, **kwargs):
            ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
            ce = self._ce_only(action_preds, action_targs)

            #  collapse the batch_size and num_heads dim into one that cross_entropy will average over to return u a scalar (this averages across the batch and heads so u dont need to do 2 separate averages)
            gaze_preds = rearrange(gaze_preds, 'b h n -> (b h) n')
            gaze_targs = rearrange(gaze_targs, 'b h n -> (b h) n')
            reg = reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

            return ce + reg


    def _train_step(self):
        pass

    def train(self):
        pass