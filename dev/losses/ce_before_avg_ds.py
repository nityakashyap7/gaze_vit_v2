import torch.nn.functional as F
from einops import rearrange, repeat

class CEBeforeAvgDS: 
    '''cross entropy happens for each head separately, then u take the average. this is for the downsampling scheme'''
    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        '''
        :param gaze_preds: non-softmaxed attn logits, shape: (b h p) 
        :param gaze_targs: shape: (b, p)
        where b = batch_size, h = number of heads, p = number of patches

        output: both should be shape: (b h p)
        '''
        
        # hook returns logits but cross entropy wants that (it internally does its own softmax)

        b, h, p = gaze_preds.shape # (b, h, p) where b = batch_size, h = number of heads, p = number of patches
        
        gaze_targs = repeat(gaze_targs, 'b p -> b h p', h=h)

        #  collapse the batch_size and num_heads dim into one that cross_entropy will average over to return u a scalar (this averages across the batch and heads so u dont need to do 2 separate averages)
        gaze_preds = rearrange(gaze_preds, 'b h p -> (b h) p')
        gaze_targs = rearrange(gaze_targs, 'b h p -> (b h) p')

        return gaze_preds, gaze_targs


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        reg = self.reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

        return ce + reg