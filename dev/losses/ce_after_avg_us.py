import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from math import sqrt

class CEAfterAvgUS:
    ''' upsamples the cls token's attention softmax values, mean pool across heads, then calculate cross entropy.
    Note: the code as it's written now assumes square patches (it infers patch size from the shapes of gaze_preds and gaze_targs). 
    if you want nonsquare patches you have refactor this code to accept the patch size tuple as an arg.'''

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        '''
        :param gaze_preds: non-softmaxed attn logits, shape: (b h p) 
        :param gaze_targs: shape: (b, H, W)
        where b = batch_size, h = number of heads, p = number of patches, H, W = og input image height and width 

        output both in shape: (b (H W))
        '''

        b, h, p = gaze_preds.shape
        b, H, W = gaze_targs.shape
        patch_size_squared = (sqrt(N)/l)^2
        

        gaze_preds = rearrange(gaze_preds, 'b h l l -> b h (l l)') # (b, h, p)
        # hook returns logits but cross entropy wants that (it internally does its own softmax)

        # mean pool across the num heads dimension
        gaze_preds = reduce(gaze_preds, 'b h p -> b p', reduction='mean')


        # upsample
        gaze_preds = repeat(gaze_preds, 'b p -> b (p patch_square)', patch_square=patch_size_squared) # p * patch_size_squared = N

        return gaze_preds, gaze_targs


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        reg = self.reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

        return ce + reg