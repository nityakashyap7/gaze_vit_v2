import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt

class CEBeforeAvgUS:
    ''' upsamples the cls token's attention softmax values. take mean across heads afterwards. 
    Note: the code as it's written now assumes square patches (it infers patch size from the shapes of gaze_preds and gaze_targs). 
    if you want nonsquare patches you have refactor this code to accept the patch size tuple as an arg.'''

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        '''
        :param gaze_preds: non-softmaxed attn logits, shape: (b h p) 
        :param gaze_targs: shape: (b, H, W)
        where b = batch_size, h = number of heads, p = number of patches, H, W = og input image height and width 

        output both in shape: ((b h) (H W))
        '''
        _, h, p = gaze_preds.shape
        _, H, W = gaze_targs.shape # H = W = 84
        patch_size_squared = int(H*W/p)
        

        # broadcast across heads
        gaze_targs = repeat(gaze_targs, 'b H W -> b h H W', h=h) 

        # hook returns logits but cross entropy wants that (it internally does its own softmax)

        # upsample
        gaze_preds = repeat(gaze_preds, 'b h p -> b h (p patch_square)', patch_square=patch_size_squared) # p * patch_size_squared = H*W (alias as "N")


        #  collapse the batch_size and num_heads dim into one that cross_entropy will average over to return u a scalar (this averages across the batch and heads so u dont need to do 2 separate averages)
        gaze_preds = rearrange(gaze_preds, 'b h N -> (b h) N') 
        gaze_targs = rearrange(gaze_targs, 'b h H W -> (b h) (H W)')

        return gaze_preds, gaze_targs


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        reg = self.reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

        return ce + reg