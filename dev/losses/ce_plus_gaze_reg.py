import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from math import sqrt

class CEPlusGazeReg: 
    def __init__(self, reg_loss_fn, reg_lambda=1.0):
        self.reg_loss_fn = reg_loss_fn
        self.reg_lambda = reg_lambda
        


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        return gaze_preds, gaze_targs

    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)
        reg = self.reg_lambda * self.reg_loss_fn(gaze_preds, gaze_targs)

        return ce + reg
    

class CEBeforeAvgUS(CEPlusGazeReg):
    ''' upsamples the cls token's attention softmax values. take mean across heads afterwards. 
    Note: the code as it's written now assumes square patches (it infers patch size from the shapes of gaze_preds and gaze_targs). 
    if you want nonsquare patches you have refactor this code to accept the patch size tuple as an arg.'''


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        '''
        :param gaze_preds: non-softmaxed attn logits, shape: (b h p) 
        :param gaze_targs: shape: (b, H, W)
        where b = batch_size, h = number of heads, p = number of patches, H, W = og input image height and width 

        output both in shape: ((b h) (H W))
        '''
        b, h, p = gaze_preds.shape
        b, H, W = gaze_targs.shape # H = W = 84
        

        # broadcast across heads
        gaze_targs = repeat(gaze_targs, 'b H W -> b h H W', h=h) 

        # hook returns logits but cross entropy wants that (it internally does its own softmax)

        
        # collapse the batch_size and num_heads dim into one that cross_entropy will average over to return u a scalar (this averages across the batch and heads so u dont need to do 2 separate averages)
        gaze_preds = rearrange(gaze_preds, 'b h p -> (b h) p') # do this before upsampling bc interpolate is strict abt dims (needs exactly 3 for mode = bilinear)
        gaze_targs = rearrange(gaze_targs, 'b h H W -> (b h) (H W)') # lets do that for targs here as well so i dont forget


        # upsample
        p1 = p2 = int(sqrt(p))
        gaze_preds = rearrange(gaze_preds, 'bh (p1 p2) -> bh 1 p1 p2', p1=p1, p2=p2) # interpolate expects a channel dim and a 2D grid
        gaze_preds = F.interpolate(gaze_preds, size=(H, W), align_corners=False, mode='bilinear')

        gaze_targs = rearrange(gaze_targs, 'bh, H, W -> bh (H W)')

        return gaze_preds, gaze_targs
    

class CEAfterAvgUS(CEPlusGazeReg):
    ''' upsamples the cls token's attention softmax values, mean pool across heads, then calculate cross entropy.
    Note: the code as it's written now assumes square patches (it infers patch size from the shapes of gaze_preds and gaze_targs). 
    if you want nonsquare patches you have refactor this code to accept the patch size tuple as an arg.'''


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        '''
        :param gaze_preds: non-softmaxed attn logits, shape: (b h p) 
        :param gaze_targs: shape: (b, H, W)
        where b = batch_size, h = number of heads, p = number of patches, H, W = og input image height and width 

        output both in shape: (b (H W))
        '''

        b, h, p = gaze_preds.shape
        b, H, W = gaze_targs.shape
        
        
        # hook returns logits but cross entropy wants that (it internally does its own softmax)

        # mean pool across the num heads dimension
        gaze_preds = reduce(gaze_preds, 'b h p -> b p', reduction='mean')

        # upsample
        p1 = p2 = int(sqrt(p))
        gaze_preds = rearrange(gaze_preds, 'b (p1 p2) -> b 1 p1 p2', p1=p1, p2=p2) # interpolate expects a channel dim and a 2D grid
        gaze_preds = F.interpolate(gaze_preds, size=(H, W), align_corners=False, mode='bilinear')

        gaze_targs = rearrange(gaze_targs, 'b, H, W -> b (H W)')

        return gaze_preds, gaze_targs
    



