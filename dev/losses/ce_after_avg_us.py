import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from math import sqrt
from ce_plus_gaze_reg import CEPlusGazeReg

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
        gaze_preds = rearrange(gaze_preds, 'b p -> b 1 p1 p2', p1=p1, p2=p2) # interpolate expects a channel dim and a 2D grid
        gaze_preds = F.interpolate(gaze_preds, size=(H, W), align_corners=False, mode='bilinear')

        gaze_targs = rearrange(gaze_targs, 'b, H, W -> b (H W)')

        return gaze_preds, gaze_targs