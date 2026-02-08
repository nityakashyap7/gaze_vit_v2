from einops import reduce, rearrange
import torch.nn.functional as F


def patch_gaze_masks(gaze_masks, patch_size=(14,14)):
    '''
    returns a (N x f x h_out*w_out) tensor of "attention" values mean-pooled and softmax-ed 
    input: N x f x h x w
    h = h_out * patch_height
    w = w_out * patch_width
    '''
    patch_height = patch_size[0]
    patch_width = patch_size[1]

    gaze_masks = reduce(gaze_masks,
        "N f (h_out ph) (w_out pw) -> N f (h_out w_out)", # note the () around h_out w_out flattens the 2d grid into 1d. why: u can only softmax over 1 dim, so a 2d grid wont work
        "mean", 
        ph=patch_height, 
        pw=patch_width)
    
    gaze_masks = F.softmax(gaze_masks, dim=-1)
    
    return gaze_masks