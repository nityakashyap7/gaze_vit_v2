import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_gaze_mask(z_oreo, beta, target_size):
    z_oreo_flat = z_oreo.abs().sum(dim=1)
    # apply softmax on the last two dimensions of the tensor
    z_oreo_shape = z_oreo_flat.shape
    z_oreo_flat = z_oreo_flat.reshape(z_oreo.shape[0], -1)

    z_oreo_softmax = torch.nn.functional.softmax(z_oreo_flat / beta, dim=-1)
    z_oreo_softmax = z_oreo_softmax.reshape(z_oreo_shape)
    
    # return z_oreo_softmax

    # scale up z_oreo to the size of xx
    z_oreo_mask = torch.nn.functional.interpolate(z_oreo_softmax.unsqueeze(1), size=target_size, mode='bicubic')

    # normalize z_oreo_mask
    flattened = z_oreo_mask.view(z_oreo_mask.shape[0], z_oreo_mask.shape[1], -1)
    gaze_mask = z_oreo_mask / flattened.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)

    # masked_xx = xx * gaze_mask
    # plot_once(xx_clean, masked_xx, 2, beta)

    return gaze_mask


def apply_gmd_dropout(z, g, test_mode = False):
    dropout_prob = 0.7
    B, C, H, W = z.shape
    A = torch.rand([B, 1, H, W], device=z.device)
    K = torch.nn.functional.interpolate(g, size=(H, W), mode='bicubic')
    K = (K - K.min()) / (K.max() - K.min())
    K = dropout_prob * K + (1 - dropout_prob)
    M = (A < K).float()
    if test_mode:
        z = z * K
    else:
        z = z * M
    return z
    
    

temp_flag = False
def plot_once(xx_clean, masked_xx, count, beta):
    global temp_flag
    os.makedirs("figs", exist_ok=True)
    if not temp_flag:
        # save 5 random images
        idxs = np.random.choice(range(xx_clean.shape[0]), count)
        for idx in idxs:
            plt.imshow(masked_xx[idx, 0].cpu(), cmap='gray')
            plt.savefig("figs/obs_masked_{}_beta_{}.png".format(idx, beta), bbox_inches='tight')
            plt.clf()

            plt.imshow(xx_clean[idx, 0].cpu(), cmap='gray')
            plt.savefig("figs/obs_clean_{}_beta_{}.png".format(idx, beta), bbox_inches='tight')
            plt.clf()
        temp_flag = True
