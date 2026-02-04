# How to reproduce my results
erm TBA

# Repo structure
`dev/` contains mostly jupyter notebooks of experiment code. `images/` contains all the visuals i could put in the paper. `finalized` is the polished code for someone else who'd wanna reproduce my results.  

# Tools used
- Wandb for logging
- Pytorch for data loading, hook, training
-  Hydra for storing various model configs
- uv for package/env management

# Credits
`utils.py` (excluding `patchify_gaze_masks()`) and `gaze_to_mask.py` yoinked from the [GABRIL-Atari repo by Fatemeh Bahrani](https://github.com/nfbahrani/GABRIL-Atari)

