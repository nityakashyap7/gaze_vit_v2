# How to reproduce my results
## Usage
### plain CE (defaults)
`python main.py`

### gaze loss with CE regularization
`python main.py loss=ce_before_avg_us reg_loss_fn=ce`

### gaze loss with BCE regularization
`python main.py loss=ce_before_avg_us reg_loss_fn=bce`

### override hyperparams
`python main.py loss=ce_before_avg_us reg_loss_fn=bce lr=0.001 trainer.training.num_epochs=10`

### sweep all loss + reg_loss_fn combos
`python main.py -m loss=ce_before_avg_us,ce_after_avg_us reg_loss_fn=ce,bce`

### print resolved config without running (useful for debugging)
`python main.py --cfg job loss=ce_before_avg_us reg_loss_fn=ce`

---

# Repo structure
`dev/` contains jupyter notebooks of experiment code, core data processing/patching and training pipeline. `finalized/` will be the polished code for someone else who'd wanna reproduce my results.  

## Config structure
```
dev/config/
  base.yaml                    # shared settings + defaults list
  model/
    vit_s_14.yaml              # ViT-Small model params
  loss/
    ce.yaml                    # plain CE, no gaze
    ce_before_avg_us.yaml      # gaze-regularized, upsample before avg
    ce_after_avg_us.yaml       # gaze-regularized, upsample after avg
  reg_loss_fn/
    ce.yaml                    # F.cross_entropy as reg loss
    bce.yaml                   # F.binary_cross_entropy_with_logits as reg loss
  vit_s_14_CE.yaml             # (old, kept for reference)
  vit_s_14_CEBeforeAvgUS.yaml  # (old, kept for reference)
  ```


# Tools used
- Wandb for logging
- Pytorch for data loading, hook, training
-  Hydra for storing various model configs
- uv for package/env management

# Credits
everything in `GABRIL_utils/` yoinked from the [GABRIL-Atari repo by Fatemeh Bahrani](https://github.com/nfbahrani/GABRIL-Atari)

