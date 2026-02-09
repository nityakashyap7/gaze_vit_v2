import torch
from utils.hook import AttentionExtractor
from GABRIL_utils.utils import load_dataset
from utils.patch_gaze_masks import patch_gaze_masks
import torch.nn.functional as F
import losses
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from vit_pytorch import ViT
from utils.hook import AttentionExtractor

class Trainer:
    def __init__(self, config):
        self.cfg = config
        self._construct_components()

    def _construct_components(self):
        self.loss_fn = instantiate(self.cfg.loss)
        self.model = instantiate(self.cfg.model)
        self.attn_extractor = instantiate(self.cfg.attention_extractor)
        self.handle = self.model.transformer.layers[-1][0].attend.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

        
    def _train_step(self, batch):
        observations, action_targs, gaze_targs = batch

        # forward pass
        self.optimizer.zero_grad()
        action_preds = self.model(observations)

        # extract out attention weights
        gaze_preds = self.attn_extractor.cls_qkt_logits

        # calculate loss w regularization
        loss = self.loss_fn(action_preds, action_targs, gaze_preds, gaze_targs)
        
        # backward pass
        loss.backward()

        # update model weights in opposite direction of gradient
        self.optimizer.step()
        
        return loss.item() # so train() can log it


    def train(self):
        pass