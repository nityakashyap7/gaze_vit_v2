import torch
from utils.hook import AttentionExtractor
import utils.patch_gaze_masks
import torch.nn.functional as F
import losses
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from vit_pytorch import ViT
from utils.hook import AttentionExtractor
import numpy as np
from utils.logger import Logger
from utils.hook import AttentionExtractor
from itertools import islice
import wandb
import gymnasium as gym
import ale_py


class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change

        self._construct_components(train_loader, val_loader)
        

    def _construct_components(self, train_loader, val_loader):
        self.loss_fn = instantiate(self.cfg.trainer.loss)
        self.model = instantiate(self.cfg.trainer.model).to(self.device)
        self.attn_extractor = AttentionExtractor()
        self.handle = self.model.transformer.layers[-1][0].attend.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        self.optimizer = instantiate(self.cfg.trainer.optimizer)(params=self.model.parameters())
        self.scheduler = instantiate(self.cfg.trainer.scheduler)(optimizer=self.optimizer) 
        self.logger = Logger(self.cfg)
        self.env = gym.make(f'ALE/{self.cfg.data_pipeline.load_dataset.env}-v5')
        
        self.train_loader = train_loader
        self.val_loader = val_loader


    def _train_step(self, batch):
        observations, action_targs, gaze_targs = batch # gaze_targs will just be torch zeroes if use_gaze is set to False

        observations = observations.to(self.device)
        action_targs = action_targs.to(self.device)
        gaze_targs = gaze_targs.to(self.device)

        # forward pass
        self.optimizer.zero_grad()
        action_preds = self.model(observations)

        # extract out attention weights
        gaze_preds = self.attn_extractor.cls_qkt_logits
        
        # calculate loss w regularization
        loss = self.loss_fn(action_preds=action_preds, action_targs=action_targs, gaze_preds=gaze_preds, gaze_targs=gaze_targs)
        
        # backward pass
        loss.backward()

        # update model weights in opposite direction of gradient
        self.optimizer.step()


    def train(self):
        for epoch in range(self.cfg.trainer.training.num_epochs):

            for batch in self.train_loader:
                self._train_step(batch)


            self.eval_validation_set(epoch=epoch)
            
            # change lr (cosine annealing) each epoch 
            self.scheduler.step()
            print(f"Epoch {epoch} done!")
        
        self.eval_in_env()
        

    @torch.no_grad()
    def eval_validation_set(self, epoch):
        """evaluating current model on training and validation sets. spits out avg loss for both in a dictionary w 2 keys"""

        val_iters = len(self.val_loader)
        splits = {"train": self.train_loader, "val": self.val_loader}
        class_names = self.env.unwrapped.get_action_meanings() # type: ignore

        self.model.eval()

        for split, loader in splits.items():
            losses = torch.zeros(val_iters)
            all_action_preds = []
            all_action_targs = []

            for i, batch in enumerate(islice(loader, val_iters)):
                observations, action_targs, gaze_targs = batch # gaze_targs will just be torch zeroes if use_gaze is set to False

                observations = observations.to(self.device)
                action_targs = action_targs.to(self.device)
                gaze_targs = gaze_targs.to(self.device)

                logits = self.model(observations)
                action_preds = F.softmax(logits, dim=-1).argmax(dim=-1)

                gaze_preds = self.attn_extractor.cls_qkt_logits # technically the confusion matrix doesnt need this but the loss fn does. also i might wanna do sth with this later

                loss = self.loss_fn(action_preds=logits, action_targs=action_targs, gaze_preds=gaze_preds, gaze_targs=gaze_targs)

                losses[i] = loss.item()

                all_action_preds.append(action_preds)
                all_action_targs.append(action_targs)

            wandb.log({f'{split}_avg_loss': losses.mean()})

            # log action preds and targs for wandb confusion matrix (for both training and validation set data)
            action_preds = torch.cat(all_action_preds)  # still on GPU
            action_targs = torch.cat(all_action_targs)

            wandb.log({f'{split}_conf_mat': wandb.plot.confusion_matrix(
                y_true=action_targs.cpu().numpy().tolist(),
                preds=action_preds.cpu().numpy().tolist(),
                class_names=class_names
            )})


        self.model.train()  # reset to training mode
    

    @torch.no_grad()
    def eval_in_env(self):
        # TODO: add code to play a few episodes in the env
        self.env.close()