import torch
from utils.hook import AttentionExtractor
from utils.patch_gaze_masks import patch_gaze_masks
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
from GABRIL_utils import utils
from GABRIL_utils.atari_env_manager import create_env as create_atari_environment

class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change
        self._construct_components(train_loader, val_loader)
        

    def _construct_components(self, train_loader, val_loader):
        self.env = create_atari_environment(self.cfg.env_name) # not in yaml bc its not gna change, also we only need env for evaluation, so maybe should move this to evaluate() method instead of init
        self.loss_fn = instantiate(self.cfg.trainer.loss)
        self.model = instantiate(self.cfg.trainer.model).to(self.device)
        self.attn_extractor = AttentionExtractor()
        self.handle = self.model.transformer.layers[-1][0].attend.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        self.optimizer = instantiate(self.cfg.trainer.optimizer)(params=self.model.parameters())
        self.logger = Logger(self.cfg)
        
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
        
        return {'train_loss' : loss.item()} # so train() can log it


    def train(self):
        for epoch in range(self.cfg.trainer.training.num_epochs):
            loss_dict = {}
            
            for batch in self.train_loader:
                loss_dict = self._train_step(batch)

            # TODO: run validation and add val_loss to loss_dict

            self.logger.log_scalar_dict(loss_dict, step=epoch) # log last timestep in the epoch
            
            if epoch % self.cfg.evaluation.eval_interval == 0:
                self.evaluate(epoch)

            print(f"Epoch {epoch} done!")

    def evaluate(self, step = 0):

        self.model.eval() 
        num_eval_episodes = self.cfg.evaluation.num_eval_episodes
        rewards = [0 for i in range(num_eval_episodes)]

        with torch.no_grad():
            for episode in range(num_eval_episodes):
                obs, _ = self.env.reset()
                done = False

                while not done:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)/255.0 # add batch dim

                    
                    action_pred = self.model(obs_tensor)

                    action = torch.argmax(action_pred, dim=-1).item() # get the predicted action index
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    rewards[episode] += reward
                    
                    


        avg_reward = np.mean(rewards)   
        self.logger.log_scalar_dict({'avg_eval_reward': avg_reward}, step=step)
        self.model.train()
        

