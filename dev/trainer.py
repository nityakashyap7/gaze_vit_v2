import torch
from utils.hook import AttentionExtractor
import utils.patch_gaze_masks
import torch.nn.functional as F
import losses
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from vit_pytorch import ViT
import numpy as np
from utils.logger import Logger
from itertools import islice
import wandb
import gymnasium as gym
import ale_py
from GABRIL_utils import atari_env_manager
import os


class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change
        os.makedirs("checkpoint_weights", exist_ok=True) # torch.save() wont create directories for u
        
        self._construct_components(train_loader, val_loader)
        

    def _construct_components(self, train_loader, val_loader):
        self.loss_fn = instantiate(self.cfg.trainer.loss)
        self.model = instantiate(self.cfg.trainer.model).to(self.device)
        self.attn_extractor = AttentionExtractor()
        self.handle = self.model.transformer.layers[-1][0].attend.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        self.optimizer = instantiate(self.cfg.trainer.optimizer)(params=self.model.parameters())
        self.scheduler = instantiate(self.cfg.trainer.scheduler)(optimizer=self.optimizer) 
        self.logger = Logger(self.cfg)
        self.env = atari_env_manager.create_env(self.cfg.create_env)
        
        
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
        best_val_loss = float('inf') # for early stopping (loss starts high and is supposed to decrease as the model gets smarter)
        best_epoch = -1
        patience_count = 0
        model_name = HydraConfig.get().runtime.choices['model']  # eg "vit_s_14"
        path = f'checkpoint_weights/{model_name}_best.pt'

        for epoch in range(self.cfg.trainer.training.num_epochs):

            for batch in self.train_loader:
                self._train_step(batch)

            val_loss = self.eval_validation_set(epoch=epoch)

            # stop early?
            if val_loss < best_val_loss:
                # save the model's weights
                torch.save(self.model.state_dict(), path)
                artifact = wandb.Artifact(name=f'{model_name}_best', type='model')
                artifact.add_file(path) # this is an upload function, so it has to already exist on ur local machine
                wandb.log_artifact(artifact)

                # reset patience counter
                patience_count = 0
                
                # new record yay!! 
                best_val_loss = val_loss

                # update which epoch gave the best results
                best_epoch = epoch

            else:
                patience_count += 1

                if patience_count >= self.cfg.trainer.training.patience:
                    print(f"Stopping training early: {epoch} epochs completed. Weights will be rolled back to epoch {best_epoch}.")
                    break

            # change lr (cosine annealing) each epoch 
            self.scheduler.step()

            
            print(f"Epoch {epoch} done!")
        
        # log which epoch gave the best validation loss
        wandb.log({'best epoch': best_epoch})

        self.model.load_state_dict(torch.load(path, weights_only=True)) # roll back weights for eval in env 
        self.eval_in_env()
        

    @torch.no_grad()
    def eval_validation_set(self, epoch):
        """evaluating current model on training and validation sets. spits out avg loss for both in a dictionary w 2 keys"""

        val_iters = len(self.val_loader)
        splits = {'train': self.train_loader, 'val': self.val_loader}
        class_names = self.env.unwrapped.get_action_meanings() # type: ignore
        val_loss = 0

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

            avg_loss = losses.mean()

            if split == 'val':
                val_loss = avg_loss

            wandb.log({f'{split}_avg_loss': avg_loss}, step=epoch) # wandb auto-increments its internal step counter each time you call wandb.log(), so the metrics might end up on different x-axis steps. pass an explicit step=epoch to keep them aligned.

            # log action preds and targs for wandb confusion matrix (for both training and validation set data)
            action_preds = torch.cat(all_action_preds)  # still on GPU
            action_targs = torch.cat(all_action_targs)

            wandb.log({f'{split}_conf_mat': wandb.plot.confusion_matrix(
                y_true=action_targs.cpu().numpy().tolist(),
                preds=action_preds.cpu().numpy().tolist(),
                class_names=class_names
            )}, step=epoch)


        self.model.train()  # reset to training mode

        return val_loss
    

    @torch.no_grad()
    def eval_in_env(self):
        self.model.eval()
        num_episodes = self.cfg.data_pipeline.load_dataset.num_episodes

        scores = []
        for episode in range(num_episodes):

            done = False
            episode_reward = 0
            step = 0

            stacked_obs, info = self.env.reset(seed=self.cfg.seed + 1000 * episode)

            while not done:

                stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

                _ = self.env.get_wrapper_attr('render_')(gaze=None, record_frame=episode < 2)

                stacked_obs = torch.as_tensor(stacked_obs, device=self.device, dtype=torch.float32).unsqueeze(0) / 255.0
                action = self.model(stacked_obs).argmax(1)[0].cpu().item()

                stacked_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated


                episode_reward += float(reward)
                step += 1

                if step > 5000:
                    done = True

            scores.append(episode_reward)
            print(f"Episode {episode} reward: {episode_reward} (mean: {np.mean(scores):.1f}, std: {np.std(scores):.1f}), steps: {step}")

        mean_score, std_score = np.mean(scores), np.std(scores)

        wandb.log({
            'eval/mean_score': mean_score,
            'eval/std_score': std_score,
            'eval/min_score': np.min(scores),
            'eval/max_score': np.max(scores),
        })

        video_path = f'vids/{self.cfg.env_name}_eval.mp4'
        self.env.get_wrapper_attr('save_video')(video_path)
        wandb.log({'eval/video': wandb.Video(video_path, fps=15, format='mp4')})

        self.model.train()
        self.env.close()

        return mean_score, std_score