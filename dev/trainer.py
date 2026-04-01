import torch
from utils.hook import AttentionExtractor
import utils.patch_gaze_masks
import torch.nn.functional as F
import losses
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from vit_pytorch import ViT
import numpy as np
from utils.logger import Logger
from itertools import islice
import wandb
import gymnasium as gym
import ale_py
from GABRIL_utils import atari_env_manager
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change
        os.makedirs("checkpoint_weights", exist_ok=True) # torch.save() wont create directories for u
        
        self._construct_components(train_loader, val_loader)

        self.wandb_logging_init()
        

    def _construct_components(self, train_loader, val_loader):
        self.loss_fn = instantiate(self.cfg.trainer.loss)

        self.model = instantiate(self.cfg.trainer.model).to(self.device)
        self.attn_extractor = AttentionExtractor()
        spatial_transformer_last_layer = self.model.spatial_transformer.layers[-1][0].attend
        self.handle = spatial_transformer_last_layer.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        spatial_transformer_last_layer.use_flash_attn = False # disable flash attention for the same layer(s) ur extracting weights from 
        
        self.optimizer = instantiate(self.cfg.trainer.optimizer)(params=self.model.parameters())
        self.scheduler = instantiate(self.cfg.trainer.scheduler)(optimizer=self.optimizer) 
        self.env = atari_env_manager.create_env(**self.cfg.create_env)
        
        
        self.train_loader = train_loader
        self.val_loader = val_loader

    def wandb_logging_init(self):
        env = self.cfg.data_pipeline.load_dataset.env # type: ignore
        reg_type = HydraConfig.get().runtime.choices.get('loss', 'unknown') # unknown is just a fallback value for the .get() dict lookup. If for some reason "loss" isn't found in Hydra's runtime choices (e.g., someone runs the script without specifying a loss group)
        reg_loss_fn = HydraConfig.get().runtime.choices.get('reg_loss_fn') or ''
        
        run_group = env 
        run_name = f'{env}_{reg_loss_fn}_{reg_type}'

        self.run = wandb.init(
            project='gaze-vit-v2',
            config=OmegaConf.to_container(self.cfg, resolve=True),  # Convert to dict # type: ignore
            name=run_name,
            group=run_group,
            tags=['debug'] 
        )

    def _train_step(self, batch):
        observations, action_targs, gaze_targs = batch # gaze_targs will just be torch zeroes if use_gaze is set to False

        observations = observations.to(self.device) # [batch_size, num_channels, H, W]: [64, 1, 84, 84]
        action_targs = action_targs.to(self.device) # [batch_size]: [64]
        gaze_targs = gaze_targs.to(self.device) # [batch_size, H, W]: [64, 84, 84]

        # forward pass
        self.optimizer.zero_grad()
        action_preds = self.model(observations) # [batch_size, len(action_space)]: [64, 18]

        # extract out attention weights
        gaze_preds = self.attn_extractor.cls_qkt_logits # [batch_size, num_heads, num_patches]: [64, 6, 36]
        
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

        wandb.finish()
        

    @torch.no_grad()
    def eval_validation_set(self, epoch):
        """evaluating current model on training and validation sets. returns avg loss for validation set"""

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

            # create and log confusion matrix
            action_preds = torch.cat(all_action_preds)  # still on GPU
            action_targs = torch.cat(all_action_targs)

            all_labels = np.arange(len(class_names))
            cm = confusion_matrix(action_targs.cpu().numpy(), action_preds.cpu().numpy(), labels=all_labels)
            n_classes = len(class_names)
            cell_size = 0.6
            fig_size = max(cell_size * n_classes, 6)
            fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size), dpi=150)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            ticks = np.arange(n_classes)
            ax.set(
                xticks=ticks, yticks=ticks,
                xticklabels=class_names, yticklabels=class_names,
                xlabel="Predicted", ylabel="True",
                title=f"{split} set confusion matrix (epoch {epoch})",
            )
            ax.tick_params(axis='x', labelrotation=90)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=7)
            fig.tight_layout()
            wandb.log({f'{split}/confusion_matrix': wandb.Image(fig)}, step=epoch)
            plt.close(fig)


        self.model.train()  # reset to training mode

        return val_loss
    

    @torch.no_grad()
    def eval_in_env(self):

        self.model.eval()
        num_episodes = self.cfg.trainer.eval_in_env.num_episodes
        record_eps = self.cfg.trainer.eval_in_env.record_eps
        frame_skip = self.cfg.trainer.eval_in_env.frame_skip
        wandb_playback_fps = self.cfg.trainer.eval_in_env.wandb_playback_fps

        scores = []
        for episode in range(num_episodes):

            done = False
            episode_reward = 0
            step = 0

            stacked_obs, info = self.env.reset(seed=self.cfg.seed + 1000 * episode)

            while not done:

                stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

                _ = self.env.get_wrapper_attr('render_')(gaze=None, record_frame=(episode < record_eps and step % frame_skip == 0))

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

            if episode < record_eps:
                video_path = f'vids/{self.cfg.env_name}_eval_ep{episode}.mp4'
                self.env.get_wrapper_attr('save_video')(video_path)
                self.env.get_wrapper_attr('img_list').clear()
                wandb.log({f'eval/video_ep{episode}': wandb.Video(video_path, fps=wandb_playback_fps, format='mp4')})

        mean_score, std_score = np.mean(scores), np.std(scores)

        wandb.log({
            'eval/mean_score': mean_score,
            'eval/std_score': std_score,
            'eval/min_score': np.min(scores),
            'eval/max_score': np.max(scores),
        })

        self.model.train()
        self.env.close()

        return mean_score, std_score
