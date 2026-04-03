import torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from GABRIL_utils.atari_env_manager import create_env
import wandb


class RolloutRunner:

    def __init__(self, model, envs, device, seed: int = 42, model_checkpoint = None):


        if model_checkpoint is not None:
            self.load_checkpoint(model_checkpoint)
        else:
            self.model = model
        self.envs = envs
        self.device = device
        self.seed = seed

    def run(self, num_episodes = 10, record_eps = 0, frame_skip = 1, video_dir ='vids'):
        
        self.model.eval()
        results = {}

        with torch.no_grad():
            for game_name, env in self.envs.items():
                scores = []
                video_paths = []

                for episode in range(num_episodes):
                    done = False
                    episode_reward = 0.0
                    step = 0

                    stacked_obs, _ = env.reset(seed=self.seed + 1000 * episode)

                    while not done:
                        stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

                        env.get_wrapper_attr('render_')(
                            gaze=None,
                            record_frame=(episode < record_eps and step % frame_skip == 0)
                        )

                        obs_tensor = torch.as_tensor(stacked_obs, device=self.device, dtype=torch.float32).unsqueeze(0) / 255.0
                        action = self.model(obs_tensor).argmax(1)[0].cpu().item()

                        stacked_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        episode_reward += float(reward)
                        step += 1

                        if step > 5000:
                            done = True

                    scores.append(episode_reward)
                    print(f"[{game_name}] Episode {episode}: reward={episode_reward:.1f}  (mean so far: {np.mean(scores):.1f})")

                    if episode < record_eps:
                        video_path = f'{video_dir}/{game_name}_ep{episode}.mp4'
                        env.get_wrapper_attr('save_video')(video_path)
                        video_paths.append(video_path)

                results[game_name] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores)),
                    'episode_scores': scores,
                    'video_paths': video_paths,
                }

        self.model.train()
        return results

    @classmethod
    def from_checkpoint(cls, run_path, games, seed, device='cuda'):


        api = wandb.Api()
        run = api.run(run_path)

        cfg = OmegaConf.create(run.config)

        # download model weights artifact
        artifact = api.artifact(f'{run_path}/model:latest')
        artifact_dir = artifact.download()
        weights_path = f'{artifact_dir}/model.pth'

        model = instantiate(cfg.trainer.model).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        envs = {game: create_env(game) for game in games}

        return cls(model=model, envs=envs, device=torch.device(device), seed=seed)

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        