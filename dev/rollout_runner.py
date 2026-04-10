from GABRIL_utils.atari_env_manager import create_env
from utils.hook import AttentionExtractor
import wandb
import numpy as np

class RolloutRunner:
    def __init__(self, model, attn_extractor, config): # attn_extractor can be None if we're testing against the model that doesn't do gaze reg
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change

        self.model = model
        self.attn_extractor = attn_extractor
        # The hook itself lives on the model, not on the extractor. The extractor is just the thing that receives the data when the hook fires. So when you pass self.model to RolloutRunner, the hook is already baked in.

        self.envs = {
            game: create_env(env_name=game, **self.cfg.create_env) 
            for game in self.cfg.games
            }

    @classmethod
    def from_wandb_checkpoint(cls, url, config): # notice how ur passing in cls instead of self
        model = wandb.load_model(url)
        attn_extractor = AttentionExtractor(model)

        return cls(model=model, attn_extractor=attn_extractor, config=config) 


    @torch.no_grad()
    def run(self):
        self.model.eval()
        num_episodes = self.cfg.trainer.eval_in_env.num_episodes
        record_eps = self.cfg.trainer.eval_in_env.record_eps
        frame_skip = self.cfg.trainer.eval_in_env.frame_skip
        wandb_playback_fps = self.cfg.trainer.eval_in_env.wandb_playback_fps

        for env_name, env in self.envs:
            
            scores = []
            for episode in range(num_episodes):

                done = False
                episode_reward = 0
                step = 0

                stacked_obs, info = env.reset(seed=self.cfg.seed + 1000 * episode)

                while not done:

                    stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

                    _ = env.get_wrapper_attr('render_')(gaze=None, record_frame=(episode < record_eps and step % frame_skip == 0))

                    stacked_obs = torch.as_tensor(stacked_obs, device=self.device, dtype=torch.float32).unsqueeze(0) / 255.0
                    action = self.model(stacked_obs).argmax(1)[0].cpu().item()

                    stacked_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated


                    episode_reward += float(reward)
                    step += 1

                    if step > 5000:
                        done = True

                scores.append(episode_reward)
                print(f"Episode {episode} reward: {episode_reward} (mean: {np.mean(scores):.1f}, std: {np.std(scores):.1f}), steps: {step}")

                if episode < record_eps:
                    video_path = f'vids/{self.cfg.env_name}_eval_ep{episode}.mp4'
                    env.get_wrapper_attr('save_video')(video_path)
                    env.get_wrapper_attr('img_list').clear()
                    wandb.log({f'eval/{env_name}_video_ep{episode}': wandb.Video(video_path, fps=wandb_playback_fps, format='mp4')})

            mean_score, std_score = np.mean(scores), np.std(scores)

            wandb.log({
                'eval/mean_score': mean_score,
                'eval/std_score': std_score,
                'eval/min_score': np.min(scores),
                'eval/max_score': np.max(scores),
            })

            self.model.train()
            env.close()

            return mean_score, std_score
