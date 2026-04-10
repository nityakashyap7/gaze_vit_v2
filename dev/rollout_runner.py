from GABRIL_utils.atari_env_manager import create_env
import wandb

class RolloutRunner:
    def __init__(self, model, attn_extractor=None, config): # attn_extractor can be None if we're testing against the model that doesn't do gaze reg
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
        attn_extractor = AttentionExtractor()

        return cls(model=model, attn_extractor=attn_extractor, config=config) 
