import wandb
from typing import Any
from omegaconf import OmegaConf

class Logger:
    def __init__(self,
                 cfg: dict):
        
        env = cfg.data_pipeline.load_dataset.env # type: ignore
        loss_type = cfg.loss_name # type: ignore

        run_group = env 
        run_name = f'{env}_{loss_type}'

        self.run = wandb.init(
            project='gaze-vit-v2',
            config=OmegaConf.to_container(cfg, resolve=True),  # Convert to dict # type: ignore
            name=run_name,
            group=run_group,
            tags=['dev'] 
        )
    
    def log_scalar_dict(self, scalar_dict:dict, step:int) -> None:
        self.run.log(scalar_dict, step)

    

