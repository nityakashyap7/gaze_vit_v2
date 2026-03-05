import wandb
from typing import Any
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

class Logger:
    def __init__(self,
                 cfg: dict):

        env = cfg.data_pipeline.load_dataset.env # type: ignore
        loss_type = HydraConfig.get().runtime.choices.get("loss", "unknown") # unknown is just a fallback value for the .get() dict lookup. If for some reason "loss" isn't found in Hydra's runtime choices (e.g., someone runs the script without specifying a loss group)

        run_group = env 
        run_name = f'{env}_{loss_type}'

        self.run = wandb.init(
            project='gaze-vit-v2',
            config=OmegaConf.to_container(cfg, resolve=True),  # Convert to dict # type: ignore
            name=run_name,
            group=run_group,
            tags=['dev'] 
        )

    

