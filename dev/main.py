from trainer import Trainer
from data_pipeline import DataPipeline
from omegaconf import OmegaConf
import torch

def main():
    config = OmegaConf.load('/scr/nityakas/gaze_vit_v2/config/vit_s_14_CEBeforeAvgUS.yaml') # later i shd separate trainer and dataloader configs out into separate yaml files
    
    torch.manual_seed(config.seed)

    data_pipeline = DataPipeline(config)

    trainer = Trainer(config=config, train_loader=data_pipeline.train_loader, val_loader=data_pipeline.val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
