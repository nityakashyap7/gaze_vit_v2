from trainer import Trainer
from data_pipeline import DataPipeline
from omegaconf import OmegaConf
import torch


def main():
    variants = {
        'vit_s_14_CEBeforeAvgUS': 'config/vit_s_14_CEBeforeAvgUS.yaml',
        'vit_s_14_CE': 'config/vit_s_14_CE.yaml',
    }
    
    config = OmegaConf.load(variants['vit_s_14_CEBeforeAvgUS']) # later i shd separate trainer and dataloader configs out into separate yaml files
    
    torch.manual_seed(config.seed)

    data_pipeline = DataPipeline(config)

    trainer = Trainer(config=config, train_loader=data_pipeline.train_loader, val_loader=data_pipeline.val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
