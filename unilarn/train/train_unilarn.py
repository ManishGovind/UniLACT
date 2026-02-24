import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import argparse
import json
from torch.utils.data import DataLoader
import omegaconf
import hydra
from functools import partial
from transformers import AutoTokenizer
from common.models.model_utils import load_model
from common.processors.preprocessor_utils import get_rgb_preprocessor
from unilarn.src.trainers.unilarn_trainer import UniLARN_Trainer
from torch.utils.data import DataLoader
from functools import partial
from common.data.data_utils import load_dataset

def main(cfg):
    # Prepare UniLARN 
    unilarn_config_path = cfg['unilarn_config_path']
    print(f"initializing UniLARN from {unilarn_config_path} ...")
    unilarn_config = omegaconf.OmegaConf.load(unilarn_config_path)
    unilarn = hydra.utils.instantiate(unilarn_config)
    unilarn.config = unilarn_config

    # Prepare rgb_processor
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor']
    )
    train_dataloader = dataloader_cls(train_dataset)
    eval_dataloader = dataloader_cls(eval_dataset)
    
    # Prepare Trainer
    trainer = UniLARN_Trainer(
        unilarn=unilarn,
        rgb_preprocessor=rgb_preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/path/to/unilarn/configs/train/data_calvin_vq_multi_modal.yaml")
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    


