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
from unilact.src.trainers.unilact_trainer import UniACT_Trainer
from torch.utils.data import DataLoader,ConcatDataset
from functools import partial
from common.data.data_utils import load_dataset
from common.data.modality_sampler import RandomModalityBatchDataloader  
from torch.utils.data import Subset
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

def main(cfg):
    # Prepare UniLACT
    seed = cfg['training_config'].get('seed', None)
    if seed is not None:
        from transformers import set_seed
        print(f"set_seed: {seed}")
        set_seed(seed)

    unilact_config_path = cfg['unilact_config_path']
    print(f"initializing UniLACT from {unilact_config_path} ...")
    print(cfg)
    unilact_config = omegaconf.OmegaConf.load(unilact_config_path)
    unilact = hydra.utils.instantiate(unilact_config)
    unilact.config = unilact_config

    # Prepare lang_tokenizer and rgb_processor
    lang_tokenizer = AutoTokenizer.from_pretrained(unilact_config['model_lang']['pretrained_model_name_or_path'])
    rgb_preprocessor = get_rgb_preprocessor(**cfg['rgb_preprocessor_config'])

    # Prepare Latent Motion Tokenizer
    if unilact_config['latent_action_pred']:
        unilarn_path = cfg['unilarn_path']
        assert unilarn_path is not None
        print(f"loading  UniLARN from {unilarn_path} ...")
        unilarn = load_model(unilarn_path)
    else:
        unilarn=None

    vq_remap = cfg.get('vq_remap', None)
    unknown_index = cfg.get('unknown_index', 'closest')
    if vq_remap is not None:
        unilarn.vector_quantizer.setup_remap(remap=vq_remap, unknown_index=unknown_index)

    # Preprepare Dataloaders
    dataset_config_path = cfg['dataset_config_path']
    
    extra_data_config = {
        'sequence_length': unilact_config['sequence_length'],
        'chunk_size': unilact_config['chunk_size'],
        'act_dim': unilact_config['act_dim'],
        'do_extract_future_frames': unilact_config['latent_action_pred'],
        'do_extract_action': unilact_config['act_pred']
    }

    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    dataloader_cls = partial(
        DataLoader, 
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        persistent_workers=True,
        num_workers=cfg['dataloader_config']['workers_per_gpu'],
        batch_size=cfg['dataloader_config']['bs_per_gpu'],
        prefetch_factor= cfg['dataloader_config']['prefetch_factor'],
        drop_last=False
    )
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    if isinstance(train_dataset, ConcatDataset) and len(train_dataset.datasets) == 3 :
        
        
        print("Using multi-modal training with", len(train_dataset.datasets), "modalities")

        modality_names = ['rgb','depth','unified']  

        train_dataloaders = {}
        eval_dataloaders = {}
        # MAX_SAMPLES = 32
        for i, modality in enumerate(modality_names):
            train_ds = train_dataset.datasets[i]
            eval_ds = eval_dataset.datasets[i]
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=True)
            train_dataloaders[modality] = dataloader_cls(train_ds, sampler=train_sampler, shuffle=False)
            eval_dataloaders[modality] = dataloader_cls(eval_ds, sampler=eval_sampler, shuffle=False)

        train_dataloader = RandomModalityBatchDataloader(train_dataloaders)
        eval_dataloader = RandomModalityBatchDataloader(eval_dataloaders)
    
    elif isinstance(train_dataset, ConcatDataset) and len(train_dataset.datasets) > 3 :
        
        
        print("Using multi-modal training with", len(train_dataset.datasets), "modalities")

        modality_names = ['rgb','depth','unified']  
        WINDOW_SIZE = 14
        train_dataloaders = {}
        eval_dataloaders = {}
        # MAX_SAMPLES = 32
        for i, modality in enumerate(modality_names):
            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE
            train_ds = ConcatDataset(train_dataset.datasets[start:end])
            eval_ds  = ConcatDataset(eval_dataset.datasets[start:end])
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=True)

            train_dataloaders[modality] = dataloader_cls(train_ds, sampler=train_sampler, shuffle=False)
            eval_dataloaders[modality] = dataloader_cls(eval_ds, sampler=eval_sampler, shuffle=False)

        train_dataloader = RandomModalityBatchDataloader(train_dataloaders)
        eval_dataloader = RandomModalityBatchDataloader(eval_dataloaders)
        
   
        
    else:
        train_dataloader = dataloader_cls(train_dataset)
        eval_dataloader = dataloader_cls(eval_dataset)


    print("training dataloader" , len(train_dataloader))
    print("eval dataloader" , len(eval_dataloader))


    # Prepare Trainer
    trainer = UniACT_Trainer(
        unilact=unilact,
        unilact_config=unilact_config,
        unilarn=unilarn,
        rgb_preprocessor=rgb_preprocessor,
        lang_tokenizer=lang_tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        bs_per_gpu=cfg['dataloader_config']['bs_per_gpu'],
        **cfg['training_config']
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/path/to/unilact/configs/train/data_calvin-model_actPredTrue_motionPredTrue_visionMaeLarge_seq2_chunk5_maskProb0.5-train_lr0.0002_bs512-aug_shiftTrue_resizedCropFalse-resume_from_predLatentOnly_calvin_Epoch10.yaml")
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.config_path)
    main(cfg)

    


