import omegaconf
import hydra
import pyrootutils
import os
import sys
import torch
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import json
from common.data.datasets import LMDBDataset_for_UniLACT_OXE, LMDBDataset_for_UniLACT_Video, LMDBDataset_Mix, JsonDataset_for_UniLACT_Video, NpzDataset_for_UniLACT_Video, LMDBDataset_for_UniLACT_CALVIN, LMDBDataset_for_UniLACT_CALVIN_UNIFIED
from common.data.datasets import    LMDBDataset_for_UniLACT_CALVIN_DEPTH, LMDBDataset_for_UniLACT_OXE_DEPTH, LMDBDataset_for_UniLACT_OXE_UNIFIED
from common.data.mix_utils import BASE_STEPSIZE, DISPLAY_KEY
from torchvision.transforms.v2 import Resize, InterpolationMode
from torch.utils.data import ConcatDataset, WeightedRandomSampler

data_type2dataset_cls = {
    'video': LMDBDataset_for_UniLACT_Video,
    'oxe': LMDBDataset_for_UniLACT_OXE,
    'oxe_depth': LMDBDataset_for_UniLACT_OXE_DEPTH,
    'oxe_unified': LMDBDataset_for_UniLACT_OXE_UNIFIED,
    'video_json': JsonDataset_for_UniLACT_Video,
    'video_npz': NpzDataset_for_UniLACT_Video,
    'calvin': LMDBDataset_for_UniLACT_CALVIN,
    'calvin_depth' : LMDBDataset_for_UniLACT_CALVIN_DEPTH,
    'calvin_unified' : LMDBDataset_for_UniLACT_CALVIN_UNIFIED
}

def load_dataset(data_config, extra_data_config):
    if type(data_config) is str:
        data_config = omegaconf.OmegaConf.load(data_config)
        data_config = dict(data_config)

    data_type = data_config.pop('data_type')
    if data_type == "calvin_mix" :
        modalities = data_config.pop('latent_modalities')

    key_map = {
        'latent_action_pred': 'do_extract_future_frames',
        'act_pred': 'do_extract_action'
    }
    for k, v in extra_data_config.items():
        mapped_k = key_map.get(k, k)
        data_config[mapped_k] = v

    if data_type == 'mix':
        sub_data_configs = data_config.pop('sub_data_configs')
        rgb_preprocessor = Resize(data_config['rgb_shape'], interpolation=InterpolationMode.BICUBIC, antialias=True)
        train_datasets = []
        eval_datasets = []
        train_sample_weights = []
        eval_sample_weights = []

        for sub_data_config in sub_data_configs:
            sub_data_config = dict(sub_data_config)
            data_name = sub_data_config.pop('data_name')
            weight = sub_data_config.pop('weight')
            if ('lmdb_dir' not in sub_data_config) and ('lmdb_dir' in data_config):
                sub_data_config['lmdb_dir'] = os.path.join(data_config['lmdb_dir'], data_name)
            if ('video_dir' not in sub_data_config) and ('video_dir' in data_config):
                sub_data_config['video_dir'] = os.path.join(data_config['video_dir'], data_name, DISPLAY_KEY.get(data_name, 'image'))
            if ('depth_video_dir' not in sub_data_config) and ('depth_video_dir' in data_config):
                sub_data_config['depth_video_dir'] = os.path.join(data_config['depth_video_dir'], data_name, DISPLAY_KEY.get(data_name, 'image'))    
            step_size = max(round(BASE_STEPSIZE.get(data_name, 1) / BASE_STEPSIZE['fractal20220817_data']), 1)
            sub_data_config['skip_frame'] = data_config['skip_frame'] * step_size
            
            if 'max_skip_frame' in data_config:
                sub_data_config['max_skip_frame'] = data_config['max_skip_frame'] * step_size
                
            sub_data_config['rgb_shape'] = data_config['rgb_shape']
            sub_data_config['rgb_preprocessor'] = rgb_preprocessor
            sub_data_config['modalities'] = data_config['modalities']

            train_dataset, eval_dataset =  load_dataset(sub_data_config, extra_data_config)
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_sample_weights.append(weight)
            eval_sample_weights.append(weight)

        
        if data_config['weighted']:
            train_dataset = LMDBDataset_Mix(datasets=train_datasets, sample_weights=train_sample_weights)
            eval_dataset = LMDBDataset_Mix(datasets=eval_datasets, sample_weights=eval_sample_weights)
        else:
            train_dataset = ConcatDataset(train_datasets)
            eval_dataset = ConcatDataset(eval_datasets)
            
    else:
        if data_type != "calvin_mix" :
            dataset_cls = data_type2dataset_cls[data_type]
            train_dataset = dataset_cls(split='train', **data_config)
            eval_dataset = dataset_cls(split='val', **data_config)
        else :
            train_datasets = []
            eval_datasets = []
            for dt in modalities:
                dataset_cls = data_type2dataset_cls[dt]
                train_dataset = dataset_cls(split='train', **data_config)
                eval_dataset = dataset_cls(split='val', **data_config)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                
            train_dataset = ConcatDataset(train_datasets)
            eval_dataset = ConcatDataset(eval_datasets)
                
    
    return train_dataset, eval_dataset