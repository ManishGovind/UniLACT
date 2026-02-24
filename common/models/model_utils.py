import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import omegaconf
import hydra
import os
import sys
import torch
from unilact.src.models.unilact_policy_wraper import UniLACT_PolicyWraper
from transformers import AutoTokenizer
from common.processors.preprocessor_utils import get_model_vision_basic_config
import json

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model(pretrained_path):
    config_path = os.path.join(pretrained_path, "config.yaml")
    checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

    config = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config)
    model.config = config
    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    missing_root_keys = set([k.split(".")[0] for k in missing_keys])
    print('load ', checkpoint_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)
    total, trainable = count_params(model)
    print(f"Params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")
    return model

def load_unilact_policy(args):
    print(f"loading UniLACT from {args.unilact_path} ...")
    unilact = load_model(args.unilact_path)
    unilact.mask_latent_action_probability = args.mask_latent_action_probability
    unilact.config['mask_latent_action_probability'] = args.mask_latent_action_probability
    unilact_config = unilact.config

    lang_tokenizer = AutoTokenizer.from_pretrained(unilact_config['model_lang']['pretrained_model_name_or_path'])
    model_vision_basic_config = get_model_vision_basic_config(unilact_config['model_vision']['pretrained_model_name_or_path'])


    variant = {
        'test_chunk_size': args.test_chunk_size,
        'is_gripper_binary': args.is_gripper_binary,
        'use_temporal_ensemble': args.use_temporal_ensemble,
        'act_dim': unilact_config['act_dim'],
        'seq_len': unilact_config['sequence_length'],
        'chunk_size': unilact_config['chunk_size'],
        'mask_latent_action_probability': unilact_config['mask_latent_action_probability'],
        'latent_action_pred': unilact_config['latent_action_pred'],
        'per_latent_action_len': unilact_config['per_latent_action_len'],
        'pred_discrete_arm_action': unilact_config.get('pred_discrete_arm_action', False) 
    }
    variant.update(model_vision_basic_config)

    latent_action_decoding_kwargs = {
        'temperature': args.temperature, 
        'sample': args.sample, 
        'top_k': args.top_k, 
        'top_p': args.top_p,
        'beam_size': args.beam_size, 
        'parallel': args.parallel
    }

    eva = UniLACT_PolicyWraper(
        policy=unilact,
        variant=variant,
        latent_action_decoding_kwargs=latent_action_decoding_kwargs,
        lang_tokenizer=lang_tokenizer
    )

    return eva