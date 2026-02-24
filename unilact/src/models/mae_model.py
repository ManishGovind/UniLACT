import torch.nn.functional as F
from transformers import ViTMAEModel
from PIL import Image
import requests
import torch.nn as nn


class MaeEncoder(nn.Module):
    def __init__(self, use_obs_feature, pretrained_model_name_or_path):
        super().__init__()
        self.use_obs_feature = use_obs_feature
        self.image_encoder = ViTMAEModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.image_encoder.config.mask_ratio = 0.0

    def forward(self, images):
        vision_outputs = self.image_encoder(images)
        last_hidden_states = vision_outputs.last_hidden_state

        obs_features = last_hidden_states[:, 0, :]
        patch_features = last_hidden_states[:, 1:, :]

        if self.use_obs_feature:
            return obs_features, patch_features
        else:
            return None, patch_features

