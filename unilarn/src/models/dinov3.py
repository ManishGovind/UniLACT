import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch.nn as nn


class Dinov3Encoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

    def forward(self, images):
        vision_outputs = self.image_encoder(images)
        last_hidden_states = vision_outputs.last_hidden_state
        print(last_hidden_states.shape)
        return last_hidden_states
        # obs_features = last_hidden_states[:, 0, :]
        # patch_features = last_hidden_states[:, 1:, :]

        # if self.use_obs_feature:
        #     return obs_features, patch_features
        # else:
        #     return None, patch_features

