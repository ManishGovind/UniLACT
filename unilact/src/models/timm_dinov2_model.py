import torch.nn.functional as F
import torch.nn as nn
import timm
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from functools import partial
import os

os.environ['XDG_CACHE_HOME']="/path/to/cache"

class TimmDinoV2Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class TimmDinoV2VsionEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path="vit_large_patch14_reg4_dinov2.lvd142m", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.featurizer = timm.create_model(
            model_name=pretrained_model_name_or_path,
            pretrained=True,
            num_classes=0,
            img_size=224,
            act_layer=None,
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )

    def forward(self, images):
        patch_features = self.featurizer(images)
        return None, patch_features

if __name__ == '__main__':
    import torch
    img = torch.randn(4,3,224,224).cuda()
    model = TimmDinoV2VsionEncoder().cuda()
    outputs = model(img)
    print(outputs[-1].shape)