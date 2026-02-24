import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from common.processors.rgb_preprocessors import RGB_PreProcessor

def get_model_vision_basic_config(model_vision_type):
    model_vision_type = model_vision_type.lower()
    if "theia" in model_vision_type:
        rgb_shape = [224, 224]
        rgb_mean = [0.5, 0.5, 0.5]
        rgb_std = [0.5, 0.5, 0.5]
    elif "mae" in model_vision_type:
        rgb_shape = [224, 224]
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
    elif "dino" in model_vision_type:
        rgb_shape = [224, 224]
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError

    model_vision_basic_config = {
        "rgb_shape": rgb_shape,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std
    }
    return model_vision_basic_config


def get_rgb_preprocessor(model_vision_type, vision_aug_config={}):
    model_vision_basic_config = get_model_vision_basic_config(model_vision_type)
    rgb_preprocessor = RGB_PreProcessor(
        **model_vision_basic_config,
        **vision_aug_config
    )
    return rgb_preprocessor