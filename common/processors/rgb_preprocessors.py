from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize, RandomResizedCrop, ColorJitter, InterpolationMode
from functools import partial
from typing import Any, Dict, List


class RandomShiftsAug(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        x = x.float()
        b, t, c, h, w = x.size()
        assert h == w
        x = x.view(b*t, c, h, w)  # reshape x to [B*T, C, H, W]
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        h_pad, w_pad = h + 2*self.pad, w + 2*self.pad  # calculate the height and width after padding
        eps = 1.0 / (h_pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(b*t, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(1, t, 1, 1, 1)  # repeat the shift for each image in the sequence
        shift = shift.view(b*t, 1, 1, 2)  # reshape shift to match the size of base_grid
        shift *= 2.0 / (h_pad)

        grid = base_grid + shift
        output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
        return output



class RGB_PreProcessor(torch.nn.Module): 
    def __init__(
            self,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            do_random_resized_crop=False,
            do_random_shift=False,
            crop_area_scale=(0.8, 1.0),
            crop_aspect_ratio=(3.0 / 4.0, 4.0 / 3.0),
            shift_pad=10
        ):
        super().__init__()
        self.rgb_shape = rgb_shape
        self.register_buffer('rgb_mean', torch.tensor(rgb_mean).view(-1, 1, 1))
        self.register_buffer('rgb_std', torch.tensor(rgb_std).view(-1, 1, 1))
        self.do_random_resized_crop = do_random_resized_crop
        self.do_random_shift= do_random_shift

        self.eval_transforms = torch.nn.Sequential(
            Resize(rgb_shape, interpolation=InterpolationMode.BICUBIC, antialias=True)
        )
        self.resize = Resize(rgb_shape, interpolation=InterpolationMode.BICUBIC, antialias=True)

        train_transforms = []
        if self.do_random_resized_crop:
            assert crop_area_scale is not None and crop_aspect_ratio is not None
            train_transforms.append(RandomResizedCrop(rgb_shape, crop_area_scale, crop_aspect_ratio, interpolation=InterpolationMode.BICUBIC, antialias=True))
        else:
            train_transforms.append(Resize(rgb_shape, interpolation=InterpolationMode.BICUBIC, antialias=True))
        if self.do_random_shift:
            assert shift_pad is not None
            train_transforms.append(RandomShiftsAug(pad=shift_pad))
        self.train_transforms = torch.nn.Sequential(*train_transforms)
        

    def forward(self, x, train=False , depth=False):
        x = x.float()*(1/255.)
        if train:
            x = self.train_transforms(x)
        else:
            x = self.eval_transforms(x)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        if depth :
            d_min = x.amin(dim=(-2, -1), keepdim=True)
            d_max = x.amax(dim=(-2, -1), keepdim=True)
            x = (x - d_min) / (d_max - d_min + 1e-6)
        else :   
            x = (x - self.rgb_mean) / (self.rgb_std + 1e-6)  
        return x

    def post_process(self, x , depth=False):
        if not depth :
            x = x * self.rgb_std + self.rgb_mean
            x = torch.clamp(x, min=0, max=1)
        else : 
            d_min = x.amin(dim=(-2, -1), keepdim=True)
            d_max = x.amax(dim=(-2, -1), keepdim=True)
            x = x * (d_max - d_min) + d_min
            x = torch.clamp(x, min=0, max=1)
        return x



if __name__ == "__main__":
    rgb_preprocessor = RGB_PreProcessor(
        rgb_shape=[224,224], rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5],
        do_random_resized_crop=True, do_random_shift=True, 
        crop_area_scale=[0.8,1.0], crop_aspect_ratio=[1.0,1.0], shift_pad=10)
    rgb_preprocessor.cuda()
    x = torch.zeros(4,2,3,224,224, dtype=torch.uint8).cuda()
    transformed_x = rgb_preprocessor(x, train=False)
    print(transformed_x)