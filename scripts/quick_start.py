import os
import sys
sys.path.append(os.getcwd())

import torch
from transformers import AutoConfig
from config_probe import parse_args
import numpy as np
import random
import copy

from model.tactile_mae import TactileVideoMAE
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

### AnyTouch 2 normalization parameters
offset = 130.0 / 255.0
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

### AnyTouch 2 sensor ID
sensor_name_to_id = {
    'gelsight': 0,
    'digit': 1,
    'gelslim': 2,
    'gelsight_mini': 3,
    'duragel': 4,
    'dm': 5,
    'universal': -1 # universal token works for all sensors (DO NOT USE FOR FORCE PREDICTION)
}

def load_data(args):

    transform = transforms.Compose([
                        transforms.Resize(size=(224, 224), antialias=True),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    to_tensor = transforms.ToTensor()

    data_path = 'example_data/'
    bg_img = to_tensor(Image.open(data_path + 'bg.png').convert('RGB')).unsqueeze(0)  # 1, C, H, W
    img_list = []
    start_idx = 0
    for i in range(start_idx, start_idx + args.num_frames * args.stride, args.stride):
        img_list.append(to_tensor(Image.open(data_path + str(i) + '.png').convert('RGB')))
    img_list = torch.stack(img_list, dim=0)  # T, C, H, W

    img_list = img_list - bg_img + offset
    img_list = torch.clamp(img_list, 0.0, 1.0)
    img_list_transformed = transform(img_list)  # T, C, H, W

    return img_list, img_list_transformed

def load_model_from_multi_clip(ckpt, model):

    new_ckpt = {}
    for key,item in ckpt.items():
        # print(key)
        if "touch_mae_model" in key and 'decoder' not in key and 'mask_token' not in key:
            new_ckpt[key.replace('touch_mae_model.','')] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    device = torch.device(args.device)

    seed = args.seed
    random_seed(seed)

    if args.model_size == 'base':
        config = AutoConfig.from_pretrained('CLIP-B-16/config.json')
    else:
        raise NotImplementedError
    
    if args.model == 'anytouch':

        # tube_size is unused
        model = TactileVideoMAE(args, config, args.num_frames, 1)  # tube_size=1 is actually not used

        load_dir = args.load_path
        ckpt = torch.load(load_dir, map_location='cpu')
        model = load_model_from_multi_clip(ckpt, model)
        print(load_dir)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented!')
    
    model.to(device)
    data, data_transformed = load_data(args)
    Batch_size = 1
    data_transformed = data_transformed.unsqueeze(0).to(device)  # B, T, C, H, W
    print('Input Data shape:', data_transformed.shape)  # T, C, H, W

    print('Visualization starting...')
    vis_dir = args.output_dir
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(data.shape[0]):
        plt.imsave(vis_dir + f"/input_{i}.png", data[i].permute(1, 2, 0).numpy())
        plt.close()
    print(f'Input images saved to {vis_dir}')

    print('Getting Sensor IDs')
    print(f'Using sensor: {args.data_sensor}')
    print(f'Sensor ID: {sensor_name_to_id[args.data_sensor]}')
    print('Sensor ID mapping:', sensor_name_to_id)
    sensor_id = sensor_name_to_id[args.data_sensor]
    sensor_id_tensor = torch.ones((Batch_size,), dtype=torch.long, device=device) * sensor_id
    print('Sensor ID tensor shape:', sensor_id_tensor.shape)

    print('Inference...')
    model.eval()
    with torch.no_grad():
        outputs = model(data_transformed, sensor_id_tensor, probe=True)
        print('Model output feature shape (before projection):', outputs.shape)
        ## Should be (1, 398, 768) for 4frames model. 398 = 1 (cls token) + 5 (sensor tokens) + 196 (patches) * 2 (time dim)

        outputs = model(data_transformed, sensor_id_tensor)
        print('Model output feature shape (after projection):', outputs.shape) 
        ## Should be (1, 398, 512) for 4frames model. 398 = 1 (cls token) + 5 (sensor tokens) + 196 (patches) * 2 (time dim)

        cls_token = outputs[:, 0, :]  # B, D
        print('CLS token shape:', cls_token.shape)

        sensor_tokens = outputs[:, 1:6, :]  # B, num_sensor_tokens, D
        print('Sensor tokens shape:', sensor_tokens.shape)

        feature_patches = outputs[:, 6:, :]  # B, num_patches, D
        print('Feature patches shape:', feature_patches.shape)

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)