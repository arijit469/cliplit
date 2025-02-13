import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import model_small
import numpy as np
from PIL import Image
import glob
from collections import OrderedDict

# --- Check CUDA availability --- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Parse hyper-parameters --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of CLIP-LIT (Liang, 2023)')
parser.add_argument('-i', '--input', help='Directory of input folder', default='./input/')
parser.add_argument('-o', '--output', help='Directory of output folder', default='./inference_result/')
parser.add_argument('-c', '--ckpt', help='Test checkpoint path', default='./pretrained_models/enhancement_model.pth')

args = parser.parse_args()

# --- Initialize model --- #
U_net = model_small.UNet_emb_oneBranch_symmetry(3, 1)

# --- Load model checkpoint correctly --- #
state_dict = torch.load(args.ckpt, map_location=device)

# Remove 'module.' prefix if it exists in checkpoint keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # Remove 'module.' if present
    new_state_dict[name] = v

U_net.load_state_dict(new_state_dict)
U_net.to(device)  # Move model to correct device

# --- Define low-light enhancement function --- #
def lowlight(image_path): 
    data_lowlight = Image.open(image_path)  # Load image
    data_lowlight = np.asarray(data_lowlight) / 255.0  # Normalize to [0,1]
    
    data_lowlight = torch.from_numpy(data_lowlight).float().to(device)  # Convert to tensor & move to device
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0)  # Adjust dimensions

    # Process image through model
    light_map = U_net(data_lowlight)
    enhanced_image = torch.clamp(data_lowlight / light_map, 0, 1)

    # Generate output file path
    output_image_path = os.path.join(args.output, os.path.basename(image_path))
    output_image_path = output_image_path.replace('.jpg', '.png').replace('.JPG', '.png')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save enhanced image
    torchvision.utils.save_image(enhanced_image, output_image_path)

# --- Main execution --- #
if __name__ == '__main__':
    with torch.no_grad():  # Disable gradient computation for inference
        file_list = os.listdir(args.input)  # Get list of files in input directory
        print("Processing files:", file_list)
  
        for file_name in file_list:
            image_path = os.path.join(args.input, file_name)
            print("Processing:", image_path)
            lowlight(image_path)

