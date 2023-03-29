#Test_On_Image

from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--image_path",default="/content/drive/MyDrive/ESRGAN-Pytorch/images/80_lr.png", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model",default="/content/drive/MyDrive/ESRGAN-Pytorch/saved_models/generator_9.pth", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor)).cpu()

# Save image
fn = opt.image_path.split("/")[-1]
save_image(sr_image, f"images/outputs/sr-{fn}")
print("\n\nSuper Resolution image saved in the outputs folder")

# Calculating PSNR wrt Original Image

lr = cv2.imread(opt.image_path)
sr = cv2.imread(f"images/outputs/sr-{fn}")


# resize the low res image to the same size as the super res image
lr = cv2.resize(lr, sr.shape[:2])

# Convert images to float32 and scale to range [0,1]
lr = lr.astype(np.float32) 
sr = sr.astype(np.float32) 

# Compute MSE
mse = np.mean((sr - lr) ** 2)

# Compute PSNR
psnr = 20 * math.log10(1.0 / math.sqrt(mse))

print("\nPSNR: ",psnr)

# Plot original and super-resolution images side by side
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# fig.suptitle(f"Original vs. Super Resolution\n{opt.image_path}", fontsize=16)
# ax[0].imshow(opt.image_path)
# ax[0].set_title("Original")
# ax[1].imshow(sr_image.permute(1, 2, 0))
# ax[1].set_title("Super Resolution")
# plt.show()
