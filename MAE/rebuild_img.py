import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from vit_pytorch import ViT
from mae_model import MAE
from vit_pytorch.deepvit import DeepViT
from einops.layers.torch import Rearrange
if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda:0")

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048
)

v.load_state_dict(torch.load('./weights/trained-vit.pt'))

mae = MAE(
    encoder=v,
    masking_ratio=0.75,   # the paper recommended 75% masked patches
    decoder_dim=512,      # paper showed good results with just 512
    decoder_depth=6       # anywhere from 1 to 8
)
mae = mae.eval()
mae = mae.cuda()

img1 = cv2.imread("1.JPEG")
img2 = cv2.imread("2.JPEG")
img1 = cv2.resize(img1, [256, 256])
img2 = cv2.resize(img2, [256, 256])

images = torch.from_numpy(img1).view(1, 3, 256, 256).float().to(device)
img = mae(images)


plt.figure(0), plt.imshow(img), plt.show()






