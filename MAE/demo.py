import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from vit_pytorch import ViT, MAE
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

mae = MAE(
    encoder=v,
    masking_ratio=0.75,   # the paper recommended 75% masked patches
    decoder_dim=512,      # paper showed good results with just 512
    decoder_depth=6       # anywhere from 1 to 8
)
mae = mae.cuda()
img = torch.randn(1, 3, 256, 256)
img = img.cuda()
print(img.device)
preds = v(img)# (1, 1000)

path = 'img_dir/0.jpg'
# print(mae.encoder.to_patch_embedding[:])
to_patch, patch_to_emb = mae.encoder.to_patch_embedding[:2]

# Sequential(
#   (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
#   (1): Linear(in_features=3072, out_features=1024, bias=True)
# )

img = cv2.imread(path)
print(img.shape)
img = cv2.resize(img, [256, 256])

images = torch.from_numpy(img).view(1, 3, 256, 256).float()
print(images.shape)

patches = mae.to_patch(images)
batch, num_patches, *_ = patches.shape
print(patches.shape)

tokens = mae.patch_to_emb(patches)
print(tokens.shape)

tokens = tokens + mae.encoder.pos_embedding[:, 1:(num_patches + 1)]

num_masked = int(mae.masking_ratio * num_patches)
rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)

masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

batch_range = torch.arange(batch, device=device)[:, None]
tokens = tokens[batch_range, unmasked_indices].to(device)
masked_patches = patches[batch_range, masked_indices].to(device)
print(masked_patches.device)
print("token", tokens.device)
tokens = tokens.to('cpu')
# attend with vision transformer
encoded_tokens = mae.encoder.transformer(tokens)
print(encoded_tokens.shape)
print(encoded_tokens.device)

decoder_tokens = mae.enc_to_dec(encoded_tokens)
print(decoder_tokens.shape)
print(decoder_tokens.device)

mask_tokens = repeat(mae.mask_token.to(device), 'd -> b n d', b=batch, n=num_masked)
mask_tokens = mask_tokens.to('cpu')
# print(masked_indices.device)
masked_indices = masked_indices.to('cpu')
mask_tokens = mask_tokens + mae.decoder_pos_emb(masked_indices)
print(mask_tokens.shape)


decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
decoded_tokens = mae.decoder(decoder_tokens)
print("-----")
print(decoded_tokens.shape)

mask_tokens = decoded_tokens[:, -num_masked:]
print(mask_tokens.shape)

pred_pixel_values = mae.to_pixels(decoded_tokens)
print(pred_pixel_values.shape)
layer = torch.nn.Sequential(
    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=32, p2=32, h=8, w=8),
)

img = layer(pred_pixel_values).view(256, 256, 3)
img = img.detach().numpy()
print(img.shape)
plt.figure(0), plt.imshow(img), plt.show()

# print(img.shape)
# loss = mae(images)
# loss.backward()

# that's all!
# do the above in a for loop many times with a lot of images and your vision transformer will learn

# save your improved vision transformer
# torch.save(v.state_dict(), './trained-vit.pt')
# print(loss)
