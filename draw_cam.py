import cv2
import os
import torch
import model_
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models
from utils import get_best_weight
from torch import nn
from torchvision import models, transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel, GradCAM, GradCAMPlusPlus, EigenGradCAM, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def reshape_transform(tensor, height=24, width=24):
    # print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # print(result.shape)
    # result = rearrange(result, "b (h w) y -> b y h w", h=24, w=24)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    # print(result.shape)

    return result

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    query_name = "query_drone"
    model_name = "vit"
    pic_name = "40"
    heights = [150, 200, 250, 300]
    csv_path = "/home/sues/media/disk2/save_model_weight"
    for height in heights:

        net_path = get_best_weight(query_name, model_name, height, csv_path)
        model = model_.model_dict[model_name](120, 0)
        model.load_state_dict(torch.load(net_path))
        target_layers = None
        end_name = None
        if "satellite" in query_name:
            model = model.model_1
            target_layers = [model.blocks[-1].norm1]
            height = "satellite"
            end_name = ".png"

        elif "drone" in query_name:
            model = model.model_2
            target_layers = [model.blocks[-1].norm1]
            end_name = ".jpg"

        image_path = os.path.join("./Heat maps",
                                  str(height), pic_name + end_name)
        print(image_path)
        save_path = os.path.join(f"./Heat maps", str(height), model_name + "_heat_" + image_path.split("/")[-1])
        print(save_path)
        model.eval()
        model.cuda()
        cam = EigenCAM(model=model,
                       target_layers=target_layers,
                       reshape_transform=reshape_transform,
                       use_cuda=True,
                       )

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (384, 384))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        targets = [ClassifierOutputTarget(100)]

        grayscale_cam = cam(input_tensor=input_tensor,
                            eigen_smooth=True,
                            aug_smooth=True
                            )
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)
        cv2.imwrite(save_path, cam_image)
        x = plt.imread(save_path)
        plt.imshow(x)
        plt.show()
        print(save_path + " has saved")
        if "satellite" in query_name:
            break

