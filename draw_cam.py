import cv2
import torch
import numpy as np
import torchvision.models
from torch import nn
from torchvision import models, transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel, GradCAM, GradCAMPlusPlus, EigenGradCAM,EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        model_ft.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 10),
            # nn.LogSoftmax(dim=1)
        )
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x


def reshape_transform(tensor, height=14, width=14):
    print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    print(result.shape)
    return result


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = ft_net()
    model.load_state_dict(torch.load("./data/weight/best.pth"))
    model.eval()
    model.cuda()
    # print(torchvision.models.resnet50().layer4[-1])

    target_layers = [model.model.layer4[-1]]

    # print(target_layers)
    cam = EigenCAM(model=model,
                   target_layers=target_layers,
                   use_cuda=True,
                   # reshape_transform=reshape_transform
                   )

    image_path = "./data/shiwai_shuimao-010.jpg"
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    targets = None

    grayscale_cam = cam(input_tensor=input_tensor,
                        # targets=targets,
                        eigen_smooth=True,
                        aug_smooth=True
                        )
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    save_path = f'./data/cam/' + image_path.split("/")[-1]
    cv2.imwrite(save_path, cam_image)
    print(save_path + " has saved")


