import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


class ResNet_base(nn.Module):
    def __init__(self):
        super(ResNet_base, self).__init__()
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.model = resnet_model

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
        x = x.view(x.size(0), x.size(1))
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, drop_rate, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        if drop_rate > 0:
            add_block += [nn.Dropout(p=drop_rate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)


        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        # print(x.shape)
        x = self.add_block(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


class ResNet(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(ResNet, self).__init__()
        self.model_1 = ResNet_base()
        self.model_2 = ResNet_base()
        self.classifier = ClassBlock(2048, class_num, drop_rate)

    def forward(self, x1, x2):
        # print(self.lcm_model)
        # print(self.classifier)

        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        return y1, y2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    net1 = models.resnet50()
    net2 = ResNet_base()
    # net = LCM(201,0.4)
    #net.classifier = nn.Sequential()
    print(net1)
    print("---------------------------")
    print(net2)
    # input = Variable(torch.FloatTensor(8, 3, 256, 256))
    # output,output = net(input,input)
    # print('net output size:')
    # print(output.shape)

model_dict = {
    "resnet" : ResNet
}