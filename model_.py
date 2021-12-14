import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from efficientnet.model_ import EfficientNet
from Incept.InceptV4 import inceptionv4
from senet.se_resnet import se_resnet50
from senet.cbam_resnet import resnet50_cbam

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


class ResNet(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(ResNet, self).__init__()
        self.model_1 = ResNet_base()
        self.model_2 = ResNet_base()
        self.classifier = ClassBlock(2048, class_num, drop_rate)

    def forward(self, x1, x2):
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


class VGG_base(nn.Module):
    def __init__(self):
        super(VGG_base, self).__init__()
        vgg_model = models.vgg16_bn(pretrained=True)
        vgg_model.avgpool2 = nn.AdaptiveAvgPool2d((7, 7))
        self.model = vgg_model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool2(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1))

        return x


class VGG(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(VGG, self).__init__()
        self.model_1 = VGG_base()
        self.model_2 = VGG_base()
        self.classifier = ClassBlock(512, class_num, drop_rate)

    def forward(self, x1, x2):
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


class DenseNet_base(nn.Module):
    def __init__(self):
        super(DenseNet_base, self).__init__()
        dense_model = models.densenet121(pretrained=True)
        self.model = dense_model

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), x.size(1))
        return x


class DenseNet(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(DenseNet, self).__init__()
        self.model_1 = DenseNet_base()
        self.model_2 = DenseNet_base()
        self.classifier = ClassBlock(1024, class_num, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            # print(x1.size())
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            # print(x2.size())
            y2 = self.classifier(x2)
        return y1, y2

class EfficientNet_base(nn.Module):
    def __init__(self):
        super(EfficientNet_base, self).__init__()
        efficient_net = EfficientNet.from_pretrained('efficientnet-b1')
        self.model = efficient_net

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        return x


class Efficient_Net(nn.Module):
    def __init__(self, classes, drop_rate):
        super(Efficient_Net, self).__init__()
        self.model_1 = EfficientNet_base()
        self.model_2 = EfficientNet_base()
        self.classifier = ClassBlock(1280, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            # print(x1.size())
            x1 = self.model_1(x1)
            # print(x1.size())
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            # print(x2.size())
            x2 = self.model_2(x2)
            # print(x2.size())
            y2 = self.classifier(x2)
        return y1, y2

class Inceptionv4_base(nn.Module):
    def __init__(self):
        super(Inceptionv4_base, self).__init__()
        inception_net = inceptionv4(pretrained=False)
        self.model = inception_net

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        return x

class Inceptionv4(nn.Module):
    def __init__(self, classes, drop_rate):
        super(Inceptionv4, self).__init__()
        self.model_1 = Inceptionv4_base()
        self.model_2 = Inceptionv4_base()
        self.classifier = ClassBlock(1536, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            # print(x1.size())
            x1 = self.model_1(x1)
            # print(x1.size())
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            # print(x2.size())
            x2 = self.model_2(x2)
            # print(x2.size())
            y2 = self.classifier(x2)
        return y1, y2


class seresnet_50_base(nn.Module):
    def __init__(self):
        super(seresnet_50_base, self).__init__()
        se_resnet50_model = se_resnet50(pretrained=True)
        se_resnet50_model.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.model = se_resnet50_model

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

        x = self.model.avgpool2(x)

        x = x.view(x.size(0), x.size(1))
        return x


class seresnet_50(nn.Module):
    def __init__(self, classes, drop_rate):
        super(seresnet_50, self).__init__()
        self.model_1 = seresnet_50_base()
        self.model_2 = seresnet_50_base()
        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            # print(x1.size())
            x1 = self.model_1(x1)
            # print(x1.size())
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            # print(x2.size())
            x2 = self.model_2(x2)
            # print(x2.size())
            y2 = self.classifier(x2)
        return y1, y2


class cbam_resnet50_base(nn.Module):
    def __init__(self):
        super(cbam_resnet50_base, self).__init__()
        cbam_resnet50_model = resnet50_cbam(True)
        cbam_resnet50_model.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.model = cbam_resnet50_model

    def forward(self, x):
        x = self.model.conv(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = self.model.avgpool2(x)

        x = x.view(x.size(0), x.size(1))
        return x


class cbam_resnet_50(nn.Module):
    def __init__(self, classes, drop_rate):
        super(cbam_resnet_50, self).__init__()
        self.model_1 = cbam_resnet50_base()
        self.model_2 = cbam_resnet50_base()
        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            # print(x1.size())
            x1 = self.model_1(x1)
            # print(x1.size())
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            # print(x2.size())
            x2 = self.model_2(x2)
            # print(x2.size())
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
    # import ssl

    # ssl._create_default_https_context = ssl._create_unverified_context
    model = Efficient_Net(100, 0.1)
    # model = EfficientNet_b()
    print(model)
    # print(model.extract_features)
    # Here I left a simple forward function.
    # Test the model, before you train it.
    input = torch.randn(16, 3, 384, 384)
    output, output = model(input, input)
    print(output.size())
    # print(output)

model_dict = {
    "resnet": ResNet,
    "se_resnet": seresnet_50,
    "cbam_resnet": cbam_resnet_50,
    "vgg": VGG,
    "dense": DenseNet,
    "efficient": Efficient_Net,
    "inception": Inceptionv4,
}
