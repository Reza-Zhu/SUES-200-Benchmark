import timm
import torch
import torch.nn as nn
from torch.nn import init
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
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(ResNet, self).__init__()
        self.model_1 = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("resnet50", pretrained=True, num_classes=0)
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


class SEResNet_50(nn.Module):
    def __init__(self, classes, drop_rate):
        super(SEResNet_50, self).__init__()
        self.model_1 = timm.create_model("seresnet50", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("seresnet50", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(2048, classes, drop_rate)

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


class ResNeSt_50(nn.Module):
    def __init__(self, classes, drop_rate):
        super(ResNeSt_50, self).__init__()
        self.model_1 = timm.create_model("resnest50d", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("resnest50d", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(2048, classes, drop_rate)

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


class CBAM_ResNet_50(nn.Module):
    def __init__(self, classes, drop_rate):
        super(CBAM_ResNet_50, self).__init__()
        self.model_1 = cbam_resnet50_base()
        self.model_2 = cbam_resnet50_base()
        self.classifier = ClassBlock(2048, classes, drop_rate)

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


class VGG(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(VGG, self).__init__()
        self.model_1 = timm.create_model("vgg16_bn", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("vgg16_bn", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(4096, class_num, drop_rate)

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


class DenseNet(nn.Module):
    def __init__(self, class_num, drop_rate):
        super(DenseNet, self).__init__()
        self.model_1 = timm.create_model("densenet121", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("densenet121", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(1024, class_num, drop_rate)

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


class Efficient_Net(nn.Module):
    def __init__(self, classes, drop_rate):
        super(Efficient_Net, self).__init__()
        self.model_1 = timm.create_model("efficientnet_b1", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("efficientnet_b1", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(1280, classes, drop_rate)

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


class Inceptionv4(nn.Module):
    def __init__(self, classes, drop_rate):
        super(Inceptionv4, self).__init__()
        self.model_1 = timm.create_model("inception_v4", pretrained=True, num_classes=0)
        self.model_2 = timm.create_model("inception_v4", pretrained=True, num_classes=0)
        self.classifier = ClassBlock(1536, classes, drop_rate)

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
    "se_resnet": SEResNet_50,
    "resnest_50": ResNeSt_50,
    "cbam_resnet": CBAM_ResNet_50,
    "vgg": VGG,
    "dense": DenseNet,
    "efficient": Efficient_Net,
    "inception": Inceptionv4,
}
