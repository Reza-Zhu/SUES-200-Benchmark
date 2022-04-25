import timm
import math
import torch
import torch.nn as nn
from torch.nn import init, functional
from senet.cbam_resnet import resnet50_cbam


def forward_(model_1, model_2, classifier, x1, x2):
    if x1 is None:
        y1 = None
    else:
        x1 = model_1(x1)
        y1 = classifier(x1)

    if x2 is None:
        y2 = None
    else:
        x2 = model_2(x2)
        y2 = classifier(x2)

    return y1, y2


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
    def __init__(self, class_num, drop_rate, share_weight=False, pretrained=True):
        super(ResNet, self).__init__()
        self.model_1 = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)

        self.classifier = ClassBlock(2048, class_num, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class SEResNet_50(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(SEResNet_50, self).__init__()
        self.model_1 = timm.create_model("seresnet50", pretrained=pretrained, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("seresnet50", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class ResNeSt_50(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(ResNeSt_50, self).__init__()
        self.model_1 = timm.create_model("resnest50d", pretrained=pretrained, num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("resnest50d", pretrained=pretrained, num_classes=0)

        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class cbam_resnet50_base(nn.Module):
    def __init__(self, pretrained=True):
        super(cbam_resnet50_base, self).__init__()
        cbam_resnet50_model = resnet50_cbam(pretrained=pretrained)
        cbam_resnet50_model.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.model = cbam_resnet50_model

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


class CBAM_ResNet_50(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(CBAM_ResNet_50, self).__init__()
        self.model_1 = cbam_resnet50_base(pretrained)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = cbam_resnet50_base(pretrained)
        self.classifier = ClassBlock(2048, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class VGG(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=False, pretrained=True):
        super(VGG, self).__init__()
        self.model_1 = timm.create_model("vgg16_bn", pretrained=pretrained, num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("vgg16_bn", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(4096, class_num, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class DenseNet(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=False, pretrained=True):
        super(DenseNet, self).__init__()
        self.model_1 = timm.create_model("densenet201", pretrained=pretrained, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("densenet201", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(1920, class_num, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class EfficientV1(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(EfficientV1, self).__init__()
        self.model_1 = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(1792, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class EfficientV2(nn.Module):
    def __init__(self, classes, drop_rate,  share_weight=False):
        super(EfficientV2, self).__init__()
        self.model_1 = timm.create_model("efficientnetv2_s", num_classes=0)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("efficientnetv2_s", num_classes=0)
        self.classifier = ClassBlock(1280, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class Inceptionv4(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(Inceptionv4, self).__init__()
        self.model_1 = timm.create_model("inception_v4", pretrained=pretrained, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("inception_v4", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(1536, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class ViT(nn.Module):
    def __init__(self, classes, drop_rate, share_weight=False, pretrained=True):
        super(ViT, self).__init__()
        self.model_1 = timm.create_model("vit_base_patch16_384", pretrained=pretrained, num_classes=0)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model("vit_base_patch16_384", pretrained=pretrained, num_classes=0)
        self.classifier = ClassBlock(768, classes, drop_rate)

    def forward(self, x1, x2):
        return forward_(self.model_1, self.model_2, self.classifier, x1, x2)



class base_LPN(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', block=4,
                 pretrained = True):
        super(base_LPN, self).__init__()
        model_ft = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        self.model = model_ft
        self.model.relu = nn.ReLU(inplace=True)
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.shape)
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)

        return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block), W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block), W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                # print("x", x.shape)
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                # print("x_curr", x_curr.shape)
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    x_pad = functional.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                # print("x_curr", x_curr.shape)
                avgpool = pooling(x_curr)
                # print("pool", avgpool.shape)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = functional.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = functional.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)


class LPN(nn.Module):
    def __init__(self, class_num, droprate, stride=1, pool='avg', share_weight=False, block=4,
                 pretrained=True):
        super(LPN, self).__init__()
        # self.LPN = LPN
        self.block = block
        self.model_1 = base_LPN(class_num, stride=stride, pool=pool, block=block, pretrained=pretrained)
        # self.model_2 = ft_net_LPN(class_num, stride=stride, pool=pool, block=block)

        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = base_LPN(class_num, stride=stride, pool=pool, block=block, pretrained=pretrained)

        if pool == 'avg+max':
            for i in range(self.block):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(4096, class_num, droprate))
        else:
            for i in range(self.block):
                name = 'classifier'+str(i)
                setattr(self, name, ClassBlock(2048, class_num, droprate))

    def forward(self, x1, x2):  # x4 is extra data
        return forward_(self.model_1, self.model_2, self.part_classifier, x1, x2)


    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            # print(c)
            predict[i] = c(part[i])
            # print(predict[i].shape)
        # print(predict)
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


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
    model = ResNet(100, 0.1).cuda()
    # model = EfficientNet_b()
    print(model.device)
    # print(model.extract_features)
    # Here I left a simple forward function.
    # Test the model, before you train it.
    input = torch.randn(16, 3, 384, 384).cuda()
    output, output = model(input, input)
    print(output.size())
    # print(output)

model_dict = {
    "LPN": LPN,
    "vgg": VGG,
    "resnet": ResNet,
    "seresnet": SEResNet_50,
    "resnest": ResNeSt_50,
    "cbamresnet": CBAM_ResNet_50,
    "dense": DenseNet,
    "efficientv1": EfficientV1,
    "efficientv2": EfficientV2,
    "inception": Inceptionv4,
    "vit": ViT,
}
