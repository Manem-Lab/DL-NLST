import os
import torch
from torch import nn
from monai.networks.nets import ResNet
from wama_modules.utils import load_weights
from wama_modules.thirdparty_lib.MedicalNet_Tencent.model import generate_model
from wama_modules.thirdparty_lib.Efficient3D_okankop.models import shufflenet, shufflenetv2, resnext
from wama_modules.thirdparty_lib.Efficient3D_okankop.models.resnet import resnet50, resnet101, resnet18
import wama_modules.thirdparty_lib.Efficient3D_okankop.models.squeezenet as squeezenet
import wama_modules.thirdparty_lib.Efficient3D_okankop.models.mobilenet as mobilenet
import wama_modules.thirdparty_lib.Efficient3D_okankop.models.mobilenetv2 as mobilenetv2
from wama_modules.thirdparty_lib.ResNets3D_kenshohara import resnet2p1d

class ResNet(nn.Module):
    def __init__(self, depth=50, pretrained='3DSeg_8', checkpoint_path='./pretrain_weights/'):
        super().__init__()
        block_inplanes = [64, 128, 256, 512]
        if pretrained == 'nnUNet' or pretrained == '3DSeg_8':
            # encoder
            self.encoder = generate_model(depth)
            if depth == 18:
                if pretrained == 'nnUNet':
                    checkpoint_path = os.path.join(checkpoint_path, 'resnet_18_23dataset.pth')
                elif pretrained == '3DSeg_8':
                    checkpoint_path = os.path.join(checkpoint_path, 'resnet_18.pth')
            elif depth == 50:
                if pretrained == 'nnUNet':
                    checkpoint_path = os.path.join(checkpoint_path, 'resnet_50_23dataset.pth')
                elif pretrained == '3DSeg_8':
                    checkpoint_path = os.path.join(checkpoint_path, 'resnet_50.pth')
            elif depth == 101:
                checkpoint_path = os.path.join(checkpoint_path, 'resnet_101.pth')
            pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
            # cls head
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
            if depth != 18:
                self.fc = nn.Linear(4 * block_inplanes[3], 2)
            else:
                self.fc = nn.Linear(block_inplanes[3], 2)
        elif pretrained == 'Kinetics':
            if depth == 18:
                checkpoint_path = os.path.join(checkpoint_path, 'kinetics_resnet_18_RGB_16_best.pth')
                self.encoder = resnet18()
            elif depth == 50:
                checkpoint_path = os.path.join(checkpoint_path, 'kinetics_resnet_50_RGB_16_best.pth')
                self.encoder = resnet50()
            elif depth == 101:
                checkpoint_path = os.path.join(checkpoint_path, 'kinetics_resnet_101_RGB_16_best.pth')
                self.encoder = resnet101()
            pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
            # cls head
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
            if depth != 18:
                self.fc = nn.Linear(4 * block_inplanes[3], 2)
            else:
                self.fc = nn.Linear(block_inplanes[3], 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class ShuffleNet(nn.Module):
    def __init__(self, version='v1', checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        if version == 'v1':
            self.encoder = shufflenet.get_model(groups=3, width_mult=2)
            checkpoint_path = os.path.join(checkpoint_path, 'kinetics_shufflenet_2.0x_G3_RGB_16_best.pth')
            self.fc = nn.Linear(1920, 2)
        elif version == 'v2':
            self.encoder = shufflenetv2.get_model(width_mult=2)
            checkpoint_path = os.path.join(checkpoint_path, 'kinetics_shufflenetv2_2.0x_RGB_16_best.pth')
            self.fc = nn.Linear(976, 2)
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class ResNeXt101(nn.Module):
    def __init__(self, checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        block_inplanes = [64, 128, 256, 512]
        self.encoder = resnext.resnext101()
        checkpoint_path = os.path.join(checkpoint_path, 'kinetics_resnext_101_RGB_16_best.pth')
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4 * block_inplanes[3], 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class SqueezeNet(nn.Module):
    def __init__(self, checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        self.encoder = squeezenet.get_model()
        checkpoint_path = os.path.join(checkpoint_path, 'kinetics_squeezenet_RGB_16_best.pth')
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class MobileNetv1(nn.Module):
    def __init__(self, checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        self.encoder = mobilenet.get_model(width_mult = 2.)
        checkpoint_path = os.path.join(checkpoint_path, 'kinetics_mobilenet_2.0x_RGB_16_best.pth')
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class MobileNetv2(nn.Module):
    def __init__(self, checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        self.encoder = mobilenetv2.get_model(width_mult = 1.)
        checkpoint_path = os.path.join(checkpoint_path, 'kinetics_mobilenetv2_1.0x_RGB_16_best.pth')
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

class R2Plus1D(nn.Module):
    def __init__(self, checkpoint_path='./pretrain_weights/'):
        super().__init__()
        # encoder
        self.encoder = resnet2p1d.generate_model(50)
        checkpoint_path = os.path.join(checkpoint_path, 'r2p1d50_K_200ep.pth')
        pretrain_weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        load_weights(self.encoder, pretrain_weights, drop_modelDOT=True)
        # cls head
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


def choose_model(args):
    checkpoint_path = './pretrained_weights/3D/'
    if args.model == 'ShuffleNetv1':
        return ShuffleNet('v1', checkpoint_path)
    elif args.model == 'ShuffleNetv2':
        return ShuffleNet('v2', checkpoint_path)
    elif args.model == 'ResNet18':
        return ResNet(depth=18, pretrained=args.pretrained, checkpoint_path=checkpoint_path)
    elif args.model == 'ResNet50':
        return ResNet(depth=50, pretrained=args.pretrained, checkpoint_path=checkpoint_path)
    elif args.model == 'ResNet101':
        return ResNet(depth=101, pretrained=args.pretrained, checkpoint_path=checkpoint_path)
    elif args.model == 'ResNeXt101':
        return ResNeXt101(checkpoint_path)
    elif args.model == 'SqueezeNet':
        return SqueezeNet(checkpoint_path)
    elif args.model == 'MobileNetv1':
        return MobileNetv1(checkpoint_path)
    elif args.model == 'MobileNetv2':
        return MobileNetv2(checkpoint_path)
    elif args.model == 'R2Plus1D':
        return R2Plus1D(checkpoint_path)