import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121, inception_v3
import timm
import yaml

class ResNet50_Backbone(nn.Module):
    def __init__(self, args, checkpoint_path):
        super().__init__()
        base_model = resnet50(pretrained=False)
        if args.pretrained == 'ImageNet':
            base_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "resnet50-0676ba61.pth")))
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)

class DenseNet121_Backbone(nn.Module):
    def __init__(self, args, checkpoint_path):
        super().__init__()
        base_model = densenet121(pretrained=False)
        if args.pretrained == 'ImageNet':
            base_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "densenet121-a639ec97.pth"), strict=False))
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:1])

    def forward(self, x):
        return self.backbone(x)

class InceptionV3_Backbone(nn.Module):
    def __init__(self, args, checkpoint_path):
        super().__init__()
        base_model = inception_v3(pretrained=False, aux_logits=False)
        if args.pretrained == 'ImageNet':
            base_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "inception_v3_google-0cc3c7bd.pth"), strict=False))
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:19])

    def forward(self, x):
        return self.backbone(x)

class Classifier(nn.Module):
    def __init__(self, model_name, num_class):
        super().__init__()
        self.model_name = model_name
        if self.model_name == 'DenseNet121':
            in_features = 1024
        elif self.model_name == 'ResNet50' or model_name == 'InceptionV3':
            in_features = 2048
        self.drop_out = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features, num_class)

    def forward(self, x):
        if self.model_name == 'DenseNet121':
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.linear(x)
        elif self.model_name == 'ResNet50':
            x = x.view(x.size(0), -1)
            x = self.drop_out(x)
            x = self.linear(x)
        elif self.model_name == 'InceptionV3':
            x = self.avgpool(x)
            x = self.drop_out(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
        return x

def choose_model(args):
    checkpoint_path = './pretrained_weights/2D/'
    if args.model == 'ResNet50':
        backbone = ResNet50_Backbone(args, checkpoint_path)
        if args.pretrained == 'RadImageNet':
            backbone.load_state_dict(torch.load(os.path.join(checkpoint_path, "ResNet50.pt")))
        classifier = Classifier(args.model, num_class=2)
        model = nn.Sequential(backbone, classifier)
        return model, None
    elif args.model == 'DenseNet121':
        backbone = DenseNet121_Backbone(args, checkpoint_path)
        if args.pretrained == 'RadImageNet':
            backbone.load_state_dict(torch.load(os.path.join(checkpoint_path, "DenseNet121.pt")))
        classifier = Classifier(args.model, num_class=2)
        model = nn.Sequential(backbone, classifier)
        return model, None
    elif args.model == 'InceptionV3':
        backbone = InceptionV3_Backbone(args, checkpoint_path)
        if args.pretrained == 'RadImageNet':
            pretrained_weights = torch.load(os.path.join(checkpoint_path, "InceptionV3.pt"))
            backbone.load_state_dict(pretrained_weights)
        classifier = Classifier(args.model, num_class=2)
        model = nn.Sequential(backbone, classifier)
        return model, None
    elif args.model == 'ViT_Large':
        checkpoint_path = os.path.join(checkpoint_path,
                                       'vit_large_patch14_clip_224.openai_ft_in12k_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'vit_large_patch14_clip_224.openai_ft_in12k_in1k/train_args.yaml')
        model = timm.create_model(
            'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights)
        model.head = nn.Linear(model.head.in_features, 2)

        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'BEiTv2_Large':
        checkpoint_path = os.path.join(checkpoint_path, 'beitv2_large_patch16_224.in1k_ft_in22k_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'beitv2_large_patch16_224.in1k_ft_in22k_in1k/train_args.yaml')
        model = timm.create_model(
            'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head = nn.Linear(model.head.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'BEiTv1_Large':
        checkpoint_path = os.path.join(checkpoint_path, 'beit_large_patch16_224.in22k_ft_in22k_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'beit_large_patch16_224.in22k_ft_in22k_in1k/train_args.yaml')
        model = timm.create_model(
            'beit_large_patch16_224.in22k_ft_in22k_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head = nn.Linear(model.head.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'CAFormer_B36':
        checkpoint_path = os.path.join(checkpoint_path, 'caformer_b36.sail_in22k_ft_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'caformer_b36.sail_in22k_ft_in1k/train_args.yaml')
        model = timm.create_model(
            'caformer_b36.sail_in22k_ft_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head.fc.fc2 = nn.Linear(model.head.fc.fc2.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'DeiT3_Large':
        checkpoint_path = os.path.join(checkpoint_path, 'deit3_large_patch16_224.fb_in22k_ft_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'deit3_large_patch16_224.fb_in22k_ft_in1k/train_args.yaml')
        model = timm.create_model(
            'deit3_large_patch16_224.fb_in22k_ft_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head = nn.Linear(model.head.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'ConvFormer_B36':
        checkpoint_path = os.path.join(checkpoint_path, 'convformer_b36.sail_in22k_ft_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'convformer_b36.sail_in22k_ft_in1k/train_args.yaml')
        model = timm.create_model(
            'convformer_b36.sail_in22k_ft_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head.fc.fc2 = nn.Linear(model.head.fc.fc2.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'Swin_Large':
        checkpoint_path = os.path.join(checkpoint_path,
                                       'swin_large_patch4_window7_224.ms_in22k_ft_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'swin_large_patch4_window7_224.ms_in22k_ft_in1k/train_args.yaml')
        model = timm.create_model(
            'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
            pretrained=False,
            # num_classes=2,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head.fc = nn.Linear(model.head.fc.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'VOLO_D4':
        checkpoint_path = os.path.join(checkpoint_path, 'volo_d4_224.sail_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'volo_d4_224.sail_in1k/train_args.yaml')
        model = timm.create_model(
            'volo_d4_224.sail_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head = nn.Linear(model.head.in_features, 2)
        model.aux_head = nn.Linear(model.aux_head.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
    elif args.model == 'TinyViT_21m':
        checkpoint_path = os.path.join(checkpoint_path, 'tiny_vit_21m_224.dist_in22k_ft_in1k/pytorch_model.bin')
        cfg_path = os.path.join(checkpoint_path, 'tiny_vit_21m_224.dist_in22k_ft_in1k/train_args.yaml')
        model = timm.create_model(
            'tiny_vit_21m_224.dist_in22k_ft_in1k',
            pretrained=False,
            num_classes=1000,  # remove classifier nn.Linear
        )
        pretrained_weights = torch.load(checkpoint_path)
        model.load_state_dict(pretrained_weights, strict=False)
        model.head.fc = nn.Linear(model.head.fc.in_features, 2)
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return model, cfg
