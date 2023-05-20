import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

import pytorch_utils.callbacks as pt_callbacks
import pytorch_utils.training_utils as pt_train
import source.data_handling as data_handling
import source.config as cf
import source.utils as utils


class CustomModelBase(pt_train.CustomModelBase):
    """
    ModelBase override for training and validation steps
    """
    def __init__(self, class_weights):
        super(CustomModelBase, self).__init__()
        self.class_weights = class_weights

    def training_step(self, batch):
        # print("batch: ", len(batch))
        images, labels = batch

        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels, weight=self.class_weights)  # Calculate loss with class weights
        acc = pt_train._accuracy(out, labels)  # Calculate accuracy
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch

        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels, weight=None)  # We are not using class weights for validation since data is balanced in validation set
        acc = pt_train._accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}


# 57M parameters
class AlexNet(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(AlexNet, self).__init__(class_weights=class_weights)
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 11M paramters
class ResNet18(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ResNet18, self).__init__(class_weights=class_weights)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 21M paramters
class ResNet34(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ResNet34, self).__init__(class_weights=class_weights)
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class ResNet50(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ResNet50, self).__init__(class_weights=class_weights)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 42M paramters
class ResNet101(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ResNet101, self).__init__(class_weights=class_weights)
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class ResNet152(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ResNet152, self).__init__(class_weights=class_weights)
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 6.6M paramters
# InceptionNet v1 - GoogLeNet
class InceptionNet(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(InceptionNet, self).__init__(class_weights=class_weights)
        self.model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 4M parameters
class EfficientNetB0(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB0, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 6M parameters
class EfficientNetB1(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB1, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 7.7M parameters
class EfficientNetB2(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB2, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNetB3(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB3, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNetB4(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB4, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class EfficientNetB5(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetB5, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


# 20M paramters
class EfficientNetV2_S(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(EfficientNetV2_S, self).__init__(class_weights=class_weights)
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 27M parameters
class ConvNext_T(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ConvNext_T, self).__init__(class_weights=class_weights)
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 1.5M paramters
class MobileNet_V3_Small(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(MobileNet_V3_Small, self).__init__(class_weights=class_weights)
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# 4.2M paramters
class MobileNet_V3_Large(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(MobileNet_V3_Large, self).__init__(class_weights=class_weights)
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

# takes 224, 224 input shape
class ViT_B_16(CustomModelBase):
    def __init__(self, num_classes=2, class_weights=None):
        super(ViT_B_16, self).__init__(class_weights=class_weights)
        self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        self.model.heads = nn.Linear(self.model.heads[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


models_dict = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'inception': InceptionNet,
    'eff_b0': EfficientNetB0,
    'eff_b1': EfficientNetB1,
    'eff_b2': EfficientNetB2,
    'eff_b3': EfficientNetB3,
    'eff_b4': EfficientNetB4,
    'eff_b5': EfficientNetB5,
    'eff_v2_s': EfficientNetV2_S,
    'convnext_t': ConvNext_T,
    'mobilenet_v3_small': MobileNet_V3_Small,
    'mobilenet_v3_large': MobileNet_V3_Large,
    'vit_b_16': ViT_B_16
}
