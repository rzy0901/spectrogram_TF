import torch
import torch.nn as nn
import torch.nn.functional as nf

class fc_part(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(512,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120,3)

    def forward(self, x):
        x = nf.relu(self.fc1(x))
        x = nf.relu(self.fc2(x))
        x = nf.relu(self.fc3(x))
        # x = self.fc1(x)
        return x

class FeatureExtractorResNet(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractorResNet, self).__init__()
        self.conv_features = None

        for name, layer in original_model.named_children():
            if isinstance(layer, nn.Linear):
                break
            setattr(self, name, layer)
            if isinstance(layer, nn.AdaptiveAvgPool2d): 
                self.conv_features = None

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                self.conv_features = x

        features = self.conv_features.view(self.conv_features.size(0), -1)
        return features

class FeatureExtractorResNet18(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractorResNet18, self).__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class ClassifierResNet(nn.Module):
    def __init__(self, original_model):
        super(ClassifierResNet, self).__init__()
        self.fc = original_model.fc

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from torchvision import models
    MODEL_FILE = './model1/best_resnet18_model.pth' # pretrained model using simulated dataset
    model = models.resnet18(pretrained=True)
    model.fc = fc_part()
    model.load_state_dict(torch.load(MODEL_FILE))
    feature_extractor_model = FeatureExtractorResNet18(model)
    classifier_model = ClassifierResNet(model)
    print(feature_extractor_model)
    print(classifier_model)

    model2 = models.resnet34(pretrained=True)
    print(model2)
    feature_extractor_model2 = FeatureExtractorResNet(model2)
    classifier_model2 = ClassifierResNet(model2)
    print(feature_extractor_model2)
    print(classifier_model2)

