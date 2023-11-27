import torch
import torch.nn as nn
from collections import OrderedDict

'''
Class used to extract the features from a trained model.

It use the hooks of pytTorch to wrap the model and the get
the features.
'''
class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layers=[]):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.target_layers = target_layers
        print(target_layers)
        self.features = OrderedDict()

        # Register hooks for target layers
        for name, layer in self.model.named_modules():
            if name in self.target_layers:
                self.features[name] = None
                layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.model(x)
        return self.features

# # Example: Using a pre-trained ResNet model
# target_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

# # Create the feature extractor with the pre-trained ResNet and target layers
# feature_extractor = FeatureExtractor(pretrained_resnet,)

# # Example input (replace this with your own data)
# input_data = torch.randn(1, 3, 224, 224)

# # Forward pass to extract features
# features = feature_extractor(input_data)

# # Access the features of each layer
# for name, feature in zip(target_layers, features):
#     print(f"Layer: {name}, Features shape: {feature.shape}")
