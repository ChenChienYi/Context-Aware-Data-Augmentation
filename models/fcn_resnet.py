
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def initialize_fcn_resnet50_model(num_output_classes, device):
  weights = FCN_ResNet50_Weights.DEFAULT
  model_fcn = fcn_resnet50(weights=weights, progress=False)
  model_fcn = model_fcn.to(device)

  in_features = model_fcn.classifier[4].in_channels

  model_fcn.classifier[4] = nn.Conv2d(in_features, num_output_classes, kernel_size=(1, 1), stride=(1, 1))

  if model_fcn.aux_classifier is not None:
    aux_in_features = model_fcn.aux_classifier[4].in_channels
    model_fcn.aux_classifier[4] = nn.Conv2d(aux_in_features, num_output_classes, kernel_size=(1, 1), stride=(1, 1))

  # freeze all parameters
  for param in model_fcn.parameters():
    param.requires_grad = False

  # Unfreeze the parameters of the classifier head
  for param in model_fcn.classifier.parameters():
    param.requires_grad = True

  # Unfreeze the parameters of the auxiliary classifier
  if model_fcn.aux_classifier is not None:
    for param in model_fcn.aux_classifier.parameters():
        param.requires_grad = True

  return model_fcn