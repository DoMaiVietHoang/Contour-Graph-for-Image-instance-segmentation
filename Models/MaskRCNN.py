import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MaskRCNNBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, trainable_layers=3):
        """
        Initialize the Mask R-CNN backbone
        
        Args:
            backbone_name (str): Name of the backbone network ('resnet50', 'resnet101')
            pretrained (bool): Whether to use pretrained weights
            trainable_layers (int): Number of trainable layers from the end
        """
        super(MaskRCNNBackbone, self).__init__()
        
        # Create the backbone with FPN
        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            trainable_layers=trainable_layers
        )
        
        # Get the output channels from the backbone
        self.out_channels = self.backbone.out_channels
        
    def forward(self, x):
        """
        Forward pass through the backbone
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            dict: Dictionary containing feature maps at different scales
        """
        return self.backbone(x)

def get_backbone(backbone_name='resnet50', pretrained=True):
    """
    Helper function to create a backbone instance
    
    Args:
        backbone_name (str): Name of the backbone network
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        MaskRCNNBackbone: Backbone instance
    """
    return MaskRCNNBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained
    )
