import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


def init_classifier_bias(module: nn.Linear, pi: float = 0.01) -> None:
    """Initialize classifier bias using a prior probability.
    Set the bias such that the initial output probability for the
    positive class is approximately ``pi`` when logits are passed through
    a sigmoid:

        b = -log((1 - pi) / pi)
    Args:
        module: Linear classification layer whose bias will be initialized.
        pi: Prior probability for the positive class (default: 0.01).
    """
    if module is None or module.bias is None:
        return

    with torch.no_grad():
        bias_value = -math.log((1.0 - pi) / pi)
        module.bias.data.fill_(bias_value)

class SEBlock(nn.Module):
    """
    Squeeze and Excitation Block
    """
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully connected layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Element-wise multiplication
        return x * y.expand_as(x)

    
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for spatial features
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: Input dimension (number of channels)
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in qkv projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            (B, C, H, W) attended feature map
        """
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, N, C)
        x = x.flatten(2).transpose(1, 2) # (B, C, H*W) -> (B, H*W, C)

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        # Attention: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class ResNet18WithSE(nn.Module):
    """
    ResNet18 with Squeeze-and-Excitation blocks
    """
    def __init__(self, num_classes=3, pretrained=True, se_reduction=16):
        super(ResNet18WithSE, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Add SE blocks after each residual block
        self.se1 = SEBlock(64, se_reduction)
        self.se2 = SEBlock(128, se_reduction)
        self.se3 = SEBlock(256, se_reduction)
        self.se4 = SEBlock(512, se_reduction)

        # Replace classifier
        self.backbone.fc = nn.Linear(512, num_classes)
        # Initialize classifier bias using prior pi
        init_classifier_bias(self.backbone.fc, pi=0.01)

        # Store layer references for SE insertion
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

    def forward(self, x):
        # Initial convolution
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 + SE
        x = self.layer1(x)
        x = self.se1(x)

        # Layer 2 + SE
        x = self.layer2(x)
        x = self.se2(x)

        # Layer 3 + SE
        x = self.layer3(x)
        x = self.se3(x)

        # Layer 4 + SE
        x = self.layer4(x)
        x = self.se4(x)

        # Final pooling and classification
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x


class ResNet18WithMHA(nn.Module):
    """
    ResNet18 with Multi-Head Attention
    """
    def __init__(self, num_classes=3,pretrained=True, num_heads=8):
        super(ResNet18WithMHA, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Add MHA after layer4 (before polling)
        self.mha = MultiHeadSelfAttention(dim=512, num_heads=num_heads)

        # Replace classifier
        self.backbone.fc = nn.Linear(512, num_classes)
        # Initialize classifier bias using prior pi
        init_classifier_bias(self.backbone.fc, pi=0.01)

    def forward(self, x):
        # Backbone forward pass up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply Multi-Head Attention
        x = self.mha(x)
        
        # Final pooling and classification
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x


class EfficientNetWithSE(nn.Module):
    """
    EfficientNet-B0 with additional SE blocks
    """
    def __init__(self, num_classes=3, pretrained=True, se_reduction=16):
        super(EfficientNetWithSE, self).__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Add SE blocks before classifier
        self.se_final = SEBlock(1280, se_reduction)

        # Replace classifier
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
        # Initialize classifier bias using prior pi
        init_classifier_bias(self.backbone.classifier[1], pi=0.01)

    def forward(self, x):
        # Extract features
        x = self.backbone.features(x)

        # Apply additional SE
        x = self.se_final(x)

        # Global pooling and classification
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        return x


class EfficientNetWithMHA(nn.Module):
    """
    EfficientNet-B0 with Multi-Head Attention
    """
    def __init__(self, num_classes=3, pretrained=True, num_heads=8):
        super(EfficientNetWithMHA, self).__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Add MHA before final pooling
        self.mha = MultiHeadSelfAttention(dim=1280, num_heads=num_heads)

        # Replace classifier
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
        # Initialize classifier bias using prior pi
        init_classifier_bias(self.backbone.classifier[1], pi=0.01)

    def forward(self, x):
        # Extract features
        x = self.backbone.features(x)

        # Apply Multi-Head Attention
        x = self.mha(x)

        # Global pooling and classification
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)

        return x


def build_model(backbone='resnet18', num_classes=3, pretrained=True, attention='none', **kwargs):
    """
    Factory function to build models with different configurations

    Args:
        backbone: 'resnet18' | 'efficientnet'
        num_classes: Number of output classes (default: 3)
        pretrained: Use ImageNet pretrained weights
        attention: 'none' | 'se' | 'mha'
        **kwargs: Additional arguments (se_reduction, num_heads, etc.)

    Returns:
        model: PyTorch model
    """
    if backbone == 'resnet18':
        if attention == 'se':
            model = ResNet18WithSE(num_classes, pretrained, 
                                   se_reduction=kwargs.get('se_reduction', 16))
        elif attention == 'mha':
            model = ResNet18WithMHA(num_classes, pretrained,
                                    num_heads=kwargs.get('num_heads', 8))
        else:
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            # Initialize classifier bias using prior pi
            init_classifier_bias(model.fc, pi=0.01)
            
    elif backbone == 'efficientnet':
        if attention == 'se':
            model = EfficientNetWithSE(num_classes, pretrained,
                                       se_reduction=kwargs.get('se_reduction', 16))
        elif attention == 'mha':
            model = EfficientNetWithMHA(num_classes, pretrained,
                                        num_heads=kwargs.get('num_heads', 8))
        else:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            # Initialize classifier bias using prior pi
            init_classifier_bias(model.classifier[1], pi=0.01)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Supported: resnet18, efficientnet")
    
    return model 


def load_pretrained_backbone(model, pretrained_path, backbone_key=None):
    """Load pretrained weights for a given backbone.
    Args:
        model: PyTorch model whose weights will be updated.
        pretrained_path: Filesystem path to the ``.pt`` checkpoint.
        backbone_key: Optional name/key of the backbone used in
            ``PRETRAINED_BACKBONES`` (e.g. ``"resnet18"``,
            ``"efficientnet"``). Used for logging only.
    """
    
    msg = f"Loading pretrained weights from: {pretrained_path}"
    if backbone_key is not None:
        msg = f"Loading pretrained weights for backbone '{backbone_key}' from: {pretrained_path}"

    checkpoint_state = torch.load(pretrained_path, map_location="cpu")

    # Some checkpoints may be stored under a top-level 'state_dict' key.
    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        state_dict = checkpoint_state["state_dict"]
    else:
        state_dict = checkpoint_state

    model_state = model.state_dict()
    adapted_state = {}

    for k, v in state_dict.items():
        # plain backbones: resnet18 / efficientnet
        if k in model_state and model_state[k].shape == v.shape:
            adapted_state[k] = v
            continue

        # 2) Match with 'backbone.' prefix for wrapper models
        prefixed = f"backbone.{k}"
        if prefixed in model_state and model_state[prefixed].shape == v.shape:
            adapted_state[prefixed] = v
            continue

    missing, unexpected = model.load_state_dict(adapted_state, strict=False)

    print(f"Loaded {len(adapted_state)} parameters from checkpoint.")
    if missing:
        print(f"Missing keys (not loaded) count: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys in checkpoint count: {len(unexpected)}")

    print("Pretrained backbone weights loaded successfully.")

    return model
