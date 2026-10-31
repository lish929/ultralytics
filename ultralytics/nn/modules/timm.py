# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adapter that turns a timm model into a multi-scale Ultralytics backbone."""

import torch
import torch.nn as nn

from ultralytics.utils.checks import check_requirements


class TimmBackbone(nn.Module):
    """Multi-scale backbone built on a timm feature extractor.

    The timm trunk is created with `features_only=True`, so it returns a list of feature maps instead of logits. The
    requested `strides` select which maps to keep and a 1x1 convolution projects each one to its declared channel
    count, giving the YOLO neck the channel contract declared in the model YAML.

    Args:
        model_name (str): timm model name, e.g. 'resnet18' or 'convnext_tiny'.
        out_channels (int | list[int]): Output channels per returned level, or one int shared by all levels.
        pretrained (bool): Load timm pretrained weights.
        strides (tuple[int, ...]): Strides to return, default (8, 16, 32) for the P3/P4/P5 pyramid.

    Attributes:
        backbone (nn.Module): timm feature extractor returning one map per stage.
        indices (list[int]): Position of each requested stride in the trunk output list, e.g. [2, 3, 4].
        projections (nn.ModuleList): 1x1 convolutions mapping each stage to its declared channel count.
    """

    def __init__(self, model_name="resnet18", out_channels=256, pretrained=False, strides=(8, 16, 32)):
        """Build the timm trunk and the per-level projection layers.

        Args:
            model_name (str): timm model name, e.g. 'resnet18'.
            out_channels (int | list[int]): Output channels per level, or one int shared by all levels.
            pretrained (bool): Load timm pretrained weights, which needs network access on first use.
            strides (tuple[int, ...]): Strides to return, each one must exist in the model's feature pyramid.

        Raises:
            ValueError: If the timm model does not provide one of the requested `strides`.
        """
        super(TimmBackbone, self).__init__()

        check_requirements("timm")  # optional dependency, installed on first use
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        # per-stage stride and width reported by the trunk, both ordered like its forward() output
        reductions, channels = self.backbone.feature_info.reduction(), self.backbone.feature_info.channels()

        if [stride for stride in strides if stride not in reductions]:
            raise ValueError(f"{model_name} provides strides {reductions} but needs strides {strides}")

        # positions of the requested strides in the trunk output list, e.g. resnet18 (8, 16, 32) -> [2, 3, 4]
        self.indices = [reductions.index(stride) for stride in strides]
        # one int means "same width for every level", a list gives per-level widths such as [256, 512, 1024]
        out_channels = (out_channels,) * len(self.indices) if isinstance(out_channels, int) else tuple(out_channels)

        # 1x1 projection: decouples the native timm widths from the channel counts declared in the model YAML
        self.projections = nn.ModuleList(
            nn.Conv2d(channels[index], channel, 1) for index, channel in zip(self.indices, out_channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the trunk and return one projected feature map per requested stride.

        Args:
            x (torch.Tensor): Input images with shape (B, 3, H, W).

        Returns:
            (list[torch.Tensor]): Feature maps ordered as `strides`, i.e. P3/8, P4/16 and P5/32 by default.
        """
        features = self.backbone(x)
        return [projection(features[index]) for projection, index in zip(self.projections, self.indices)]
