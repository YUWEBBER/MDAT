import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.datasets.pipelines import to_tensor


class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply

class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight=0.1):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr



class DAClassifer(nn.Module):
    def __init__(self, weight=0.1, in_channels=256, name='da_loss'):
        super(DAClassifer, self).__init__()
        self.weight = weight
        self.grl = GradientScalarLayer(-1.0*weight)
        self.grl_consist = GradientScalarLayer(1.0*weight)
        self.name = name
        # img head
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)


    def forward(self, feat_source, feat_target, weight=False):
        assert len(feat_source) == len(feat_target)
        img_features = []
        bs = feat_source[0].shape[0]
        for i in range(len(feat_source)):
            feat = torch.cat([feat_source[i], feat_target[i]], dim=0)
            img_features.append(feat)

        img_grl_fea = [self.grl(fea) for fea in img_features]
        img_features = []
        for feature in img_grl_fea:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))


        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in img_features:
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[bs:, :] = 1

            da_img_flattened.append(da_img_per_level)  # [N, w*h*c]
            da_img_labels_flattened.append(da_img_label_per_level)  # [N, w*h*c]

        da_img_flattened = torch.cat(da_img_flattened, dim=1)  # [N*l, w*h*c]
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=1)  # [N*l, w*h*c], [0,1]

        da_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )

        if weight:
            da_loss = self.weight * da_loss
        return {self.name: da_loss}
    