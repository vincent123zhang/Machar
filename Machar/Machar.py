import os
import copy
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from model.fdblock import FDBlock, Conv2d_BN, RepDW, FFN

# Machar model configurations
Machar_width = {
    'S': [32, 48],
    'B': [48, 96, 192, 384],
    'L': [64, 128, 384, 512],
}

Machar_depth = {
    'S': [2, 1],
    'B': [4, 3, 10, 5],
    'L': [4, 4, 12, 6],
}


def stem(in_chs, out_chs):
    """
    Stem Layer with two 7x1 convolutions for temporal feature extraction.
    Output: sequence of layers with final shape of [B, C, H/4, W]
    """
    return nn.Sequential(
        Conv2d_BN(in_chs, out_chs // 2, (7, 1), (2, 1), (3, 0)), 
        nn.GELU(),
        Conv2d_BN(out_chs // 2, out_chs, (7, 1), (2, 1), (3, 0)),
        nn.GELU(),
    )


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer implemented with convolution.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=2, padding=0,
                 in_chans=3, embed_dim=48):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = Conv2d_BN(in_chans, embed_dim, patch_size, stride, padding)

    def forward(self, x):
        return self.proj(x)


class Conv2DBlock(nn.Module):
    """
    Conv2D Block with RepDW-3 and FFN for local feature extraction.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(self, dim, hidden_dim=64, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(dim).unsqueeze(-1).unsqueeze(-1), 
                requires_grad=True
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.mlp(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


def create_stage(dim, stage_idx, layers, mlp_ratio=4.,
                 frequency_alpha=0.5, num_fdblocks=1,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    Create a Machar stage with Conv2DBlocks and FDBlocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    blocks = []
    num_layers = layers[stage_idx]
    
    for block_idx in range(num_layers):
        # Use FDBlock for the last num_fdblocks blocks
        if num_layers - block_idx <= num_fdblocks:
            blocks.append(FDBlock(
                hidden_dim=dim, 
                frequency_alpha=frequency_alpha,
                stage_idx=stage_idx
            ))
        else:
            # Use Conv2DBlock for local feature extraction
            blocks.append(Conv2DBlock(
                dim=dim, 
                hidden_dim=int(mlp_ratio * dim)
            ))
    
    return nn.Sequential(*blocks)


class Machar(nn.Module):
    """
    Machar: Frequency-aware Mamba-Convolution Hybrid Architecture for HAR
    """
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 num_fdblocks=1,
                 frequency_alphas=None,
                 distillation=True,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        # Frequency scheduling: alpha values for each stage
        if frequency_alphas is None:
            frequency_alphas = [0.5, 0.75]  # Stage 1: 1/2, Stage 2: 3/4

        # Stem layer for initial feature extraction
        self.patch_embed = stem(1, embed_dims[0])

        # Build network stages
        network = []
        for i in range(len(layers)):
            alpha = frequency_alphas[i] if i < len(frequency_alphas) else 0.5
            stage = create_stage(
                embed_dims[i], i, layers, 
                mlp_ratio=mlp_ratios,
                frequency_alpha=alpha,
                num_fdblocks=num_fdblocks,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value
            )
            network.append(stage)
            
            # Add patch embedding between stages
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbedding(
                        patch_size=down_patch_size, 
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], 
                        embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # Add norm layer for each output (for dense prediction tasks)
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = nn.BatchNorm2d(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head for activity recognition
            self.norm = nn.BatchNorm2d(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes
            ) if num_classes > 0 else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes
                ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_features(x)
        
        if self.fork_feat:
            return x

        x = self.norm(x)
        
        if self.dist:
            cls_out = (
                self.head(x.flatten(2).mean(-1)), 
                self.dist_head(x.flatten(2).mean(-1))
            )
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        
        return cls_out


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': .95, 
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def machar(pretrained=False, **kwargs):
    """Machar model for HAR"""
    model = Machar(
        layers=Machar_depth['S'],
        embed_dims=Machar_width['S'],
        downsamples=[True, True],
        frequency_alphas=[0.5, 0.75],
        num_fdblocks=1,
        **kwargs
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


