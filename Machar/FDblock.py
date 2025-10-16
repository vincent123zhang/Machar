import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
try:
    import selective_scan_cuda
except ImportError:
    print("Warning: selective_scan_cuda not found. Mamba functionality may not work.")


class Conv2d_BN(torch.nn.Sequential):
    """Conv2d with BatchNorm fusion support"""
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
            stride=self.c.stride, padding=self.c.padding, 
            dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepDW(torch.nn.Module):
    """Reparameterized Depthwise Convolution (RepDW-3)"""
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @torch.no_grad()
    def fuse(self):
        """Fuse conv3x3, conv1x1 and identity for inference"""
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        # Pad conv1x1 to 3x3
        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
        
        # Identity kernel
        identity = torch.nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), 
            [1, 1, 1, 1]
        )

        # Fuse all paths
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        # Fuse with BN
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class FFN(nn.Module):
    """
    Feed-Forward Network with 1x1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_dim, mid_dim=None, out_dim=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = Conv2d_BN(in_dim, mid_dim, 1)
        self.fc2 = Conv2d_BN(mid_dim, out_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelectiveScan(torch.autograd.Function):
    """Selective Scan operation for Mamba"""
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, 
                delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        assert u.shape[1] % (B.shape[1] * nrows) == 0
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        
        # Ensure contiguous
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        out, x, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C, D, None, delta_bias, delta_softplus
        )
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, 
            ctx.delta_softplus, False
        )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class CrossScan(torch.autograd.Function):
    """Cross-scan operation for 2D selective scan"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    """Cross-merge operation for 2D selective scan"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    force_fp32=True,
):
    """Cross-directional selective scan for 2D data"""
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    
    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)

    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, 
                      delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, 
                                   delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)
    
    y: torch.Tensor = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class FreqDecoupler(nn.Module):
    """
    Frequency Decoupler module for separating high and low frequency components.
    Uses Laplacian pyramid-like decomposition.
    """
    def __init__(self, dim, frequency_alpha=0.5, downsample_factor=2):
        super().__init__()
        self.frequency_alpha = frequency_alpha
        self.downsample_factor = downsample_factor
        
        # Split channels based on frequency scheduling
        self.low_channels = int(dim * frequency_alpha)
        self.high_channels = dim - self.low_channels
        
        # Low-frequency path: refine with RepDW then downsample-upsample
        self.low_refine = RepDW(self.low_channels)
        self.pool = nn.AvgPool2d(downsample_factor)
        
        # High-frequency path: two 1x1 convs for detail enhancement
        self.high_enhance = nn.Sequential(
            nn.Conv2d(self.high_channels, self.high_channels, 1),
            nn.Conv2d(self.high_channels, self.high_channels, 1)
        )
        
        # High-pass residual from low-frequency path
        self.high_residual_conv = nn.Conv2d(self.low_channels, 
                                           self.high_channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            low_out: Low-frequency features [B, alpha*C, H, W]
            high_out: High-frequency features [B, (1-alpha)*C, H, W]
        """
        # Split channels
        low, high = torch.split(x, [self.low_channels, self.high_channels], dim=1)
        
        B, C, H, W = low.shape
        
        # Low-frequency path: Laplacian pyramid decomposition
        low_refined = self.low_refine(low)
        low_down = self.pool(low_refined)
        low_up = F.interpolate(low_down, (H, W), mode='nearest')
        
        # High-frequency residual from low path
        low_residual = low_refined - low_up
        high_from_low = self.high_residual_conv(low_residual)
        
        # High-frequency path: detail enhancement
        high_enhanced = self.high_enhance(high)
        
        # Combine high-frequency components
        high_out = high_enhanced + high_from_low
        low_out = low_up
        
        return low_out, high_out


class MambaBranch(nn.Module):
    """
    Mamba branch for processing low-frequency global features.
    """
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)

        self.K = 4  # Four scanning directions

        # Input projection
        self.in_proj = Conv2d_BN(d_model, d_inner, 1)
        self.act = act_layer()
        
        # Optional depthwise convolution
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                d_inner, d_inner, kernel_size=d_conv, 
                padding=d_conv // 2, groups=d_inner, bias=conv_bias
            )

        # x projection
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt projection
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # A, D parameters
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K, merge=True)

        # Output projection
        self.out_proj = Conv2d_BN(d_inner, d_model, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        x = self.in_proj(x)
        if self.d_conv > 1:
            x = self.act(self.conv2d(x))

        # Cross-directional selective scan
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=1, delta_softplus=True, force_fp32=self.training,
        )
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(self.out_proj(x))
        
        return x


class FDBlock(nn.Module):
    """
    Frequency-Domain Decoupling Block (FDBlock).
    Combines FreqDecoupler with Mamba branch (low-freq) and Conv branch (high-freq).
    """
    def __init__(
        self,
        hidden_dim: int = 96,
        drop_path: float = 0,
        frequency_alpha: float = 0.5,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        stage_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stage_idx = stage_idx
        self.frequency_alpha = frequency_alpha

        # Conv2D Block for initial local processing
        self.conv_block = Conv2DBlock(
            dim=hidden_dim,
            hidden_dim=int(mlp_ratio * hidden_dim)
        )

        # FreqDecoupler for frequency separation
        self.freq_decoupler = FreqDecoupler(
            dim=hidden_dim,
            frequency_alpha=frequency_alpha,
            downsample_factor=2
        )

        # Mamba branch for low-frequency global modeling
        low_dim = int(hidden_dim * frequency_alpha)
        self.mamba_branch = MambaBranch(
            d_model=low_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
        )

        # RepDW-3 for high-frequency local details
        high_dim = hidden_dim - low_dim
        self.high_conv = RepDW(high_dim)
        self.high_proj = nn.Conv2d(high_dim, high_dim, 1)

        # FFN for final feature mixing
        self.ffn = FFN(
            in_dim=hidden_dim,
            mid_dim=int(hidden_dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, x: torch.Tensor):
        """
        Forward pass of FDBlock.
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Step 1: Conv2D Block for initial processing
        x_conv = self.conv_block(x)
        
        # Step 2: FreqDecoupler - split into low and high frequency
        low_freq, high_freq = self.freq_decoupler(x_conv)
        
        # Step 3: Process low-frequency with Mamba (global context)
        if self.stage_idx < 2:  # Apply upsampling only for early stages
            B_low, C_low, H_low, W_low = low_freq.shape
            low_processed = self.mamba_branch(low_freq)
            # Upsample back to original size
            low_processed = F.interpolate(low_processed, (H, W), mode='bilinear')
        else:
            low_processed = self.mamba_branch(low_freq)
        
        # Step 4: Process high-frequency with RepDW-3 (local details)
        high_processed = self.high_conv(high_freq)
        high_processed = self.high_proj(high_processed)
        
        # Step 5: Concatenate and add residual
        x_merged = torch.cat([low_processed, high_processed], dim=1)
        x_merged = x_merged + x_conv  # Residual connection
        
        # Step 6: FFN with residual
        x_out = x_merged + self.drop_path(self.ffn(x_merged))
        
        return x_out

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)


class Conv2DBlock(nn.Module):
    """
    Conv2D Block with RepDW-3 and FFN.
    Used before FreqDecoupler for initial feature extraction.
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
