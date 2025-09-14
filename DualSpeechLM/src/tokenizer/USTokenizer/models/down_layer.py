import torch
from torch import nn
import torch.nn.functional as F
import math
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

def norm_block(is_layer_norm, dim, affine=True):
    mod = Fp32GroupNorm(1, dim, affine=affine)
    return mod

class ConvAdapter(nn.Module):
    """Conv adapter that combines two modules with different label rate with downsample or upsample.
    To allow different ratios than integer, two convs are utilized with first to upsample (numerator)
    and the second to downsample (denominator)"""
    def __init__(
        self,
        k,
        label_rate,
        dropout,
        channels,
        activation,
        skip_connections=True,
        highway=True,
        residual_scale=0.4,
        non_affine_group_norm=False,
    ):
        super().__init__()

        def downsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.Conv1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=(k - 1) // 2,
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        def upsample_block(channel, k, stride):
            return nn.Sequential(
                # with padding (k - 1) // 2 to keep the same size
                nn.ConvTranspose1d(
                    channel,
                    channel,
                    k,
                    stride=stride,
                    bias=False,
                    padding=0,  # padding=(k - 1) // 2,
                    output_padding=(stride - 1),
                ),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=channel, affine=not non_affine_group_norm
                ),
                activation,
            )

        assert len(label_rate) == 2, "label_rate should be sized two to apply fusion"
        # Lout =(Lin~H~R1)~Wstride~H~R2~Wpadding+dilation~W(kernel_size~H~R1)+output_padding+1
        self.upsample_conv = upsample_block(channels, k, label_rate[0]) # 
        self.downsample_conv = downsample_block(channels, k, label_rate[1])

        self.upsample_rate, self.downsample_rate = label_rate
        self.skip_connections = skip_connections
        self.highway = highway
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # Assume x1 = (B, T, C) as input
        x = x.permute(0, 2, 1)
        residual_before_upsample = x
        x = self.upsample_conv(x) # 先上采样一下
        upsample_size = x.size(2)

        # conduct upsample
        if self.skip_connections:
            residual_upsample = torch.repeat_interleave(
                residual_before_upsample, self.upsample_rate, dim=2
            ) # repeat
            upsample_size = min(upsample_size, residual_upsample.size(2))
            x = (
                x[..., :upsample_size] + residual_upsample[..., :upsample_size]
            ) * self.residual_scale # x = x_conv + x_repeat

        residual_before_downsample = x # 
        x = self.downsample_conv(x) # down samping
        downsample_size = x.size(2)

        if self.skip_connections:
            residual_downsample = residual_before_downsample[
                ..., :: self.downsample_rate
            ] # down-samping with choose pooling
            downsample_size = min(x.size(2), residual_downsample.size(2))
            x = (
                x[..., :downsample_size] + residual_downsample[..., :downsample_size]
            ) * self.residual_scale

        if self.highway:
            residual_after_sample = residual_upsample[..., :: self.downsample_rate] # down
            final_size = min(x.size(2), residual_after_sample.size(2))
            x = (
                x[..., :final_size] + residual_after_sample[..., :final_size]
            ) * self.residual_scale

        x = x.permute(0, 2, 1)
        return x

