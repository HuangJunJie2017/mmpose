import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from ..registry import HEADS
from .pgcn import PGCN
from .top_down_simple_head import TopDownSimpleHead


@HEADS.register_module()
class TopDownUnetHead(nn.Module):
    """Top-down model head of a U-structure baseline net proposed in ``Structure-aware human pose 
    estimation with graph convolutional networks``.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(1, 1, 1),
                 extra=None):
        super().__init__()

        self.in_channels = in_channels

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers1, self.deconv_layers2, self.deconv_layers3 = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        # add lateral connections
        self.lateral_ds1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1
            ),
            nn.BatchNorm2d(256)
        )
        self.lateral_ds2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1
            ),
            nn.BatchNorm2d(256)
        )
        self.lateral_ds3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=256,
                kernel_size=1
            ),
            nn.BatchNorm2d(256)
        )

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            self.final_layer = build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=num_deconv_filters[-1]
                if num_deconv_layers > 0 else in_channels,
                out_channels=out_channels*16,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

    def forward(self, x):
        """Forward function."""
        assert len(x) == 4
        lateral_ds1 = self.lateral_ds1(x[0])
        lateral_ds2 = self.lateral_ds2(x[1])
        lateral_ds3 = self.lateral_ds3(x[2])

        up1 = self.deconv_layers1(x[3])
        up1 = up1 + lateral_ds3

        up2 = self.deconv_layers2(up1)
        up2 = up2 + lateral_ds2

        up3 = self.deconv_layers3(up2)
        up3 = up3 + lateral_ds1

        x = self.final_layer(up3)
        return x

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            layer = []

            planes = num_filters[i]
            kernels = num_kernels[i]
            layer.append(
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            layer.append(
                nn.Conv2d(
                    in_channels=self.in_channels, 
                    out_channels=planes,
                    kernel_size=kernels,
                    bias=False
                )
            )
            layer.append(nn.BatchNorm2d(planes))
            layer.append(nn.ReLU(inplace=True))
            self.in_channels = planes
            layers.append(nn.Sequential(*layer))

        return layers

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for layer in [self.deconv_layers1, self.deconv_layers2, self.deconv_layers3]:
            for _, m in layer.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


@HEADS.register_module()
class TopDownPGCNHead(TopDownUnetHead):
    """Top-down model head of PGCN proposed in ``Structure-aware human pose 
    estimation with graph convolutional networks``.

    TopDownSimpleHead is a U-structure baseline net followed by 
    the proposed pgcn.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        extra (dict): configs of pgcn
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(1, 1, 1),
                 extra=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_deconv_layers=num_deconv_layers
        )

        self.pgcn = PGCN(
            in_channels=extra['in_channels'],
            out_channels=extra['out_channels'],
            type=extra['att_type'])

    def forward(self, x):
        """Forward function."""
        assert len(x) == 4
        lateral_ds1 = self.lateral_ds1(x[0])
        lateral_ds2 = self.lateral_ds2(x[1])
        lateral_ds3 = self.lateral_ds3(x[2])

        up1 = self.deconv_layers1(x[3])
        up1 = up1 + lateral_ds3

        up2 = self.deconv_layers2(up1)
        up2 = up2 + lateral_ds2

        up3 = self.deconv_layers3(up2)
        up3 = up3 + lateral_ds1

        x = self.final_layer(up3)
        x = self.pgcn(x)

        return x

@HEADS.register_module()
class TopDownSimPGCNHead(TopDownSimpleHead):
    """TopDownSimPGCNHead is a variant of simple baseline, whose last 
    conv2d layer followed by a proposed PGCN in ``Structure-aware human 
    pose estimation with graph convolutional networks``.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels
        )

        self.pgcn = PGCN(
            in_channels=extra['in_channels'],
            out_channels=extra['out_channels'],
            type=extra['att_type'])

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        x = self.pgcn(x)
        return x