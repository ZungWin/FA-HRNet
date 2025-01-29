import torch.nn as nn
import logging
from nets.layer import _round_filters, ConvBNReLU, MBConvBlock, _round_repeats, Hourglass, BasicBlock, Bottleneck, \
    CoordAtt
from config import cfg

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1
blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class EfficientNet(nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()
        self.depth_mult = cfg.width_mult
        settings = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        out_channels = _round_filters(32, cfg.width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]
        in_channels = out_channels

        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, cfg.width_mult)
            repeats = _round_repeats(n, cfg.depth_mult)

            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
        self.features = nn.Sequential(*features)

    def init_weights(self):
        logger.info('=> init EfficientNet weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.depth_mult == 0.483:
            for i in range(0, 3):
                x = self.features[i](x)
            x1 = x
            for i in range(3, 4):
                x = self.features[i](x)
            x2 = x
            for i in range(4, 8):
                x = self.features[i](x)
            x3 = x
            for i in range(8, 11):
                x = self.features[i](x)
            x4 = x
        if self.depth_mult == 0.578:
            for i in range(0, 4):
                x = self.features[i](x)
            x1 = x
            for i in range(4, 6):
                x = self.features[i](x)
            x2 = x
            for i in range(6, 10):
                x = self.features[i](x)
            x3 = x
            for i in range(10, 14):
                x = self.features[i](x)
            x4 = x
        if self.depth_mult == 0.694:
            for i in range(0, 4):
                x = self.features[i](x)
            x1 = x
            for i in range(4, 6):
                x = self.features[i](x)
            x2 = x
            for i in range(6, 12):
                x = self.features[i](x)
            x3 = x
            for i in range(12, 16):
                x = self.features[i](x)
            x4 = x
        if self.depth_mult == 1 or self.depth_mult == 0.833:
            for i in range(0, 4):
                x = self.features[i](x)
            x1 = x
            for i in range(4, 6):
                x = self.features[i](x)
            x2 = x
            for i in range(6, 12):
                x = self.features[i](x)
            x3 = x
            for i in range(12, 17):
                x = self.features[i](x)
            x4 = x
        elif self.depth_mult == 1.1 or self.depth_mult == 1.2:
            for i in range(0, 6):
                x = self.features[i](x)
            x1 = x
            for i in range(6, 9):
                x = self.features[i](x)
            x2 = x
            for i in range(9, 17):
                x = self.features[i](x)
            x3 = x
            for i in range(17, 24):
                x = self.features[i](x)
            x4 = x
        elif self.depth_mult == 1.4:
            for i in range(0, 6):
                x = self.features[i](x)
            x1 = x
            for i in range(6, 9):
                x = self.features[i](x)
            x2 = x
            for i in range(9, 19):
                x = self.features[i](x)
            x3 = x
            for i in range(19, 27):
                x = self.features[i](x)
            x4 = x
        elif self.depth_mult == 1.8:
            for i in range(0, 7):
                x = self.features[i](x)
            x1 = x
            for i in range(7, 11):
                x = self.features[i](x)
            x2 = x
            for i in range(11, 23):
                x = self.features[i](x)
            x3 = x
            for i in range(23, 33):
                x = self.features[i](x)
            x4 = x

        return [x1, x2, x3, x4]


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.hg_up_layers, self.hg_down_layers, self.fuse_layers, self.conv1ds = self._make_fuse_layers()
        self.relu = nn.ReLU(True)
        self.downsamples = nn.ModuleDict()
        for i in range(num_branches):
            for j in range(num_branches):
                if i > j:
                    self.downsamples[f"{i}_{j}"] = self._make_downsample_layer(i, j)

    def _make_downsample_layer(self, i, j):
        return nn.Upsample(scale_factor=2 ** (i - j), mode='nearest')

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        hg_up_layers = []
        hg_down_layers = []
        conv1d_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer, hg_up_layer, hg_down_layer, conv1d_layer = [], [], [], []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                    )
                    )
                    hg_up_layer.append(
                        Hourglass(blocks_dict['BOTTLENECK'], cfg.hour_blocks, num_inchannels[i] // 2, j - i))
                    hg_down_layer.append(
                        Hourglass(blocks_dict['BOTTLENECK'], cfg.hour_blocks, num_inchannels[i] // 2, j - i, True))
                    conv1d_layer.append(None)
                elif j == i:
                    fuse_layer.append(None)
                    hg_up_layer.append(None)
                    hg_down_layer.append(None)
                    conv1d_layer.append(None)
                else:
                    conv3x3s, conv1x1s = [], []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    conv1d_layer.append(nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[i], num_inchannels[j], 1, 1, 0, bias=False
                        ),
                        nn.BatchNorm2d(num_inchannels[j]),
                        nn.ReLU(True)
                    ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
                    hg_down_layer.append(
                        Hourglass(blocks_dict['BOTTLENECK'], cfg.hour_blocks, num_inchannels[i] // 2, i - j, True))
                    hg_up_layer.append(None)
            fuse_layers.append(nn.ModuleList(fuse_layer))
            hg_down_layers.append(nn.ModuleList(hg_down_layer))
            hg_up_layers.append(nn.ModuleList(hg_up_layer))
            conv1d_layers.append(nn.ModuleList(conv1d_layer))

        return nn.ModuleList(hg_up_layers), nn.ModuleList(hg_down_layers), nn.ModuleList(fuse_layers), nn.ModuleList(
            conv1d_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    if j - i == 1:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:
                        x_tmp = self.fuse_layers[i][j](x[j])
                        _, downs = self.hg_down_layers[i][j](x_tmp)
                        y = self.hg_up_layers[i][j](y, downs)
                        y = y + x_tmp
                else:
                    if i - j == 1:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:
                        x_up = self.downsamples[f"{i}_{j}"](y)
                        _, downs, = self.hg_down_layers[i][j](x_up)
                        downs = [self.conv1ds[i][j](downs[k]) for k in range(len(downs) - 1, -1, -1)]
                        x_1 = x[j] + downs[0]
                        for k, layer in enumerate(self.fuse_layers[i][j]):
                            if k < len(downs) - 1:
                                x_1 = layer(x_1) + downs[k + 1]
                            else:
                                x_1 = layer(x_1)
                        y = x_1
            x_fuse.append(self.relu(y))
        return x_fuse


class PoseHigherResolutionNet(nn.Module):

    def __init__(self, blocks, num_joints):

        super(PoseHigherResolutionNet, self).__init__()

        self.stage2_cfg = cfg.stage2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        self.trans1_branch1 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.trans1_branch2 = nn.Sequential(nn.Conv2d(40, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2, pre_stage_channels = self._make_stage(blocks, self.stage2_cfg,
                                                           num_channels)

        self.stage3_cfg = cfg.stage3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        self.trans2_branch1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.trans2_branch2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.trans2_branch3 = nn.Sequential(nn.Conv2d(112, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.stage3, pre_stage_channels = self._make_stage(blocks, self.stage3_cfg,
                                                           num_channels)

        self.stage4_cfg = cfg.stage4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        self.trans3_branch1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.trans3_branch2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.trans3_branch3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.trans3_branch4 = nn.Sequential(nn.Conv2d(320, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.stage4, pre_stage_channels = self._make_stage(blocks, self.stage4_cfg, num_channels,
                                                           multi_scale_output=False)
        deconv_cfg = cfg.deconv
        self.deconv = self._make_deconv_layers(deconv_cfg['NUM_CHANNELS'], deconv_cfg['NUM_CHANNELS'], deconv_cfg)
        self.coord_at = CoordAtt(pre_stage_channels[0], pre_stage_channels[0])
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=num_joints,
            kernel_size=cfg.final_conv_kernel,
            stride=1,
            padding=1 if cfg.final_conv_kernel == 3 else 0
        )

    def _make_deconv_layers(self, input_channels, out_channels, deconv_cfg):

        deconv_kernel, padding, output_padding = self._get_deconv_cfg(deconv_cfg['KERNEL_SIZE'])

        layers = []

        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=deconv_kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False), nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        ))
        for _ in range(deconv_cfg['NUM_BASIC_BLOCKS']):
            layers.append(nn.Sequential(
                BasicBlock(out_channels, out_channels),
            ))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_stage(self, block, layer_config, num_inchannels, multi_scale_output=True):

        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, inputs):

        x1, x2, x3, x4 = inputs

        x_list = []
        x_list.append(self.trans1_branch1(x1))
        x_list.append(self.trans1_branch2(x2))

        y_list = self.stage2(x_list)

        x_list = []
        x_list.append(self.trans2_branch1(y_list[-2]))
        x_list.append(self.trans2_branch2(y_list[-1]))
        x_list.append(self.trans2_branch3(x3))

        y_list = self.stage3(x_list)

        x_list = []
        x_list.append(self.trans3_branch1(y_list[-3]))
        x_list.append(self.trans3_branch2(y_list[-2]))
        x_list.append(self.trans3_branch3(y_list[-1]))
        x_list.append(self.trans3_branch4(x4))

        y_list = self.stage4(x_list)
        features = self.deconv(y_list[0])
        features = self.coord_at(features)
        heatmaps = self.final_layer(features)

        return heatmaps

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
