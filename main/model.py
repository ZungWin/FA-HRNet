import torch.nn as nn
from nets.loss import JointsMSELoss
from config import cfg
from nets.layer import BasicBlock, Bottleneck
from nets.module import EfficientNet, PoseHigherResolutionNet

class Model(nn.Module):

    def __init__(self, mode, efficientNet, higherResolutionNet):

        super(Model, self).__init__()

        self.mode = mode

        self.efficientNet = efficientNet

        self.higherResolutionNet = higherResolutionNet

        self.j2d_loss = JointsMSELoss(cfg.use_target_weight)

        self.idx = 0

    def forward(self, inputs, targets):

        features = self.efficientNet(inputs['img'])
        heatmaps = self.higherResolutionNet(features)

        if self.mode == 'train':

            target, target_weight = targets['heatmap'], targets['target_weight']
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = {}

            loss['joint_2d'] = self.j2d_loss(heatmaps, target, target_weight)

            return loss

        else:
            return heatmaps


def get_model(mode, num_joints):
    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck
    }

    if mode == 'train':

        efficientNet = EfficientNet()
        higherResolutionNet = PoseHigherResolutionNet(blocks_dict['BASIC'], num_joints)
        efficientNet.init_weights()
        higherResolutionNet.init_weights()
        model = Model(mode, efficientNet, higherResolutionNet)

        return model

    else:

        efficientNet = EfficientNet()
        higherResolutionNet = PoseHigherResolutionNet(blocks_dict['BASIC'], num_joints)
        model = Model(mode, efficientNet, higherResolutionNet)

        return model
