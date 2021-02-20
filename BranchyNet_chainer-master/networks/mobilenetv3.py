"""
构建MobileNetV3网络模型：由109个卷积层和1个全连接层组成
"""
"""
MobileNetV3
"""

from typing import Callable, List, Optional
from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
import chainer


def relu6(x):
    """ReLU 6 activation function."""
    return F.clipped_relu(x, 6.)


def hard_sigmoid(x):
    """Hard version of sigmoid function."""
    return relu6(x + 3.) / 6.

def hard_swish(x):
    """Hard version of swish function."""
    return x * relu6(x + 3.) / 6.

class ConvBnActiv(chainer.Chain):
    """Conv-BN-Activation block."""

    def __init__(self, in_ch, out_ch, ksize, activation=F.relu, stride=1):
        w = chainer.initializers.HeNormal() 
        assert ksize in (1, 3)
        pad = (ksize - 1) // 2
        activation_ = activation
        super(ConvBnActiv, self).__init__(conv = L.Convolution2D(in_ch, out_ch, ksize,
                                          stride=stride, pad=pad, initialW=w), 
                                          bn = L.BatchNormalization(out_ch),
                                          activ = FL(activation_))
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.stride = stride

    def __deepcopy__(self, memo):
        new = type(self)(self.in_ch, self.out_ch, self.ksize, self.activ, self.stride)
        return new

    def __call__(self, x):
        h = self.activ(self.bn(self.conv(x)))
        return h

class SEModule(chainer.Chain):
    """Squeeze-and-Excitation module."""

    def __init__(self, ch):
        super(SEModule, self).__init__(fc1 = L.Linear(ch, ch // 4),
                                       fc2 = L.Linear(ch // 4, ch)
                                       )
        self.ch = ch

    def __deepcopy__(self, memo):
        new = type(self)(self.ch)
        return new                                   

    def __call__(self, x):
        N, C, H, W = x.shape
        h = F.average_pooling_2d(x, (H, W)).reshape(N, C)
        h = F.relu(self.fc1(h))
        h = hard_sigmoid(self.fc2(h))
        h = F.transpose(F.broadcast_to(h, (H, W, N, C)), (2, 3, 0, 1))
        return x * h


class Bneck(chainer.Chain):
    """Bottleneck module."""

    def __init__(self, in_ch, exp_ch, out_ch, ksize,
                 stride=1, use_se=False, activation=F.relu):
        assert ksize in (3, 5)
        pad = (ksize - 1) // 2
        w = chainer.initializers.HeNormal()
        activation_ = activation
        super(Bneck, self).__init__(conv1 = L.Convolution2D(in_ch, exp_ch, 1, nobias=True, initialW=w),
                                    bn1 = L.BatchNormalization(exp_ch),
                                    conv2 = L.Convolution2D(exp_ch, exp_ch, ksize, stride, pad,
                                         groups=exp_ch, nobias=True, initialW=w),
                                    bn2 = L.BatchNormalization(exp_ch),
                                    conv3 = L.Convolution2D(exp_ch, out_ch, 1, nobias=True, initialW=w),
                                    bn3 = L.BatchNormalization(out_ch),
                                    se = SEModule(exp_ch),
                                    activ = FL(activation_))
        self.in_ch      = in_ch
        self.exp_ch     = exp_ch
        self.out_ch     = out_ch
        self.ksize      = ksize
        self.stride     = stride
        self.use_se     = use_se
        self.skip = in_ch == out_ch and stride == 1

    def __deepcopy__(self, memo):
        new = type(self)(self.in_ch, self.exp_ch, self.out_ch, 
                         self.ksize, self.stride, self.use_se,
                         self.activ)
        return new

    def __call__(self, x):
        h = self.activ(self.bn1(self.conv1(x)))
        h = self.activ(self.bn2(self.conv2(h)))
        if self.use_se:
            h = self.se(h)
        h = self.activ(self.bn3(self.conv3(h)))
        if self.skip:
            h += x
        return h

cap = lambda n: [L.Linear(n, 10)]

class MobileNetV3:

    def build(self, num_classes, percentTrainKeeps=1):
        branches :List[Branch] = []
        branch1 = [ Bneck(40, 40, 160, ksize=5,stride=2), ConvBnActiv(160, 80, ksize=3,stride=2),
                    ConvBnActiv(80, 80, ksize=3), FL(F.average_pooling_2d, 7, 1), 
                    FL(F.squeeze)] + cap(80)
        branches.append(branch1)

        branch2 = [Bneck(80, 80, 160, ksize=5), ConvBnActiv(160, 80, ksize=3),
                    ConvBnActiv(80, 80, ksize=3, stride=2), FL(F.average_pooling_2d, 7, 1),
                    FL(F.squeeze)] + cap(80)
        branches.append(branch2)   

        branch3 = [ConvBnActiv(112, 80, ksize=3, stride=2), ConvBnActiv(80, 80, ksize=3), 
                    FL(F.average_pooling_2d, 7, 1), FL(F.squeeze)] + cap(80)
        branches.append(branch3)

        branch4 = [ConvBnActiv(160, 80, ksize=3), FL(F.average_pooling_2d, 7, 1), 
                    FL(F.squeeze)] + cap(80)
        branches.append(branch4)

        network = self.gen_nb(branches, num_classes)
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net

    def gen_nb(self, branches, num_classes):
        w = chainer.initializers.HeNormal()
        hs = {'activ': hard_swish}
        network = [
            ConvBnActiv(3, 16, ksize=3, activation=hard_swish, stride=2),
            Bneck(16, 16, 16, ksize=3),
            Bneck(16, 64, 24, ksize=3, stride=2),
            Bneck(24, 72, 24, ksize=3),
            Bneck(24, 72, 40, ksize=5, use_se=True, stride=2),
            Bneck(40, 120, 40, ksize=5, use_se=True),
            Bneck(40, 120, 40, ksize=5, use_se=True)
        ]#7

        network += [Branch(branches[0])]#branch1 6

        network += [
            Bneck(40, 240, 80, ksize=3, activation=hard_swish, stride=2),
            Bneck(80, 200, 80, ksize=3, activation=hard_swish),
            Bneck(80, 184, 80, ksize=3, activation=hard_swish),
            Bneck(80, 184, 80, ksize=3, activation=hard_swish)
        ]  #4
            
        network += [Branch(branches[1])]#branch2  6

        network += [
            Bneck(80, 480, 112, ksize=3, use_se=True, activation=hard_swish),
            Bneck(112, 672, 112, ksize=3, use_se=True, activation=hard_swish)
        ]#2

        network += [Branch(branches[2])]#branch2 5

        network += [
            Bneck(112, 672, 160, ksize=5, use_se=True, activation=hard_swish, stride=2),
            Bneck(160, 960, 160, ksize=5, use_se=True, activation=hard_swish),
            Bneck(160, 960, 160, ksize=5, use_se=True, activation=hard_swish)
        ]#3

        network += [Branch(branches[3])]#branch2 4

        network += [
            ConvBnActiv(160, 960, ksize=1, activation=hard_swish),# 7 x 7 x 960
            FL(F.average_pooling_2d, 7, 1)
        ] #2

        network += [
            Branch([FL(F.squeeze),
                L.Linear(960, 1280),#18
                FL(hard_swish),
                FL(F.dropout),
                L.Linear(1280, num_classes)])   
        ]

        return network
