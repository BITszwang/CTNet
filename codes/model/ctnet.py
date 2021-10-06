import math
import torch
import torch.nn as nn

import torch.nn.functional as F


def make_model(args, parent=False):
    return CTNET(in_nc=args.in_nc, out_nc=args.out_nc, nf=args.nf, unf=args.unf, nb=args.nb, scale=args.scale[0])

class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()

        self.k1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                      padding=0, dilation=dilation,
                      groups=groups, bias=False)
        )

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=2),
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                                padding=0, dilation=dilation,
                                groups=groups, bias=False)
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                                padding=0, dilation=dilation,
                                groups=groups, bias=False)
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                                padding=0, dilation=dilation,
                                groups=groups, bias=False)
                    )

    def forward(self, x):
        identity = self.k1(x)

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(identity, out) # k3 * sigmoid(identity + k2)
        # out = self.k4(out) # k4

        return out




class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


## 20201122 new added  to evaluate the effectiveness of CCA layer

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        # y = self.conv(y)
        y = self.sigmoid(y)
        return x * y



class DElayer(nn.Module):

    def __init__(self, nf):

        super(DElayer, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.cca = CCALayer(nf)

    def forward(self, x):

        y = self.cca(x)
        y = self.k1(y)
        out = x + y

        return out

####


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)


        #1204 pool_r =2
        #1209 pool_r =3

        self.primary_conv = nn.Sequential(
            SCConv(inplanes=inp, planes=init_channels, stride=stride, padding=1, dilation=1, groups=1, pooling_r=2),
            # nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            # nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )


        self.detail_improve = nn.Sequential(
            nn.Conv2d(inp, init_channels+new_channels, 1, stride, 0, bias=False),
            # nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )


    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        x_imporved = self.detail_improve(x)
        out = torch.cat([x1, x2], dim=1)
        out = out + x_imporved

        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                # nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class CTNET(nn.Module):

    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(CTNET, self).__init__()

        self.scale = scale
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.delayer1 = ESA(nf)
        self.ghost1 = GhostBottleneck(in_chs=nf, mid_chs=nf//2, out_chs=nf)
        self.alt1 = nn.Conv2d(nf, nf//nb, 3, 1, 3 // 2, groups=nf//nb, bias=False)

        self.delayer2 = ESA(nf)
        self.ghost2 = GhostBottleneck(in_chs=nf, mid_chs=nf//2, out_chs=nf)
        self.alt2 = nn.Conv2d(nf, nf//nb, 3, 1, 3 // 2, groups=nf//nb, bias=False)

        self.delayer3 = ESA(nf)
        self.ghost3 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt3 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer4 = ESA(nf)
        self.ghost4 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt4 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer5 = ESA(nf)
        self.ghost5 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt5 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer6 = ESA(nf)
        self.ghost6 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt6 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer7 = ESA(nf)
        self.ghost7 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt7 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer8 = ESA(nf)
        self.ghost8 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt8 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer9 = ESA(nf)
        self.ghost9 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt9 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer10 = ESA(nf)
        self.ghost10 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt10 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer11 = ESA(nf)
        self.ghost11 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt11 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer12 = ESA(nf)
        self.ghost12 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt12 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer13 = ESA(nf)
        self.ghost13 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt13 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer14 = ESA(nf)
        self.ghost14 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt14 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer15 = ESA(nf)
        self.ghost15 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt15 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.delayer16 = ESA(nf)
        self.ghost16 = GhostBottleneck(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.alt16 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)


        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.c = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

    def forward(self, x):

        fea = self.conv_first(x)

        # fea = self.delayer1(fea)
        out_b1 = self.ghost1(fea)
        skip1 = self.alt1(out_b1)

        out_b1 = self.delayer2(out_b1)
        out_b2 = self.ghost2(out_b1)
        skip2 = self.alt2(out_b2)

        out_b2 = self.delayer3(out_b2)
        out_b3 = self.ghost3(out_b2)
        skip3 = self.alt3(out_b3)

        out_b3 = self.delayer4(out_b3)
        out_b4 = self.ghost4(out_b3)
        skip4 = self.alt4(out_b4)

        out_b4 = self.delayer5(out_b4)
        out_b5 = self.ghost5(out_b4)
        skip5 = self.alt5(out_b5)

        out_b5 = self.delayer6(out_b5)
        out_b6 = self.ghost6(out_b5)
        skip6 = self.alt6(out_b6)


        out_b6 = self.delayer7(out_b6)
        out_b7 = self.ghost7(out_b6)
        skip7 = self.alt7(out_b7)

        out_b7 = self.delayer8(out_b7)
        out_b8 = self.ghost8(out_b7)
        skip8 = self.alt8(out_b8)

        out_b8 = self.delayer9(out_b8)
        out_b9 = self.ghost9(out_b8)
        skip9 = self.alt9(out_b9)

        out_b9 = self.delayer10(out_b9)
        out_b10 = self.ghost10(out_b9)
        skip10 = self.alt10(out_b10)

        out_b10 = self.delayer11(out_b10)
        out_b11= self.ghost11(out_b10)
        skip11 = self.alt11(out_b11)

        out_b11 = self.delayer12(out_b11)
        out_b12 = self.ghost12(out_b11)
        skip12 = self.alt12(out_b12)

        out_b12 = self.delayer13(out_b12)
        out_b13 = self.ghost13(out_b12)
        skip13 = self.alt13(out_b13)


        out_b13 = self.delayer14(out_b13)
        out_b14 = self.ghost14(out_b13)
        skip14 = self.alt14(out_b14)

        out_b14 = self.delayer15(out_b14)
        out_b15 = self.ghost15(out_b14)
        skip15 = self.alt15(out_b15)

        out_b15 = self.delayer16(out_b15)
        out_b16 = self.ghost16(out_b15)
        skip16 = self.alt16(out_b16)

        trunk = self.c(torch.cat((skip1,skip2,skip3,skip4,skip5,skip6,skip7,skip8,skip9,skip10,skip11,skip12,skip13,skip14,skip15,skip16),dim=1))

        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out