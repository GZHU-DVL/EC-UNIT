import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_network import BaseNetwork, AdaptiveInstanceNorm


# The code is developed based on SPADE
# https://github.com/NVlabs/SPADE/blob/master/models/networks/

class AdaResnetBlock(BaseNetwork):
    def __init__(self, fin, fout, style_nc=256):
        super(AdaResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = fout

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = AdaptiveInstanceNorm(style_nc, fin)
        self.norm_1 = AdaptiveInstanceNorm(style_nc, fmiddle)
        if self.learned_shortcut:
            self.norm_s = AdaptiveInstanceNorm(style_nc, fin)

    def forward(self, x, style):
        x_s = self.shortcut(x, style)

        dx = self.conv_0(self.actvn(self.norm_0(x, style)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, style)))

        out = x_s + dx

        return out

    def shortcut(self, x, style):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, style))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class DynamicSkipLayer(BaseNetwork):
    def __init__(self, hidden_nc, feature_nc):
        super(DynamicSkipLayer, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.trans = nn.Conv2d(hidden_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        self.reset = nn.Conv2d(feature_nc * 2, feature_nc, 3, padding=1, padding_mode='reflect')
        self.mask = nn.Conv2d(feature_nc * 2, feature_nc, 3, padding=1, padding_mode='reflect')
        self.update = nn.Conv2d(feature_nc * 2, feature_nc, 3, padding=1, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, fdec, fenc, s):
        state = F.leaky_relu(self.trans(self.up(s)), 2e-1)
        reset_gate = self.sigmoid(self.reset(torch.cat((state, fenc), dim=1)))
        new_state = reset_gate * state
        mask = self.mask(torch.cat((state, fenc), dim=1))
        mask = (F.relu(mask)).tanh()
        new_fenc = self.update(torch.cat((new_state, fenc), dim=1))
        output = (1 - mask) * fdec + mask * new_fenc
        return output, new_state, mask


class GeneratorLayer(BaseNetwork):
    """
    Set usepost = True to use two AdaRes blocks, otherwise, one AdaRes block. We use 2 blocks for the 8*8 layer.
    Set useskip = True to add dynamic skip connection to the current layer
    """

    def __init__(self, fin, fout, content_nc, style_nc, usepost=False, useskip=False):
        super(GeneratorLayer, self).__init__()

        self.reslayer = AdaResnetBlock(fin, fout, style_nc)
        self.postlayer = AdaResnetBlock(fout, fout, style_nc) if usepost else None
        self.rgblayer = nn.Conv2d(fout, 3, 7, padding=3)
        self.skiplayer = DynamicSkipLayer(content_nc, fout) if useskip else None

    def forward(self, x, style, content=None, state=None):
        mask = None
        new_state = None
        x = self.reslayer(x, style)
        if self.postlayer != None:
            x = self.postlayer(x, style)
        rgb = self.rgblayer(F.leaky_relu(x, 2e-1))
        if self.skiplayer != None and content != None and state != None:
            x, new_state, mask = self.skiplayer(x, content, state)
        return x, rgb, new_state, mask


class Generator(BaseNetwork):
    def __init__(self, content_nc=[1, 1, 512, 256, 128, 64], ngf=64):
        super(Generator, self).__init__()


        self.mlplayers = 2
        self.compress_nc = 64
        self.style_code_nc = 256
        self.style_encoder = StyleEncoder(mlp_nlayers=self.mlplayers, compress_nc=self.compress_nc,
                                          style_code_nc=self.style_code_nc)

        sequence = []
        sequence.append(GeneratorLayer(1, 8 * ngf, content_nc[0], self.style_code_nc, usepost=True))
        sequence.append(GeneratorLayer(8 * ngf, 8 * ngf, content_nc[1], self.style_code_nc, useskip=True))
        sequence.append(GeneratorLayer(8 * ngf, 4 * ngf, content_nc[2], self.style_code_nc, useskip=True))
        sequence.append(GeneratorLayer(4 * ngf, 2 * ngf, content_nc[3], self.style_code_nc))
        sequence.append(GeneratorLayer(2 * ngf, 1 * ngf, content_nc[4], self.style_code_nc))
        sequence.append(GeneratorLayer(1 * ngf, 1 * ngf, content_nc[5], 128))

        self.model = nn.Sequential(*sequence)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, content, style, scale=5, useZ=False, useskip=True):
        #generate image from multi-level features
        masks = []
        if not useZ:
            style = self.style_encoder(style, True)
        state = content[0] if useskip else None

        x, rgb, _, _ = self.model[0](content[0], style[0])
        if scale == 0:
            return torch.tanh(rgb), masks

        # content[0] and content[1] are both 8*8 features
        # current layer is 16*16, so we use content[2]
        x = self.up(x)
        x, rgb, state, mask = self.model[1](x, style[1], content[2], state)
        if mask != None:
            masks += [mask]
        if scale == 1:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[2](x, style[2], content[3], state)
        if mask != None:
            masks += [mask]
        if scale == 2:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[3](x, style[3], content[4], state)
        if mask != None:
            masks += [mask]
        if scale == 3:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[4](x, style[4], content[5], state)
        if mask != None:
            masks += [mask]
        if scale == 4:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, _, _ = self.model[5](x, style[5])

        return torch.tanh(rgb), masks


class StyleEncoder(BaseNetwork):
    """
    Set mlplayers = N to use N layers for the transform of style code
    Set compress_nc = N to transform 512-channel style code to N channels and then back to style_code_nc=256 channels
    """

    def __init__(self, nef=64, enc_nlayers=5, mlp_nlayers=1, compress_nc=64, style_code_nc=256):
        super(StyleEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, nef, 7, padding=3, padding_mode='reflect'),
            nn.ReLU()
            # B*64*256*256
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(nef, nef * 2, 4, padding=1, stride=2, padding_mode='reflect'),
            nn.ReLU()  # 128*128*128
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(nef * 2, nef * 4, 4, padding=1, stride=2, padding_mode='reflect'),
            nn.ReLU()  # 256*64*64
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(nef * 4, nef * 4, 4, padding=1, stride=2, padding_mode='reflect'),
            nn.ReLU()  # 256*32*32
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(nef * 4, nef * 4, 4, padding=1, stride=2, padding_mode='reflect'),
            nn.ReLU()  # 256*16*16
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(nef * 4, nef * 4, 4, padding=1, stride=2, padding_mode='reflect'),
            nn.ReLU()  # 256*8*8
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        num_filters = 256
        mlp = []
        mlp.append(nn.Linear(num_filters, compress_nc))
        mlp.append(nn.ReLU())
        # 64
        for _ in range(mlp_nlayers):
            mlp.append(nn.Linear(compress_nc, num_filters))
            compress_nc = num_filters
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x, get_Feature=False):
        #get features for MSE
        x = self.layer1(x)
        s1 = self.avg_pool(x)
        s1 = s1.view(s1.size(0), -1)
        x = self.layer2(x)
        s2 = self.avg_pool(x)
        s2 = s2.view(s2.size(0), -1)
        x = self.layer3(x)
        s3 = self.avg_pool(x)
        s3 = s3.view(s3.size(0), -1)
        x = self.layer4(x)
        s4 = self.avg_pool(x)
        s4 = s4.view(s4.size(0), -1)
        x = self.layer5(x)
        s5 = self.avg_pool(x)
        s5 = s5.view(s5.size(0), -1)
        x = self.layer6(x)
        s6 = self.avg_pool(x)
        s6 = s6.view(s6.size(0), -1)
        s7 = self.mlp(s6)
        if (get_Feature == False):
            return s7
        else:
            return [s7, s6, s5, s4, s3, s2]
