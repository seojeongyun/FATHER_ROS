import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class convblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(convblock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


class squash(nn.Module):
    def __init__(self, eps=10e-21, **kwargs):
        super(squash, self).__init__()
        """
        Squash activation function presented in 'Dynamic routing between capsules'/

        eps: fuzz factor used in numeric expression

        """
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return torch.multiply(n ** 2 / (1 + n ** 2) / (n + self.eps), s)


class squash_effi(nn.Module):
    def __init__(self, eps=10e-21, **kwargs):
        """
        Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'

        Args:
            eps: fuzz factor used in numeric expression
            **kwargs:
        """
        super(squash_effi, self).__init__()
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return (1 - 1 / (torch.exp(n) + self.eps)) * (s / (n + self.eps))


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, kernel_size, padding, stride):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=nin)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(out)
        return out


class primarycapsules(nn.Module):
    def __init__(self, in_channels, dim_caps, in_w, stride=1, padding=0):
        super(primarycapsules, self).__init__()
        """
        Initialize the layer

        Args:
                in_channels:    Number of input channels
                out_channels:   Number of output channels
                dim_caps:       Dimensionality, i.e. length of the output capsule vector
        """
        self.dim_caps = dim_caps
        self.in_channels = in_channels
        self.conv = depthwise_conv(nin=in_channels, kernels_per_layer=self.dim_caps, kernel_size=in_w, stride=stride,
                                   padding=0)
        self.squash = squash()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(2), out.size(2), self.dim_caps)
        out = out.view(out.size(0), -1, self.dim_caps)
        # return out
        return self.squash(out)


class fccaps(nn.Module):
    def __init__(self, input_shape, N, D, **kwargs):
        super(fccaps, self).__init__()
        """
        Fully-connected caps layer.
        It exploites the routing mechanism, explained 'Efficient CpasNet'
        to create a parent layer of capsules (or high-level representation capsules)

        N: number of primary capsules
        D: primary capsules dimension (number of properties)
        """

        self.N = N
        self.D = D
        self.input_shape = input_shape

        self.squash = squash()
        self._initialize_W()

    def _initialize_W(self):
        self.W = nn.Parameter(torch.zeros(self.input_shape[-2], self.N, self.input_shape[-1], self.D))
        self.b = nn.Parameter(torch.zeros(self.input_shape[-2], self.N))
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, u):
        # shape of u: (batch_size, in_caps, in_dim) - bin
        # shape of W: (in_caps, out_caps, in_dim, out_dim) - ijnm
        # shape of u_hat = W * x: (batch_size, in_caps, out_caps, out_dim) - bijm
        u_hat = torch.einsum('ijnm,bin->bijm', self.W, u)
        #
        u_hat_detach = u_hat.detach()
        # self-attention tensor
        # shape of a: (batch_size, in_caps, out_caps)
        a = torch.einsum('bijm,bijm->bij', u_hat_detach, u_hat_detach) \
            / torch.sqrt(torch.tensor(self.D, dtype=torch.float32))
        # shape of c: softmax(a) + b - bij
        #             (batch_size, in_caps, out_caps) + (in_caps, out_caps)
        # c = F.softmax(a, dim=2)
        c = F.softmax(a, dim=2) + self.b
        # shape of s = u_hat * c: (batch_size, out_caps, out_dim)
        s = torch.einsum('bij,bijm->bjm', c, u_hat)
        v = self.squash(s)

        return v


class efficientcapsnet(nn.Module):
    def __init__(self, data_h, data_w, capdimen, predcapdimen, numpricap, num_final_cap):
        super(efficientcapsnet, self).__init__()
        #
        self.img_shape = (3, data_h, data_w)
        self.capdimen = capdimen
        self.predcapdimen = predcapdimen
        #
        self.num_final_cap = num_final_cap
        self.numpricap = numpricap
        # ------- stem block -------
        self.convs = nn.Sequential(
            # input: (#, 3, 224, 224)
            convblock(in_channel=3, out_channel=32,
                      kernel_size=7, stride=3, padding=1),
            # out: (#, 32, 74, 74)
            convblock(in_channel=32, out_channel=int(self.numpricap / 4),
                      kernel_size=3, stride=2, padding=0),
            # # out: (#, 64, 36, 36)
            convblock(in_channel=int(self.numpricap / 4), out_channel=int(self.numpricap / 2),
                      kernel_size=3, stride=2, padding=0),
            # # out: (#, 128, 17, 17)
            convblock(in_channel=int(self.numpricap / 2), out_channel=self.numpricap,
                      kernel_size=3, stride=1, padding=0),
            # out: (#, 256, 15, 15)
        )
        # ------- primary capsules -------
        self.primarycaps = primarycapsules(in_channels=self.numpricap, dim_caps=self.capdimen,
                                           in_w=self.calc_w_h_stem())
        # out: (#, self.channels, self.capdimen)

        # ------- fccaps -------
        input_shape_fccaps = self.calc_input_shape_fccaps()
        assert (
                       self.num_final_cap * self.predcapdimen) == 4096, f'num_final_cap * predcapdimen should be 4096... But currently {self.num_final_cap * self.predcapdimen}...'
        self.fccaps = fccaps(N=self.num_final_cap, D=self.predcapdimen, input_shape=input_shape_fccaps)

    def forward(self, x):
        out = self.convs(x)
        out = self.primarycaps(out)
        out = self.fccaps(out)
        return out.view(x.shape[0], -1)

    def margin_loss(self, v_j, label):
        batch_size = v_j.size(0)

        v_j_norm = torch.norm(v_j, dim=2, keepdim=True)

        left = F.relu(self.margin_loss_upper - v_j_norm).view(batch_size, -1) ** 2
        right = F.relu(v_j_norm - self.margin_loss_lower).view(batch_size, -1) ** 2

        loss = label * left + self.lambda_margin * (1 - label) * right
        loss = loss.sum(dim=1).mean()

        if torch.isnan(loss):
            print('loss is nan...')

        return loss

    def predict(self):
        y = torch.norm(self.output_cl, dim=2)
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_networks(self, net, net_type, device, weight_path=None):
        load_filename = 'latest_net_{}.pth'.format(net_type)
        if weight_path is None:
            ValueError('Should set the weight_path, which is the path to the folder including weights')
        else:
            load_path = os.path.join(weight_path, load_filename)
        net = net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        else:
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net

    def calc_input_shape_fccaps(self):
        temp_input = torch.zeros(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        out = self.convs(temp_input)
        out = self.primarycaps(out)
        return out.shape

    def calc_w_h_stem(self):
        temp_input = torch.randn(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        out = self.convs(temp_input)
        return out.size(2)


class efficientrescaps(nn.Module):
    def __init__(self, data_h, data_w, capdimen, predcapdimen, numpricap, num_final_cap):
        super(efficientrescaps, self).__init__()
        #
        self.img_shape = (3, data_h, data_w)
        self.capdimen = capdimen
        self.predcapdimen = predcapdimen
        self.in_channels = 64
        self.use_se = True
        #
        self.num_final_cap = num_final_cap
        self.numpricap = numpricap
        # ------- stem block -------
        block = IRBlock
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, self.numpricap, 2, stride=2)
        self.bn4 = nn.BatchNorm2d(self.numpricap)
        self.dropout = nn.Dropout()
        # ------- primary capsules -------
        self.primarycaps = primarycapsules(in_channels=self.numpricap, dim_caps=self.capdimen,
                                           in_w=self.calc_w_h_stem())
        # out: (#, self.channels, self.capdimen)

        # ------- fccaps -------
        input_shape_fccaps = self.calc_input_shape_fccaps()
        assert (
                       self.num_final_cap * self.predcapdimen) == 4096, f'num_final_cap * predcapdimen should be 4096... But currently {self.num_final_cap * self.predcapdimen}...'
        self.fccaps = fccaps(N=self.num_final_cap, D=self.predcapdimen, input_shape=input_shape_fccaps)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)

        out = self.primarycaps(x)
        out = self.fccaps(out)
        return out.view(x.shape[0], -1)

    def _make_layer(self, block, in_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, in_channels, stride, downsample, use_se=self.use_se))
        self.in_channels = in_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, in_channels, use_se=self.use_se))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_networks(self, net, net_type, device, weight_path=None):
        load_filename = 'latest_net_{}.pth'.format(net_type)
        if weight_path is None:
            ValueError('Should set the weight_path, which is the path to the folder including weights')
        else:
            load_path = os.path.join(weight_path, load_filename)
        net = net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        else:
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net

    def calc_input_shape_fccaps(self):
        temp_input = torch.zeros(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        x = self.conv1(temp_input)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.primarycaps(x)
        return out.shape

    def calc_w_h_stem(self):
        temp_input = torch.randn(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        x = self.conv1(temp_input)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.size(2)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.SiLU()
        self.conv2 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


if __name__ == '__main__':
    net = efficientcapsnet(data_h=224, data_w=224,
                           capdimen=24, predcapdimen=32, numpricap=1024,
                           num_final_cap=128)
    out = net(torch.randn(2, 3, 224, 224))
