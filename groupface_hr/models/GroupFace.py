import torch
import torch.nn as nn
from models.resnet import *

backbone = {18: resnet_face18(),
            50: resnet_face50(),
            101: resnet_face101()}


class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return self.act(x)


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax(dim=1)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        return intermediate, self.softmax(out)


class GroupFace(nn.Module):
    def __init__(self, resnet=18, feature_dim=512, groups=4, mode='S'):
        super(GroupFace, self).__init__()
        self.mode = mode
        self.groups = groups
        self.Backbone = backbone[resnet]
        self.feature_dim = feature_dim
        self.instance_fc = FC(4096, feature_dim)
        self.GDN = GDN(feature_dim, groups)
        self.group_fc = nn.ModuleList([FC(4096, feature_dim) for i in range(groups)])

    def forward(self, x):
        B = x.shape[0]
        x = self.Backbone(x)  # [B, 4096]
        instance_representation = self.instance_fc(x)

        # GDN
        group_inter, group_prob = self.GDN(instance_representation)
        # print(group_prob)
        # Group aware representation
        v_G = [Gk(x) for Gk in self.group_fc]  # [B, 512]

        # self. distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(dim=-1).expand(self.groups, B).T) / self.groups + (
                1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data

        # group ensemble
        group_mul_p_vk = list()
        if self.mode == 'S':
            for k in range(self.groups):
                Pk = group_prob[:, k].unsqueeze(dim=-1).expand(B, self.feature_dim)
                group_mul_p_vk.append(torch.mul(v_G[k], Pk))
            group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
        else:
            ValueError('Not Implemented....')
        # instance , group aggregation
        final = instance_representation + group_ensembled
        return group_inter, final, group_prob, group_label


if __name__ == "__main__":
    x = torch.randn(5, 3, 112, 112)
    model = GroupFace(resnet=18)
    out = model(x)
    print("==output==")
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
    print(torch.argmax(out[2], dim=1), '\n', out[3])
