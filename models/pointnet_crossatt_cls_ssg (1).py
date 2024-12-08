import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3 # nomral feature 들어오면 6, 아니면 3
        self.normal_channel = normal_channel
        self.global_feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_channel)
        # [B, 1024]

        # local feature extraction
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 512, 1024], group_all=False) # mlp = 출력 채널
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=in_channel, mlp=[128, 256, 1024], group_all=False) # mlp = 출력 채널
        # group all ==> global feature / group_all = True

        # attention layer
        self.key_layer = nn.Linear(1024, 1024)
        self.value_layer = nn.Linear(1024, 1024)
        self.query_layer = nn.Linear(1024, 1024)
        
        # classification layer
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        global_feat, trans, trans_feat = self.global_feat(xyz)
        # [24, 1024]

        if self.normal_channel:
            norm = xyz[:, 3:, :] # normal feature 
            xyz = xyz[:, :3, :] # physical position info (x, y, z coordinate)
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # [24, 3, 512], [24, 1024, 512]
        
        # l1_xyz, l1_points = self.sa2(l1_points)
        # attention layer
        query = self.query_layer(global_feat) # [24, 1024]
        key = self.key_layer(l1_points.permute(0, 2, 1)) # [24, 512, 1024] [batch, num points, dim]
        value = self.value_layer(l1_points.permute(0, 2, 1)) # [24, 512, 1024] [batch, num points, dim]

        query = query.unsqueeze(1) # [24, 1, 1024]
        qk_dot = torch.sum(query*key, dim=2) # [24, 512, 1024] -> [24, 512]
        attention_score = torch.softmax(qk_dot/(1024**0.5), dim=1).unsqueeze(2) # [24, 512, 1]
        attention_output = torch.sum(attention_score * value, dim=1) # [24, 1024]

        # global feat이 너무 local feat으로만 표현되는 것을 방지하고자 glbol feat 더해줌. 
        attention_scale = nn.Parameter(torch.ones(1, device='cuda'))
        attention_output = (1-attention_scale)*attention_output + attention_scale*global_feat # [24, 1024]

        # attention_output = torch.concat([attention_output, global_feat], dim=-1) # [24, 2048]
        # classification layer
        x = attention_output.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, trans_feat

class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale # STN3d loss 얼마나 반영할지

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat) # pointnet encoder에서는 STN3d를 사용함. input 으로 들어온느 포인트클라우드 정렬함.

        # loss 항 2개 - classification loss + STN3d loss
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    model = get_model()
