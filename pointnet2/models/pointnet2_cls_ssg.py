import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

from src.chamfer import get_chamfer_loss_tensor, get_chamfer_loss, get_chamfer_loss_from_param_tensor, get_correspondence_loss_from_param_tensor

class get_model(nn.Module):
    def __init__(self,outputs,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, outputs)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)


        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, points, cat):
        loss = nn.MSELoss()
        total_loss = loss(pred, target)
        chamfer_scale = 0.0005
        #chamfer_loss = get_chamfer_loss_tensor(pred, points, cat) * chamfer_scale
        chamfer_loss = get_chamfer_loss_from_param_tensor(pred, target, cat) * chamfer_scale
        #correspondence_loss = get_correspondence_loss_from_param_tensor(pred, target, cat)
        #print("l", total_loss, chamfer_loss)

        #return total_loss + chamfer_loss
        #return total_loss
        return chamfer_loss
        #return chamfer_scale*correspondence_loss +total_loss