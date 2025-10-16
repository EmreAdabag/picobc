import torch
import torch.nn as nn
from torchvision.models import resnet18


class BCModel(nn.Module):
    """
    Minimal behavior cloning model:
      inputs: image (3xHxW), state (pos[2], vel[2])
      output: delta_pos[2] between current and next timestep
      vision encoder: ResNet18 (untrained, weights=None)
    """

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # img: Bx3xHxW in [0,1]; state: Bx4 (pos, vel)
        z = self.backbone(img)
        x = torch.cat([z, state], dim=-1)
        dpos = self.head(x)
        return dpos

