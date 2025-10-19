import torch
import torch.nn as nn
from torchvision.models import resnet18


class BCModel(nn.Module):
    """
    Minimal behavior cloning model:
      inputs: image (3xHxW), state (pos[2]), command [object_color, goal_color] (2)
      output: delta_pos[2] between current and next timestep
      vision encoder: ResNet18 (untrained, weights=None)
    """

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2),
        )

        # normalization buffers (per-dim percentiles)
        self.register_buffer('state_p02', torch.zeros(2))
        self.register_buffer('state_p98', torch.ones(2))
        self.register_buffer('dpos_p02', torch.zeros(2))
        self.register_buffer('dpos_p98', torch.ones(2))

    def forward(self, img: torch.Tensor, state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        # img: Bx3xHxW in [0,1]; state: Bx2 (pos); command: Bx2 ([obj_color, goal_color])
        z = self.backbone(img)
        s = (2.0 * (state - self.state_p02) / (self.state_p98 - self.state_p02) - 1.0).clamp(-1.5, 1.5)
        x = torch.cat([z, s, command], dim=-1)
        dpos_norm = self.head(x)
        return dpos_norm
