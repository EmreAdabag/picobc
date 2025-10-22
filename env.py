from operator import mul
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from torchvision.io import write_video
import h5py
import zarr

class PickAndPlaceEnv:
    """
    Batched 2D pick-and-place environment in torch.

    - Always batched: all state tensors have leading batch dimension B.
    - Agent in R^2 with force control u in R^2 per batch item.
    - Two objects and two goals per batch item; each episode commands one of each.
    - Minimal physics: x_{t+1} = x_t + v_t*dt ; v_{t+1} = v_t + u*dt.
    - Optional video recording; save one video per batch item.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        exclude_combo: tuple[int, int] | None = (1, 1),
        batch_size: int = 1,
        multi_task: bool = True, 
        dt: float = 0.005
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = torch.Generator(device="cpu").manual_seed(int(seed))
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.B = int(batch_size)

        # World and physics constants (fixed)
        self.dt = torch.tensor(dt, device=self.device, dtype=self.dtype)
        self.mass = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.object_mass = torch.tensor(2.0, device=self.device, dtype=self.dtype)

        # Geometry (fixed)
        self.multi_task = multi_task
        self.agent_color = torch.tensor([230, 57, 70], device=self.device, dtype=torch.uint8)  # red (agent)
        self.object_colors = [
            torch.tensor([69, 123, 157], device=self.device, dtype=torch.uint8),   # blue-ish
            torch.tensor([255, 140, 0], device=self.device, dtype=torch.uint8),    # orange
        ]
        self.object_color_names = ["blue", "orange"]
        self.goal_colors = [
            torch.tensor([29, 185, 84], device=self.device, dtype=torch.uint8),    # green-ish
            torch.tensor([155, 89, 182], device=self.device, dtype=torch.uint8),   # purple
        ]
        self.goal_color_names = ["green", "purple"]
        self.bg_color = torch.tensor([245, 245, 245], device=self.device, dtype=torch.uint8)

        self.agent_radius_px = 3
        self.object_radius_px = 6
        self.goal_half_size = torch.tensor(0.07, device=self.device, dtype=self.dtype)

        # Interaction thresholds (fixed)
        self.pick_radius = torch.tensor(0.03, device=self.device, dtype=self.dtype)
        self.drop_radius = torch.tensor(0.03, device=self.device, dtype=self.dtype)

        # Rendering settings
        self.H = 128
        self.W = 128
        ys = torch.linspace(0.0, 1.0, self.H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(0.0, 1.0, self.W, device=self.device, dtype=self.dtype)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")  # HxW each

        # State (batched) placeholders
        self.agent_pos: torch.Tensor | None = None           # [B,2]
        self.agent_vel: torch.Tensor | None = None           # [B,2]
        self.objects_pos: torch.Tensor | None = None         # [B,2,2]
        self.objects_vel: torch.Tensor | None = None         # [B,2,2]
        self.goals_center: torch.Tensor | None = None        # [B,2,2]

        # Aliases for active object and goal
        self.object_pos: torch.Tensor | None = None          # [B,2]
        self.object_vel: torch.Tensor | None = None          # [B,2]
        self.goal_center: torch.Tensor | None = None         # [B,2]

        self.picked: torch.Tensor | None = None              # [B] bool
        self.delivered: torch.Tensor | None = None           # [B] bool
        self.success: torch.Tensor | None = None             # [B] bool

        # Command and indices
        self.command = torch.zeros(self.B, 2, device=self.device, dtype=self.dtype)  # [B,2]
        self._target_object_idx = torch.zeros(self.B, dtype=torch.long)
        self._target_goal_idx = torch.zeros(self.B, dtype=torch.long)
        self.exclude_combo = exclude_combo

        # Rendering buffer
        self._frames: list[torch.Tensor] = []  # list of [B,H,W,3] uint8
        self._record = False

    def reset(self, record_video: bool = False):
        def rnd2(B: int):
            return torch.rand(B, 2, generator=self.rng)

        B = self.B
        self.agent_pos = rnd2(B).to(device=self.device, dtype=self.dtype)
        self.agent_vel = torch.zeros(B, 2, device=self.device, dtype=self.dtype)
        self.objects_pos = torch.rand(B, 2, 2, generator=self.rng, device="cpu").to(device=self.device, dtype=self.dtype)
        self.objects_vel = torch.zeros(B, 2, 2, device=self.device, dtype=self.dtype)
        self.goals_center = torch.rand(B, 2, 2, generator=self.rng, device="cpu").to(device=self.device, dtype=self.dtype)

        # Sample command indices per batch with optional exclusion
        obj_idx = torch.randint(0, 2, (B,), generator=self.rng)
        goal_idx = torch.randint(0, 2, (B,), generator=self.rng)
        if self.multi_task is True:
            if self.exclude_combo is not None:
                excl_o, excl_g = int(self.exclude_combo[0]), int(self.exclude_combo[1])
                bad = (obj_idx == excl_o) & (goal_idx == excl_g)
                while bool(bad.any()):
                    obj_idx[bad] = torch.randint(0, 2, (int(bad.sum().item()),), generator=self.rng)
                    goal_idx[bad] = torch.randint(0, 2, (int(bad.sum().item()),), generator=self.rng)
                    bad = (obj_idx == excl_o) & (goal_idx == excl_g)
        else:
            obj_idx = goal_idx = torch.zeros((B,))
        self._target_object_idx = obj_idx.to(torch.long)
        self._target_goal_idx = goal_idx.to(torch.long)
        self.command = torch.stack([self._target_object_idx.to(self.dtype), self._target_goal_idx.to(self.dtype)], dim=-1).to(self.device)

        arangeB = torch.arange(B, device=self.device)
        self.object_pos = self.objects_pos[arangeB, self._target_object_idx]
        self.object_vel = self.objects_vel[arangeB, self._target_object_idx]
        self.goal_center = self.goals_center[arangeB, self._target_goal_idx]

        self.picked = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.delivered = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(B, dtype=torch.bool, device=self.device)

        self._record = bool(record_video)
        self._frames = []
        if self._record:
            self._frames.append(self._render_frame())

        return self._obs()

    def step(self, force: torch.Tensor):
        u = force.to(device=self.device, dtype=self.dtype).view(self.B, 2)
        
        active = self.picked & (~self.delivered)
        agent_object_mass = self.mass + active * self.object_mass

        # Dynamics
        self.agent_vel = self.agent_vel + torch.div(u, agent_object_mass.unsqueeze(1)) * self.dt
        self.agent_pos = self.agent_pos + self.agent_vel * self.dt
        self.agent_pos = self.agent_pos.clamp(0.0, 1.0)

        # Attach object if picked and not delivered
        if bool(active.any()):
            idx = torch.where(active)[0]
            self.objects_pos[idx, self._target_object_idx[idx]] = self.agent_pos[idx]
            self.objects_vel[idx, self._target_object_idx[idx]] = self.agent_vel[idx]
            # Update aliases for all
            arangeB = torch.arange(self.B, device=self.device)
            self.object_pos = self.objects_pos[arangeB, self._target_object_idx]
            self.object_vel = self.objects_vel[arangeB, self._target_object_idx]

        arangeB = torch.arange(self.B, device=self.device)
        pick_dist = torch.linalg.vector_norm(self.agent_pos - self.objects_pos[arangeB, self._target_object_idx], dim=-1)
        within_pick = pick_dist <= self.pick_radius
        self.picked = self.picked | within_pick

        need_drop = self.picked & (~self.delivered)
        drop_dist = torch.linalg.vector_norm(self.agent_pos - self.goals_center[arangeB, self._target_goal_idx], dim=-1)
        within_drop = drop_dist <= self.drop_radius
        just_delivered = need_drop & within_drop
        self.delivered = self.delivered | just_delivered
        self.success = self.delivered

        obs = self._obs()
        if self._record:
            self._frames.append(self._render_frame())

        return {
            "obs": obs,
            "picked": self.picked.clone(),
            "delivered": self.delivered.clone(),
            "success": self.success.clone(),
        }

    def _obs(self):
        arangeB = torch.arange(self.B, device=self.device)
        return torch.cat([
            self.agent_pos,
            self.agent_vel,
            self.objects_pos[arangeB, self._target_object_idx],
            self.objects_vel[arangeB, self._target_object_idx],
            self.goals_center[arangeB, self._target_goal_idx],
            self.command.to(self.dtype),
            torch.stack([self.picked.to(self.dtype), self.delivered.to(self.dtype)], dim=-1),
        ], dim=-1)

    def _render_frame(self) -> torch.Tensor:
        """Return [B,H,W,3] uint8 frame batch."""
        B = self.B
        frames = torch.empty(B, self.H, self.W, 3, dtype=torch.uint8, device=self.device)

        obj_r = self._world_radius_to_px(self.object_radius_px)
        ag_r = self._world_radius_to_px(self.agent_radius_px)

        for b in range(B):
            frame = self.bg_color.view(1, 1, 3).expand(self.H, self.W, 3).clone()
            # Goals
            half = self.goal_half_size
            for gi in range(2):
                gx, gy = self.goals_center[b, gi, 0], self.goals_center[b, gi, 1]
                mask_goal = (torch.abs(self.grid_x - gx) <= half) & (torch.abs(self.grid_y - gy) <= half)
                frame[mask_goal] = self.goal_colors[gi]
            # Objects
            for oi in range(2):
                ox, oy = self.objects_pos[b, oi, 0], self.objects_pos[b, oi, 1]
                mask_obj = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2 <= obj_r ** 2
                frame[mask_obj] = self.object_colors[oi]
            # Agent
            ax, ay = self.agent_pos[b, 0], self.agent_pos[b, 1]
            mask_agent = (self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2 <= ag_r ** 2
            frame[mask_agent] = self.agent_color
            frames[b] = frame.to(torch.uint8)

        return frames

    def _world_radius_to_px(self, radius_px: int) -> torch.Tensor:
        px_to_world = 1.0 / float(max(self.H, self.W) - 1)
        return torch.tensor(radius_px * px_to_world, device=self.device, dtype=self.dtype)

    def current_frame(self) -> torch.Tensor:
        return self._render_frame()

    def command_info(self, b: int = 0) -> dict:
        if not (0 <= b < self.B):
            raise IndexError("batch index out of range")
        oi = int(self._target_object_idx[b].item())
        gi = int(self._target_goal_idx[b].item())
        return {
            "object_index": oi,
            "object_name": self.object_color_names[oi],
            "object_rgb": self.object_colors[oi].tolist(),
            "goal_index": gi,
            "goal_name": self.goal_color_names[gi],
            "goal_rgb": self.goal_colors[gi].tolist(),
            "command": [float(self.command[b, 0].item()), float(self.command[b, 1].item())],
        }

    def _annotate_frames_with_command(self, frames: torch.Tensor) -> torch.Tensor:
        """Annotate [T,B,H,W,C] frames with chips for commanded colors."""
        if frames.ndim != 5:
            raise ValueError("frames must be [T,B,H,W,C]")
        T, B, H, W, C = frames.shape
        out = frames.clone()

        pad = 2
        size = 12
        o_y0, o_y1 = pad, min(H, pad + size)
        o_x0, o_x1 = pad, min(W, pad + size)
        gap = 4
        g_y0, g_y1 = pad, min(H, pad + size)
        g_x0, g_x1 = min(W, o_x1 + gap), min(W, o_x1 + gap + size)

        for b in range(B):
            info = self.command_info(b)
            o_col = torch.tensor(info["object_rgb"], dtype=torch.uint8).view(1, 1, 3)
            g_col = torch.tensor(info["goal_rgb"], dtype=torch.uint8).view(1, 1, 3)
            out[:, b, o_y0:o_y1, o_x0:o_x1, :] = o_col
            out[:, b, g_y0:g_y1, g_x0:g_x1, :] = g_col

        border = 1
        white = torch.tensor([255, 255, 255], dtype=torch.uint8)
        if o_y1 - o_y0 > 2 and o_x1 - o_x0 > 2:
            out[:, :, o_y0:o_y0+border, o_x0:o_x1, :] = white
            out[:, :, o_y1-border:o_y1, o_x0:o_x1, :] = white
            out[:, :, o_y0:o_y1, o_x0:o_x0+border, :] = white
            out[:, :, o_y0:o_y1, o_x1-border:o_x1, :] = white
        if g_y1 - g_y0 > 2 and g_x1 - g_x0 > 2:
            out[:, :, g_y0:g_y0+border, g_x0:g_x1, :] = white
            out[:, :, g_y1-border:g_y1, g_x0:g_x1, :] = white
            out[:, :, g_y0:g_y1, g_x0:g_x0+border, :] = white
            out[:, :, g_y0:g_y1, g_x1-border:g_x1, :] = white
        return out

    def save_video(self, path: str):
        """Save recorded frames to a single mp4 by concatenating batch along time."""
        fps = int(1 / self.dt)
        if not self._frames:
            raise RuntimeError("No frames recorded. Call reset(record_video=True) and step().")

        frames = torch.stack(self._frames, dim=0).to(torch.uint8)  # [T,B,H,W,3]
        frames = self._annotate_frames_with_command(frames)
        # Concatenate batch over time -> [T*B,H,W,3]
        T, B, H, W, C = frames.shape
        video = frames.permute(1,0,2,3,4).reshape(T*B, H, W, C).to("cpu")

        target_path = path
        write_video(filename=target_path, video_array=video, fps=fps)

        # Sidecar includes commands for each batch item
        sidecar = {
            "video_path": target_path,
            "fps": int(fps),
            "batch_size": int(self.B),
            "commands": [self.command_info(b) for b in range(self.B)],
        }
        sidecar_path = os.path.splitext(target_path)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)
