import os
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video


class PickAndPlaceEnv:
    """
    Minimal batched 2D pick-and-place with emoji objects.

    - Action: (ux, uy, gripper) where gripper>0.5 means closed
    - Rendering: one emoji sprite as the object; faded sprite at goal as a hint
    - Physics: x_{t+1} = x_t + v_t*dt ; v_{t+1} = v_t + u*dt
    - Rendering: one emoji sprite as the object; faded sprite at goal as a hint
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        batch_size: int = 1,
        dt: float = 0.05,
        objects_dir: str | Path = "objects",
        sprite_size_px: int = 32,
        image_size: int = 256,
        record_video: bool = False,
        max_distractors: int = 3,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = torch.Generator(device=device).manual_seed(int(seed))
        self.B = int(batch_size)

        # Physics
        self.dt = torch.tensor(dt, device=self.device, dtype=self.dtype)
        self.mass = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.object_mass = torch.tensor(2.0, device=self.device, dtype=self.dtype)

        # Rendering
        self.H = int(image_size)
        self.W = int(image_size)
        self.bg_color = torch.tensor([245, 245, 245], dtype=torch.uint8)
        self.agent_color = torch.tensor([255, 0, 0], dtype=torch.uint8)  # red
        self.agent_radius_px = 4
        self.sprite_size_px = int(sprite_size_px)
        self.max_distractors = int(max(0, max_distractors))

        # Distances (world units 0..1)
        self.pick_radius = torch.tensor(0.02, device=self.device, dtype=self.dtype)
        self.drop_radius = torch.tensor(0.02, device=self.device, dtype=self.dtype)

        # Precompute pixel grid for agent circle
        ys = torch.linspace(0.0, 1.0, self.H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(0.0, 1.0, self.W, device=self.device, dtype=self.dtype)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Pixel-space grids for blitting [1,H,W]
        self.pix_x = (self.grid_x * float(self.W - 1)).view(1, self.H, self.W)
        self.pix_y = (self.grid_y * float(self.H - 1)).view(1, self.H, self.W)

        # Load emoji sprites (RGBA)
        self.objects_dir = Path(objects_dir)
        self.sprites: List[dict] = self._load_sprites(self.objects_dir, self.sprite_size_px)
        assert len(self.sprites) > 0, f"No PNG sprites found in '{self.objects_dir}'."
        # Pre-stack sprites for batched gather: [N, h*w, 4] (rgb 0..255, alpha 0..1)
        self.sprite_h = self.sprite_w = int(self.sprite_size_px)
        self.sprites_flat4 = torch.stack([s["flat4"] for s in self.sprites], dim=0).to(self.device)
        # Use one of the object sprites as the initial goal sprite
        self.goal_id = 0
        self.goal_sprite_flat4 = self.sprites[self.goal_id]["flat4"]

        # State
        self.agent_pos = None         # [B,2]
        self.agent_vel = None         # [B,2]
        self.object_pos = None        # [B,2]
        self.object_vel = None        # [B,2]
        self.goal_pos = None          # [B,2]

        self.object_id = None         # [B] long (index into sprites)
        self.picked = None            # [B] bool
        self.delivered = None         # [B] bool
        self.success = None           # [B] bool
        self.gripper_closed = None    # [B] bool

        # Distractors (fixed per env after reset)
        self.distractor_ids = None        # [B, K] long
        self.distractor_pos = None        # [B, K, 2] float in world coords 0..1

        # Recording
        self._record = record_video
        self._frames: list[torch.Tensor] = []

        self.reset()

    # ---- environment API ----
    def reset(self, obj_idx=None, goal_idx=None):
        B = self.B
        def r2(n=B):
            return torch.clip(torch.rand(n, 2, generator=self.rng, device=self.device).to(self.dtype) * 1.04 - 0.02, min=0., max=1.)

        # Optionally override goal sprite by an object sprite
        if goal_idx is not None:
            # Use the selected object sprite as the goal sprite for this episode
            gidx = int(goal_idx)
            self.goal_id = gidx
            self.goal_sprite_flat4 = self.sprites[gidx]["flat4"]

        self.agent_pos = r2()
        # Always start not picked with gripper open; object at random position
        self.object_pos = r2()
        self.goal_pos = r2()
        
        self.agent_vel = torch.zeros(B, 2, device=self.device, dtype=self.dtype)
        self.object_vel = torch.zeros(B, 2, device=self.device, dtype=self.dtype)

        if obj_idx is not None:
            self.object_id = torch.full((B,), obj_idx, device=self.device, dtype=torch.long)
        else:
            self.object_id = torch.randint(0, len(self.sprites), (B,), device=self.device, generator=self.rng)

        # Task flags: start with open gripper and not picked
        self.picked = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.gripper_closed = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.delivered = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Distractors: sample once per env and keep fixed
        if self.max_distractors > 0:
            K = self.max_distractors
            N = len(self.sprites)
            B = self.object_id.shape[0]

            forbidden = torch.stack([self.object_id, torch.full((B,), self.goal_id, device=self.device)], dim=1)  # (B, 2)
            base = torch.randint(0, N - 2, (B, K), device=self.device, generator=self.rng)

            # shift up once per forbidden id
            base += (base >= forbidden[:, [0]]).long()
            base += (base >= forbidden[:, [1]]).long()

            self.distractor_ids = base
            self.distractor_pos = torch.rand(B, K, 2, device=self.device, generator=self.rng).to(dtype=self.dtype)
        else:
            self.distractor_ids = None
            self.distractor_pos = None

        # Video
        self._frames = []
        if self._record:
            self._frames.append(self.current_frame())
        return None

    def step(self, action: torch.Tensor):
        a = action.to(device=self.device, dtype=self.dtype)
        u = a.view(self.B, -1)[:, :2]
        prev_grip = self.gripper_closed.clone()
        gripper_close_cmd = (a.view(self.B, -1)[:, 2] > 0.5)

        # Use previous step's attachment to determine mass during this integration
        total_mass = self.mass + prev_grip * self.object_mass

        # Physics integration
        self.agent_vel = self.agent_vel + (u / total_mass.unsqueeze(1)) * self.dt
        self.agent_pos = (self.agent_pos + self.agent_vel * self.dt).clamp(0.0, 1.0)

        # Carry object during this step if it was attached at the start of the step
        self.object_pos = torch.where(prev_grip.unsqueeze(1), self.agent_pos, self.object_pos)
        self.object_vel = torch.where(prev_grip.unsqueeze(1), self.agent_vel, self.object_vel)

        # Evaluate drop at release (end-of-step)
        drop_dist_now = torch.linalg.vector_norm(self.agent_pos - self.goal_pos, dim=-1)

        # Decide realized gripper state after integration
        pick_dist_now = torch.linalg.vector_norm(self.agent_pos - self.object_pos, dim=-1)
        close_attempt = (~prev_grip) & gripper_close_cmd
        success_close = close_attempt & (pick_dist_now <= self.pick_radius)
        new_grip = (prev_grip & gripper_close_cmd) | success_close

        # On successful close, snap object to agent at end-of-step
        self.object_pos = torch.where(success_close.unsqueeze(1), self.agent_pos, self.object_pos)
        self.object_vel = torch.where(success_close.unsqueeze(1), self.agent_vel, self.object_vel)

        # Opening is based on realized gripper state
        opening = prev_grip & (~new_grip)
        self.delivered = self.delivered | (opening & (drop_dist_now <= self.drop_radius))

        # Update attachment flags
        self.gripper_closed = new_grip
        self.picked = new_grip

        # Unified task: success only when delivered (pick + place)
        self.success = self.delivered.clone()

        if self._record:
            self._frames.append(self.current_frame())

    # ---- rendering ----
    def current_frame(self) -> torch.Tensor:
        B = self.B
        # Start with a batched background frame [B,H,W,3]
        # Build frames on device in float for blending
        frames = self.bg_color.to(self.device).to(torch.float32).view(1, 1, 1, 3).expand(B, self.H, self.W, 3).clone()
        ag_r = self._world_radius_to_px(self.agent_radius_px)

        half_w = float(self.sprite_w) / 2.0
        half_h = float(self.sprite_h) / 2.0

        # ---- Fixed distractor sprites per env (0..max_distractors) ----
        if self.max_distractors > 0 and self.distractor_ids is not None:
            K = self.max_distractors
            for k in range(K):
                didx = self.distractor_ids[:, k]
                cx_d = (self.distractor_pos[:, k, 0].view(B, 1, 1) * float(self.W - 1))
                cy_d = (self.distractor_pos[:, k, 1].view(B, 1, 1) * float(self.H - 1))
                dx_d = self.pix_x - cx_d
                dy_d = self.pix_y - cy_d
                inside_d = (dx_d.abs() <= half_w) & (dy_d.abs() <= half_h)
                sx_d = torch.clamp((dx_d + half_w).floor().to(torch.long), 0, self.sprite_w - 1)
                sy_d = torch.clamp((dy_d + half_h).floor().to(torch.long), 0, self.sprite_h - 1)
                idx_lin_d = (sy_d * self.sprite_w + sx_d).view(B, -1, 1)
                dflat = self.sprites_flat4.index_select(0, didx)  # [B,h*w,4]
                dsel = torch.gather(dflat, 1, idx_lin_d.expand(-1, -1, 4)).view(B, self.H, self.W, 4)
                dsel = dsel * (inside_d).unsqueeze(-1).to(dsel.dtype)
                alpha_d = dsel[..., 3:4]
                rgb_d = dsel[..., :3]
                frames = rgb_d * alpha_d + frames * (1.0 - alpha_d)

        # ---- Batched goal sprite compositing (same sprite across batch) ----
        cx_g = (self.goal_pos[:, 0].view(B, 1, 1) * float(self.W - 1))
        cy_g = (self.goal_pos[:, 1].view(B, 1, 1) * float(self.H - 1))
        dx_g = self.pix_x - cx_g
        dy_g = self.pix_y - cy_g
        inside_g = (dx_g.abs() <= half_w) & (dy_g.abs() <= half_h)
        sx_g = torch.clamp((dx_g + half_w).floor().to(torch.long), 0, self.sprite_w - 1)
        sy_g = torch.clamp((dy_g + half_h).floor().to(torch.long), 0, self.sprite_h - 1)
        idx_lin_g = (sy_g * self.sprite_w + sx_g).view(B, -1, 1)  # [B,H*W,1]
        goal_flat = self.goal_sprite_flat4.unsqueeze(0).expand(B, -1, -1)  # [B,h*w,4]
        goal_sel = torch.gather(goal_flat, 1, idx_lin_g.expand(-1, -1, 4)).view(B, self.H, self.W, 4)
        goal_sel = goal_sel * inside_g.unsqueeze(-1).to(goal_sel.dtype)
        alpha_g = goal_sel[..., 3:4]
        rgb_g = goal_sel[..., :3]
        frames = rgb_g * alpha_g + frames * (1.0 - alpha_g)

        # ---- Batched object sprite compositing (sprite differs per batch) ----
        cx_o = (self.object_pos[:, 0].view(B, 1, 1) * float(self.W - 1))
        cy_o = (self.object_pos[:, 1].view(B, 1, 1) * float(self.H - 1))
        dx_o = self.pix_x - cx_o
        dy_o = self.pix_y - cy_o
        inside_o = (dx_o.abs() <= half_w) & (dy_o.abs() <= half_h)
        sx_o = torch.clamp((dx_o + half_w).floor().to(torch.long), 0, self.sprite_w - 1)
        sy_o = torch.clamp((dy_o + half_h).floor().to(torch.long), 0, self.sprite_h - 1)
        idx_lin_o = (sy_o * self.sprite_w + sx_o).view(B, -1, 1)
        objs_flat = self.sprites_flat4.index_select(0, self.object_id)  # [B,h*w,4]
        obj_sel = torch.gather(objs_flat, 1, idx_lin_o.expand(-1, -1, 4)).view(B, self.H, self.W, 4)
        obj_sel = obj_sel * inside_o.unsqueeze(-1).to(obj_sel.dtype)
        alpha_o = obj_sel[..., 3:4]
        rgb_o = obj_sel[..., :3]
        frames = rgb_o * alpha_o + frames * (1.0 - alpha_o)

        # Vectorized agent circle overlay for all batch elements (on device)
        ax = self.agent_pos[:, 0].view(B, 1, 1)
        ay = self.agent_pos[:, 1].view(B, 1, 1)
        mask_agent = (self.grid_x.view(1, self.H, self.W) - ax) ** 2 + (self.grid_y.view(1, self.H, self.W) - ay) ** 2 <= ag_r ** 2
        frames[mask_agent] = self.agent_color.to(self.device).to(torch.float32)

        # Gripper dot: concentric small blue dot when closed
        dot_mask = (
            (self.grid_x.view(1, self.H, self.W) - ax) ** 2
            + (self.grid_y.view(1, self.H, self.W) - ay) ** 2
            <= (0.5 * ag_r) ** 2
        ) & self.gripper_closed.view(B, 1, 1)
        frames[dot_mask] = torch.tensor([0, 0, 255], device=self.device, dtype=torch.float32)

        # Keep frames on device; save_video handles CPU move
        return frames.clamp(0, 255).to(torch.uint8)

    def _world_radius_to_px(self, radius_px: int) -> torch.Tensor:
        px_to_world = 1.0 / float(max(self.H, self.W) - 1)
        return torch.tensor(radius_px * px_to_world, device=self.device, dtype=self.dtype)

    def command(self) -> str:
        # Assumes batch size 1 usage for commands
        oid = int(self.object_id[0].item())
        obj_name = self.sprites[oid]["name"]
        goal_name = self.sprites[int(self.goal_id)]["name"]
        return f"place the {obj_name} on the {goal_name}"

    def save_video(self, path: str):
        fps = int(1.0 / float(self.dt))
        assert len(self._frames) > 0, "No frames recorded. Call reset(record_video=True) and step()."

        frames = torch.stack(self._frames, dim=0).to(torch.uint8)  # [T,B,H,W,3]
        T, B, H, W, C = frames.shape
        video = frames.permute(1, 0, 2, 3, 4).reshape(T * B, H, W, C).to("cpu")
        write_video(filename=path, video_array=video, fps=fps)

        sidecar = {
            "video_path": path,
            "fps": int(fps),
            "batch_size": int(self.B),
            "command": self.command(),
        }
        sidecar_path = os.path.splitext(path)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

    # ---- helpers ----
    def _load_sprites(self, directory: Path, size_px: int):
        sprites = []
        for p in sorted(directory.glob("*.png")):
            img = Image.open(p).convert("RGBA").resize((size_px, size_px), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)
            rgba_u8 = torch.from_numpy(arr).to(torch.uint8)
            # Precompute float RGB (0..255) and alpha (0..1) on device for fast blending
            rgb_f = rgba_u8[..., :3].to(torch.float32).to(self.device)
            alpha_f = (rgba_u8[..., 3].to(torch.float32) / 255.0).to(self.device)
            flat4 = torch.cat([rgb_f, alpha_f.unsqueeze(-1)], dim=-1).view(-1, 4)  # [h*w,4]
            sprites.append({
                "name": p.stem,
                "path": str(p),
                "flat4": flat4,
            })
        return sprites

    def _load_sprite_file(self, path: str | Path, size_px: int):
        p = Path(path)
        img = Image.open(p).convert("RGBA").resize((size_px, size_px), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        rgba_u8 = torch.from_numpy(arr).to(torch.uint8)
        rgb_f = rgba_u8[..., :3].to(torch.float32).to(self.device)
        alpha_f = (rgba_u8[..., 3].to(torch.float32) / 255.0).to(self.device)
        flat4 = torch.cat([rgb_f, alpha_f.unsqueeze(-1)], dim=-1).view(-1, 4)
        return {"name": p.stem, "path": str(p), "flat4": flat4}

    def get_env_state(self):
        state = {
            "agent_pos": self.agent_pos.cpu(),
            "agent_vel": self.agent_vel.cpu(),
            "object_pos": self.object_pos.cpu(),
            "object_vel": self.object_vel.cpu(),
            "goal_pos": self.goal_pos.cpu(),
            "object_id": self.object_id.cpu(),
            "picked": self.picked.cpu(),
            "delivered": self.delivered.cpu(),
            "success": self.success.cpu(),
            "gripper_closed": self.gripper_closed.cpu(),
            "goal_id": torch.tensor(self.goal_id),
        }
        if self.distractor_ids is not None:
            state["distractor_ids"] = self.distractor_ids.cpu()
            state["distractor_pos"] = self.distractor_pos.cpu()
        return state

    def set_env_state(self, state):
        self.agent_pos = state["agent_pos"].to(device=self.device, dtype=self.dtype)
        self.agent_vel = state["agent_vel"].to(device=self.device, dtype=self.dtype)
        self.object_pos = state["object_pos"].to(device=self.device, dtype=self.dtype)
        self.object_vel = state["object_vel"].to(device=self.device, dtype=self.dtype)
        self.goal_pos = state["goal_pos"].to(device=self.device, dtype=self.dtype)
        self.object_id = state["object_id"].to(device=self.device, dtype=torch.long)
        self.picked = state["picked"].to(device=self.device, dtype=torch.bool)
        self.delivered = state["delivered"].to(device=self.device, dtype=torch.bool)
        self.success = state["success"].to(device=self.device, dtype=torch.bool)
        self.gripper_closed = state["gripper_closed"].to(device=self.device, dtype=torch.bool)
        self.goal_id = int(state["goal_id"].item())
        self.goal_sprite_flat4 = self.sprites[self.goal_id]["flat4"]
        if "distractor_ids" in state:
            self.distractor_ids = state["distractor_ids"].to(device=self.device, dtype=torch.long)
            self.distractor_pos = state["distractor_pos"].to(device=self.device, dtype=self.dtype)
        else:
            self.distractor_ids = None
            self.distractor_pos = None
