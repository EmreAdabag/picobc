import os
import json
import torch
from tqdm import tqdm
from torchvision.io import write_video

class PickAndPlaceEnv:
    """
    Minimal 2D pick-and-place environment in torch.

    - Agent: colored point mass in R^2 commands force u in R^2.
    - Object: colored circle; picked after lingering near its center.
    - Goal: colored square region; drop after lingering near its center.
    - Linear dynamics: x_{t+1} = x_t + v_t*dt ; v_{t+1} = v_t + u*dt.
    - Optional video rendering via torchvision.io.write_video.

    This is intentionally minimal: fixed sizes/thresholds, no rewards, no extras.
    """

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32, seed: int = 0, exclude_combo: tuple[int, int] | None = (1, 1)):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng = torch.Generator(device="cpu").manual_seed(int(seed))
        # Optionally exclude a specific (object_color_idx, goal_color_idx) combination
        # from ever appearing. Set to None to allow all combinations.
        self.exclude_combo = exclude_combo

        # World and physics constants (fixed; minimal, no configuration)
        self.dt = torch.tensor(0.0025, device=self.device, dtype=self.dtype)
        self.mass = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        # Geometry (fixed)
        self.agent_color = torch.tensor([230, 57, 70], device=self.device, dtype=torch.uint8)  # red (agent)
        # Two object colors and two goal colors
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
        self.bg_color = torch.tensor([245, 245, 245], device=self.device, dtype=torch.uint8)  # near white

        self.agent_radius_px = 3
        self.object_radius_px = 6
        self.goal_half_size = torch.tensor(0.07, device=self.device, dtype=self.dtype)  # half side length (world)

        # Interaction thresholds (fixed)
        self.pick_radius = torch.tensor(0.04, device=self.device, dtype=self.dtype)
        self.drop_radius = torch.tensor(0.06, device=self.device, dtype=self.dtype)
        self.pick_linger_steps = 10
        self.drop_linger_steps = 10

        # Rendering settings (fixed dimensions; minimal)
        self.H = 128
        self.W = 128
        # Precompute a 2D grid in world coords [0,1]x[0,1]
        ys = torch.linspace(0.0, 1.0, self.H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(0.0, 1.0, self.W, device=self.device, dtype=self.dtype)
        self.grid_y, self.grid_x = torch.meshgrid(ys, xs, indexing="ij")  # HxW each

        # State placeholders
        self.agent_pos = None
        self.agent_vel = None
        # Multiple objects and multiple goals
        # Stored as shape (2, 2): index 0/1 for color A/B, last dim x,y
        self.objects_pos = None
        self.objects_vel = None
        self.goals_center = None

        # Convenience alias to the currently commanded target object and goal
        self.object_pos = None
        self.object_vel = None
        self.goal_center = None
        self.picked = False
        self.delivered = False
        self.success = False
        self._pick_linger = 0
        self._drop_linger = 0

        # Command: a 2-vector [object_color, goal_color] with continuous values (here 0.0 or 1.0)
        self.command = torch.zeros(2, device=self.device, dtype=self.dtype)
        self._target_object_idx = 0
        self._target_goal_idx = 0

        # Rendering buffer
        self._frames = []  # list of HxWx3 uint8 tensors
        self._record = False

    def reset(self, record_video: bool = False):
        """Reset environment to a simple fixed initial state.

        record_video: if True, collect frames at each step for later saving.
        """
        # Random starts in [0.1, 0.9]^2 (reproducible with seed)
        def rnd2():
            # Uniform in [0,1]^2 to fully sweep state space up to the walls
            return torch.rand(2, generator=self.rng)

        self.agent_pos = rnd2().to(device=self.device, dtype=self.dtype)
        self.agent_vel = torch.zeros(2, device=self.device, dtype=self.dtype)
        # Two objects and two goals
        self.objects_pos = torch.stack([rnd2(), rnd2()], dim=0).to(device=self.device, dtype=self.dtype)  # (2,2)
        self.objects_vel = torch.zeros(2, 2, device=self.device, dtype=self.dtype)
        self.goals_center = torch.stack([rnd2(), rnd2()], dim=0).to(device=self.device, dtype=self.dtype)  # (2,2)

        # Random command: choose one of each color (continuous scalars 0.0 or 1.0), resample if excluded
        while True:
            self._target_object_idx = int(torch.randint(0, 2, (1,), generator=self.rng).item())
            self._target_goal_idx = int(torch.randint(0, 2, (1,), generator=self.rng).item())
            if self.exclude_combo is None or (self._target_object_idx, self._target_goal_idx) != self.exclude_combo:
                break
        self.command = torch.tensor([float(self._target_object_idx), float(self._target_goal_idx)], device=self.device, dtype=self.dtype)

        # Expose aliases for target object / goal (backward-compat)
        self.object_pos = self.objects_pos[self._target_object_idx]
        self.object_vel = self.objects_vel[self._target_object_idx]
        self.goal_center = self.goals_center[self._target_goal_idx]

        self.picked = False
        self.delivered = False
        self.success = False
        self._pick_linger = 0
        self._drop_linger = 0

        self._record = bool(record_video)
        self._frames = []

        if self._record:
            self._frames.append(self._render_frame())

        return self._obs()

    def step(self, force: torch.Tensor):
        """Advance dynamics by one step using applied force (R^2).

        Returns a dict with observation and flags.
        """
        u = force.to(device=self.device, dtype=self.dtype).view(2)

        # Agent linear dynamics
        self.agent_vel = self.agent_vel + (u / self.mass) * self.dt
        self.agent_pos = self.agent_pos + self.agent_vel * self.dt
        self.agent_pos = self.agent_pos.clamp(0.0, 1.0)

        # Object follows only if picked (rigidly attached) — update only the commanded object
        if self.picked and not self.delivered:
            self.objects_pos[self._target_object_idx] = self.agent_pos.clone()
            self.objects_vel[self._target_object_idx] = self.agent_vel.clone()
            # Keep aliases up to date
            self.object_pos = self.objects_pos[self._target_object_idx]
            self.object_vel = self.objects_vel[self._target_object_idx]

        # Check picking condition if not yet picked
        if not self.picked:
            # Only picking the commanded object counts
            if torch.linalg.vector_norm(self.agent_pos - self.objects_pos[self._target_object_idx]) <= self.pick_radius:
                self._pick_linger += 1
            else:
                self._pick_linger = 0
            if self._pick_linger >= self.pick_linger_steps:
                self.picked = True

        # Check dropping condition if picked but not delivered
        if self.picked and not self.delivered:
            # Must deliver to the commanded goal
            if torch.linalg.vector_norm(self.agent_pos - self.goals_center[self._target_goal_idx]) <= self.drop_radius:
                self._drop_linger += 1
            else:
                self._drop_linger = 0
            if self._drop_linger >= self.drop_linger_steps:
                # Drop object at current position (within goal)
                self.delivered = True
                self.success = True

        obs = self._obs()

        if self._record:
            self._frames.append(self._render_frame())

        return {
            "obs": obs,
            "picked": self.picked,
            "delivered": self.delivered,
            "success": self.success,
        }

    def _obs(self):
        # Concatenate state into a flat tensor (minimal observation)
        # Expose target object and goal along with continuous command for clarity
        return torch.cat([
            self.agent_pos,
            self.agent_vel,
            self.objects_pos[self._target_object_idx],
            self.objects_vel[self._target_object_idx],
            self.goals_center[self._target_goal_idx],
            self.command.to(self.dtype),
            torch.tensor([float(self.picked), float(self.delivered)], device=self.device, dtype=self.dtype),
        ])

    def _render_frame(self) -> torch.Tensor:
        """Rasterize a single frame as HxWx3 uint8 torch tensor."""
        frame = self.bg_color.view(1, 1, 3).expand(self.H, self.W, 3).clone()

        # Draw two goal squares
        half = self.goal_half_size
        for gi in range(2):
            gx, gy = self.goals_center[gi][0], self.goals_center[gi][1]
            mask_goal = (torch.abs(self.grid_x - gx) <= half) & (torch.abs(self.grid_y - gy) <= half)
            frame[mask_goal] = self.goal_colors[gi]

        # Draw two object circles
        obj_r = self._world_radius_to_px(self.object_radius_px)
        for oi in range(2):
            ox, oy = self.objects_pos[oi][0], self.objects_pos[oi][1]
            mask_obj = (self.grid_x - ox) ** 2 + (self.grid_y - oy) ** 2 <= obj_r ** 2
            frame[mask_obj] = self.object_colors[oi]

        # Draw agent small circle
        ax, ay = self.agent_pos[0], self.agent_pos[1]
        ag_r = self._world_radius_to_px(self.agent_radius_px)
        mask_agent = (self.grid_x - ax) ** 2 + (self.grid_y - ay) ** 2 <= ag_r ** 2
        frame[mask_agent] = self.agent_color

        return frame.to(torch.uint8)

    def _world_radius_to_px(self, radius_px: int) -> torch.Tensor:
        # Convert a pixel radius to an approximate world radius using image scale.
        # This keeps circles visually consistent with world units on the grid.
        # 1 pixel corresponds to 1/(max(H,W)-1) in world units (approx).
        px_to_world = 1.0 / float(max(self.H, self.W) - 1)
        return torch.tensor(radius_px * px_to_world, device=self.device, dtype=self.dtype)

    def current_frame(self) -> torch.Tensor:
        """Return a freshly rendered frame without recording."""
        return self._render_frame()

    def command_info(self) -> dict:
        """Return a dict describing the current commanded colors (indices, names, RGB)."""
        oi = int(self._target_object_idx)
        gi = int(self._target_goal_idx)
        return {
            "object_index": oi,
            "object_name": self.object_color_names[oi],
            "object_rgb": self.object_colors[oi].tolist(),
            "goal_index": gi,
            "goal_name": self.goal_color_names[gi],
            "goal_rgb": self.goal_colors[gi].tolist(),
            "command": [float(self.command[0].item()), float(self.command[1].item())],
        }

    def _annotate_frames_with_command(self, frames: torch.Tensor) -> torch.Tensor:
        """Overlay small color chips for commanded object/goal on each frame (in-place copy)."""
        if frames.ndim != 4:
            return frames
        T, H, W, C = frames.shape
        out = frames.clone()
        # Chip geometry
        pad = 2
        size = 12
        # object chip at top-left
        o_y0, o_y1 = pad, min(H, pad + size)
        o_x0, o_x1 = pad, min(W, pad + size)
        # goal chip to the right of object chip
        gap = 4
        g_y0, g_y1 = pad, min(H, pad + size)
        g_x0, g_x1 = min(W, o_x1 + gap), min(W, o_x1 + gap + size)

        info = self.command_info()
        o_col = torch.tensor(info["object_rgb"], dtype=torch.uint8).view(1, 1, 3)
        g_col = torch.tensor(info["goal_rgb"], dtype=torch.uint8).view(1, 1, 3)

        # Fill chips across all frames
        out[:, o_y0:o_y1, o_x0:o_x1, :] = o_col
        out[:, g_y0:g_y1, g_x0:g_x1, :] = g_col

        # White border for visibility
        border = 1
        white = torch.tensor([255, 255, 255], dtype=torch.uint8)
        if o_y1 - o_y0 > 2 and o_x1 - o_x0 > 2:
            out[:, o_y0:o_y0+border, o_x0:o_x1, :] = white
            out[:, o_y1-border:o_y1, o_x0:o_x1, :] = white
            out[:, o_y0:o_y1, o_x0:o_x0+border, :] = white
            out[:, o_y0:o_y1, o_x1-border:o_x1, :] = white
        if g_y1 - g_y0 > 2 and g_x1 - g_x0 > 2:
            out[:, g_y0:g_y0+border, g_x0:g_x1, :] = white
            out[:, g_y1-border:g_y1, g_x0:g_x1, :] = white
            out[:, g_y0:g_y1, g_x0:g_x0+border, :] = white
            out[:, g_y0:g_y1, g_x1-border:g_x1, :] = white

        return out

    def save_video(
        self,
        path: str,
    ):
        """Save recorded frames to an .mp4 video using torchvision.

        Requires `reset(record_video=True)` before stepping to collect frames.
        """
        fps = int(1 / self.dt)
        if not self._frames:
            raise RuntimeError("No frames recorded. Call reset(record_video=True) and step().")

        # Stack to T x H x W x C uint8 and convert to tensor on CPU for write_video
        frames = torch.stack(self._frames, dim=0).to(torch.uint8).to("cpu")
        frames = self._annotate_frames_with_command(frames)

        info = self.command_info()
        target_path = path
        # write_video expects CxHxW or HxWxC? In torchvision>=0.14, it expects T x H x W x C uint8
        write_video(filename=target_path, video_array=frames, fps=fps)

        sidecar = {
            "video_path": target_path,
            "fps": int(fps),
            "command": info,
        }
        sidecar_path = os.path.splitext(target_path)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)


class ExpertController:
    """
    Very simple PD controller using full state knowledge.
    Phases:
      1) Drive to object until picked.
      2) Drive to goal until delivered (success).
    """

    def __init__(self):
        # Fixed PD gains and clamp; minimal and not configurable
        self.kp = 30.0
        self.kd = 8.0
        self.u_clip = 5.0

    def act(self, env: PickAndPlaceEnv) -> torch.Tensor:
        # Choose target based on task phase
        if not env.picked:
            target = env.object_pos
        elif not env.delivered:
            target = env.goal_center
        else:
            # Done; no force needed
            return torch.zeros(2, device=env.device, dtype=env.dtype)

        pos = env.agent_pos
        vel = env.agent_vel

        u = self.kp * (target - pos) - self.kd * vel
        # Clamp for stability
        u = torch.clamp(u, min=-self.u_clip, max=self.u_clip)
        return u


def collect_expert_dataset(N: int, path: str):
    """
    Collect N expert demonstrations and save to an HDF5 file.

    Each episode group 'ep_i' contains:
      - images: T x H x W x 3 (uint8)
      - agent_pos: T x 2 (float32)
      - agent_vel: T x 2 (float32)
      - agent_ctrl: T x 2 (float32)
      - object_pos: T x 2 (float32)            (commanded object's trajectory)
      - goal_center: T x 2 (float32)           (commanded goal, constant per episode, repeated over T)
      - command: 2 (float32)                   ([object_color, goal_color] continuous values)
    """
    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("h5py is required for dataset export.") from e

    env = PickAndPlaceEnv()
    ctrl = ExpertController()

    total_samples = 0

    with h5py.File(path, "w") as f:
        for i in tqdm(range(int(N))):
            env.reset(record_video=False)

            frames = []
            poss = []
            vels = []
            ctrls = []
            obj_poss = []
            goal_centers = []

            while 1:
                u = ctrl.act(env)
                out = env.step(u)
                frames.append(env.current_frame().to("cpu"))
                poss.append(env.agent_pos.to("cpu"))
                vels.append(env.agent_vel.to("cpu"))
                ctrls.append(u.to("cpu"))
                obj_poss.append(env.object_pos.to("cpu"))
                goal_centers.append(env.goal_center.to("cpu"))
                if out["success"]:
                    break

            # Stack and write to HDF5
            frames_t = torch.stack(frames, dim=0).to(torch.uint8).numpy()
            poss_t = torch.stack(poss, dim=0).to(torch.float32).numpy()
            vels_t = torch.stack(vels, dim=0).to(torch.float32).numpy()
            ctrls_t = torch.stack(ctrls, dim=0).to(torch.float32).numpy()
            obj_poss_t = torch.stack(obj_poss, dim=0).to(torch.float32).numpy()
            goal_centers_t = torch.stack(goal_centers, dim=0).to(torch.float32).numpy()

            g = f.create_group(f"ep_{i}")
            g.create_dataset("images", data=frames_t, compression="gzip")
            g.create_dataset("agent_pos", data=poss_t)
            g.create_dataset("agent_vel", data=vels_t)
            g.create_dataset("agent_ctrl", data=ctrls_t)
            g.create_dataset("object_pos", data=obj_poss_t)
            g.create_dataset("goal_center", data=goal_centers_t)
            # Store episode command as a 2-vector [object_color, goal_color]
            g.create_dataset("command", data=env.command.to(torch.float32).cpu().numpy())

            total_samples += len(frames_t)
    
    print(f"wrote {total_samples * env.dt / 3600} hours of data from {N} demonstrations totalling {total_samples} datapoints")
