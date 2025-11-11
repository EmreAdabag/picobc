import argparse
import os
import torch

from env import PickAndPlaceEnv


class ExpertController:
    """Simple PD controller for unified pick+place."""

    def __init__(self):
        # task_type kept for backward compatibility but ignored
        self.kp = 14.0
        self.kd = 6.0
        self.kp_loaded = 14.0
        self.kd_loaded = 12.0
        self.u_clip = 4.0

    def act(self, env: PickAndPlaceEnv) -> torch.Tensor:
        # Target: go to object until picked, then go to goal
        target = torch.where((~env.picked).unsqueeze(-1), env.object_pos, env.goal_pos)
        kp = torch.where(env.picked, self.kp_loaded, self.kp).unsqueeze(1)
        kd = torch.where(env.picked, self.kd_loaded, self.kd).unsqueeze(1)
        pos = env.agent_pos
        vel = env.agent_vel
        u = kp * (target - pos) - kd * vel
        u = torch.clamp(u, min=-self.u_clip, max=self.u_clip)

        # Gripper schedule: close near object; hold while carrying; open near goal to drop
        dist_obj = torch.linalg.vector_norm(env.agent_pos - env.object_pos, dim=-1)
        dist_goal = torch.linalg.vector_norm(env.agent_pos - env.goal_pos, dim=-1)
        pick_thresh = 0.2 * float(env.pick_radius.item())
        drop_thresh = 0.2 * float(env.drop_radius.item())

        close_to_obj = (~env.picked) & (dist_obj <= pick_thresh) & (~env.delivered)
        carry_phase = env.picked & (~env.delivered)
        open_to_drop = carry_phase & (dist_goal <= drop_thresh)

        closed_mask = (close_to_obj | carry_phase) & (~open_to_drop)
        g = closed_mask.view(-1, 1).to(env.dtype)

        return torch.cat([u, g], dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Roll out expert policy and save a video")
    parser.add_argument("--out", type=str, default="expert_rollout.mp4", help="Output video path")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (episodes in parallel)")
    parser.add_argument("--timeout_s", type=float, default=10.0, help="Max episode time in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PickAndPlaceEnv(batch_size=1, dt=float(args.dt), device=device, record_video=True)
    ctrl = ExpertController()

    env.reset()
    print(f"command: {env.command()}")
    max_steps = int(float(args.timeout_s) / float(env.dt))
    steps = 0
    while steps < max_steps:
        act = ctrl.act(env)
        env.step(act)
        if bool(env.success.all()):
            break
        steps += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env.save_video(args.out)

    print(f"Saved video to {args.out}")
    print(f"Success flags: {env.success.tolist()} in {steps+1} steps")


if __name__ == "__main__":
    main()
