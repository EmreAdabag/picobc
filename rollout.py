import argparse
import os
import json
import torch
from torchvision.io import write_video
from bc_model import BCVisionModel, BCStateModel, normalize, unnormalize
from env import PickAndPlaceEnv

u_clip = 4.0
kp, kd = 200.0, 4.0

@torch.no_grad()
def rollout(model, episodes: int = 2, out_dir: str = ".", timeout_s: float = 5., render_video: bool = True, mode: str = 'vision'):
    device = next(model.parameters()).device
    assert mode in ['state', 'vision']

    os.makedirs(out_dir, exist_ok=True)

    B = int(episodes)
    env = PickAndPlaceEnv(seed=123, batch_size=B)
    max_steps = int(timeout_s / float(env.dt))
    env.reset(record_video=render_video)
    env.step(torch.zeros(B, 2))

    done_step = torch.full((B,), -1, dtype=torch.long)
    for t in range(max_steps):
        state = env.agent_pos.to(device)                            # [B,2]
        cmd = env.command.to(device)                                # [B,2]
        state_norm = normalize(state, model.state_min, model.state_max)

        if mode == 'vision':
            frame = env.current_frame().to(torch.uint8)                 # [B,H,W,3]
            img = frame.permute(0, 3, 1, 2).float().div_(255.0).to(device)  # [B,3,H,W]
            dpos_n = model(img, state_norm, cmd)                        # [B,2] (normalized)
        else:
            obj_pos = torch.stack([env.objects_pos[b, int(env._target_object_idx[b].item())] for b in range(B)]).to(device)
            goal_pos = torch.stack([env.goals_center[b, int(env._target_goal_idx[b].item())] for b in range(B)]).to(device)
            obs = torch.cat([state_norm, obj_pos, goal_pos, cmd], dim=-1)
            dpos_n = model(obs)

        dpos = unnormalize(dpos_n, model.dpos_min, model.dpos_max).to("cpu")
        u = kp * dpos - kd * env.agent_vel                          # [B,2]
        u = torch.clamp(u, -u_clip, u_clip)

        out = env.step(u)
        newly_done = (done_step < 0) & out["success"].to("cpu")
        done_step[newly_done] = t
        if bool(out["success"].all()):
            break

    successes = int((done_step >= 0).sum().item())
    print(f"successes: {successes}/{episodes}")

    if render_video and env._frames:
        frames5 = torch.stack(env._frames, dim=0).to(torch.uint8)  # [T,B,H,W,3]
        frames5 = env._annotate_frames_with_command(frames5)
        T, Bv, H, W, C = frames5.shape
        video = frames5.permute(1, 0, 2, 3, 4).reshape(T * Bv, H, W, C).to("cpu")
        out_video = os.path.join(out_dir, "bc_rollout.mp4")
        fps = int(1.0 / float(env.dt))
        write_video(filename=out_video, video_array=video, fps=fps)

        episodes_meta = []
        for b in range(B):
            steps_b = int(done_step[b].item()) + 1 if int(done_step[b].item()) >= 0 else int(t)
            episodes_meta.append({
                "episode": int(b),
                "success": bool(done_step[b].item() >= 0),
                "steps": steps_b,
                "command": env.command_info(b),
            })

        sidecar = {
            "video_path": out_video,
            "fps": fps,
            "episodes": episodes_meta,
            "total_episodes": int(episodes),
            "total_successes": int(successes),
        }
        sidecar_path = os.path.splitext(out_video)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

    return successes

@torch.no_grad()
def batch_eval(model, episodes: int = 100, timeout_s: float = 10., mode: str = 'vision'):
    device = next(model.parameters()).device
    assert mode in ['state', 'vision']

    B = int(episodes)
    env = PickAndPlaceEnv(seed=123, batch_size=B)
    max_steps = int(timeout_s / float(env.dt))
    env.reset()
    env.step(torch.zeros(B, 2))

    for _ in range(max_steps):
        state = env.agent_pos.to(device)  # [B,2]
        cmd = env.command.to(device)      # [B,2]
        state_norm = normalize(state, model.state_min, model.state_max)

        if mode == 'vision':
            frame = env.current_frame().to(torch.uint8)  # [B,H,W,3]
            img = frame.permute(0, 3, 1, 2).float().div_(255.0).to(device)  # [B,3,H,W]
            dpos_n = model(img, state_norm, cmd)
        else:
            obj_pos = torch.stack([env.objects_pos[b, int(env._target_object_idx[b].item())] for b in range(B)]).to(device)
            goal_pos = torch.stack([env.goals_center[b, int(env._target_goal_idx[b].item())] for b in range(B)]).to(device)
            obs = torch.cat([state_norm, obj_pos, goal_pos, cmd], dim=-1)
            dpos_n = model(obs)

        dpos = unnormalize(dpos_n, model.dpos_min, model.dpos_max).to("cpu")
        u = kp * dpos - kd * env.agent_vel       # [B,2]
        u = torch.clamp(u, -u_clip, u_clip)

        out = env.step(u)
        if bool(out["success"].all()):
            break

    successes = int(env.success.sum().item())
    print(f"successes: {successes}/{episodes}")
    return successes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--no-video", dest="no_video", action="store_true", help="Disable video recording and saving")
    parser.add_argument("--mode", type=str, default='vision', choices=['state', 'vision'], help='Model mode: state or vision')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (BCVisionModel() if args.mode == 'vision' else BCStateModel()).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    rollout(model, episodes=args.episodes, out_dir=args.out_dir, render_video=(not args.no_video), timeout_s=5., mode=args.mode)
