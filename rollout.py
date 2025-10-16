import argparse
import os
import torch
from bc_model import BCModel
from env import PickAndPlaceEnv


@torch.no_grad()
def rollout(model, episodes: int = 2, out_dir: str = ".", max_steps: int = 300, render_video: bool = True, device=None):
    # Preserve caller's mode and ensure restoration after eval
    prev_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"log.txt")

    successes = 0
    env = PickAndPlaceEnv(seed=123)
    for ep in range(episodes):
        env.reset(record_video=render_video)
        # Align image/state timing with training (frames were captured post-step)
        env.step(torch.zeros(2))

        # dt = float(env.dt.item())
        u_clip = 5.0
        # Simple PD gains (minimal, fixed)
        kp, kd = 30.0, 8.0

        for t in range(max_steps):
            # Prepare image and state
            frame = env.current_frame().to(torch.uint8)  # HxWx3 uint8 torch tensor
            img = frame.unsqueeze(0).permute(0, 3, 1, 2).float().div_(255.0).to(device)  # 1x3xHxW
            state = env.agent_pos.view(1, 2).to(device)

            # Predict next-position delta
            dpos = model(img, state).squeeze(0).to("cpu")  # 2
            # PD control towards target position: pos + dpos
            pos_err = dpos  # target_pos - pos = (pos + dpos) - pos
            vel = env.agent_vel
            # PD: spring towards target and damping on velocity
            u = kp * pos_err - kd * vel
            u = torch.clamp(u, -u_clip, u_clip)

            out = env.step(u)
            if out["success"]:
                successes += 1
                break

        # Optionally save video
        if render_video:
            print(f"episode {ep}: success={env.success} steps={t+1}")
            video_path = os.path.join(out_dir, f"bc_episode_{ep}.mp4")
            env.save_video(video_path, fps=30)
        with open(log_path, 'a') as f:
            f.write(f"episode {ep}: success={env.success} steps={t+1}\n")

    with open(log_path, 'a') as f:
        f.write(f"successes: {successes}/{episodes}\n")
    print(f"successes: {successes}/{episodes}")
    # Restore original mode
    if prev_training:
        model.train()
    else:
        model.eval()
    return successes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--no-video", dest="no_video", action="store_true", help="Disable video recording and saving")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel().to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)

    rollout(model, episodes=args.episodes, out_dir=args.out_dir, render_video=(not args.no_video), device=device)
