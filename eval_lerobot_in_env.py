import argparse
from pathlib import Path
import numpy as np
import torch

from env import PickAndPlaceEnv

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.control_utils import predict_action


def main():
    parser = argparse.ArgumentParser(description="Roll out a SmolVLA policy in PickAndPlaceEnv")
    parser.add_argument("--model_path", type=str, default="model/pretrained_model")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--task", type=str, default="place", choices=["pick", "place"])
    parser.add_argument("--save_video", type=bool, default=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--object_id", type=int, default=None)
    parser.add_argument("--model_type", type=str, default='smolvla')
    parser.add_argument("--save_prefix", type=str, default='', help="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    cfg.device = str(device)

    if args.model_type=='smolvla':
        policy = SmolVLAPolicy.from_pretrained(args.model_path, config=cfg)
    else:
        policy = DiffusionPolicy.from_pretrained(args.model_path, config=cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    env = PickAndPlaceEnv(batch_size=1, task_type=args.task, device=device, record_video=args.save_video, dt=args.dt, seed=123456)

    max_steps = int(args.max_seconds / float(env.dt))
    kp, kd = 500.0, 6.0
    u_clip = 4.0

    for ep in range(args.episodes):
        # Reset env and policy queues each episode
        env.reset(task_type=args.task, obj_idx=args.object_id)
        policy.reset()
        task_str = env.command()
        print(task_str)
        for _ in range(max_steps):
            # Keep raw uint8; preprocessors handle scaling/normalization
            frame = env.current_frame().to("cpu").numpy()[0]

            # Observation state matches dataset: [x, y, gripper]
            rob_pos = env.agent_pos[0].to(torch.float32).to("cpu").numpy().astype(np.float32)
            g_abs = np.float32(float(env.gripper_closed[0].item()))
            robot_state = np.array([rob_pos[0], rob_pos[1], g_abs], dtype=np.float32)

            observation = {
                "observation.images.camera1": frame,
                "observation.state": robot_state,
            }

            action = predict_action(
                observation=observation,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task_str,
                use_amp=bool(policy.config.use_amp),
            )
            print(f'action: {action}')
            action = action.squeeze(0).to(torch.float32).to(env.device)

            # Action format matches dataset: [ux, uy, gripper]
            dpos = action[..., :2]
            grip = action[..., 2:].clamp(0.0, 1.0)

            u = kp * dpos - kd * env.agent_vel[0]
            u = torch.clamp(u, -u_clip, u_clip).unsqueeze(0)

            env.step(torch.cat([u, grip.unsqueeze(0)], dim=-1))
            if bool(env.success[0].item()):
                print(f"SUCCESS")
                break

        if args.save_video:
            out_dir = Path(args.model_path) / "eval_videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            env.save_video(str(out_dir / f"{args.save_prefix}rollout_{ep}.mp4"))
            print(f"Saved video: {out_dir / f'{args.save_prefix}rollout_{ep}.mp4'}")


if __name__ == "__main__":
    main()
