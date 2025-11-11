import argparse
from pathlib import Path
import csv
import numpy as np
import torch

from env import PickAndPlaceEnv

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.random_utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Roll out a SmolVLA policy in PickAndPlaceEnv")
    parser.add_argument("--model_path", type=str, default="model/pretrained_model")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save_video", action='store_true')
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--model_type", type=str, default='smolvla')
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--save_prefix", type=str, default='', help="")
    parser.add_argument("--obj_lo", type=int, default=0)
    parser.add_argument("--obj_hi", type=int, default=2)
    parser.add_argument("--csv_path", type=str, default="sim_eval_results.csv")
    args = parser.parse_args()
    print(args.save_video)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

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

    max_steps = int(args.max_seconds / float(args.dt))
    kp, kd = 200.0, 3.0
    u_clip = 4.0

    obj_lo = int(args.obj_lo)
    obj_hi = int(args.obj_hi)
    goal_lo = obj_lo
    goal_hi = obj_hi

    header = ["object_id", "goal_id"] + [f"discrete_{i}" for i in range(args.episodes)] + [f"continuous_{i}" for i in range(args.episodes)]
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

    sqrt2 = np.sqrt(2.0)
    for oid in range(obj_lo, obj_hi):
        for gid in range(goal_lo, goal_hi):
            if oid == gid:
                continue
            env = PickAndPlaceEnv(batch_size=args.episodes, device=device, record_video=args.save_video, dt=args.dt, seed=args.seed)
            env.reset(obj_idx=oid, goal_idx=gid)
            policy.reset()
            task_str = env.command()
            tasks = [task_str for _ in range(args.episodes)]

            # Track initial distances and success events for continuous scoring
            init_agent = env.agent_pos.clone()
            init_object = env.object_pos.clone()
            init_goal = env.goal_pos.clone()
            ever_picked = torch.zeros(env.B, dtype=torch.bool, device=env.device)
            ever_delivered = torch.zeros(env.B, dtype=torch.bool, device=env.device)

            for _ in range(max_steps):
                frames = env.current_frame().to(torch.float32) / 255.0   # [B,H,W,3]
                frames = frames.permute(0, 3, 1, 2).contiguous()         # [B,3,H,W]
                rob_pos = env.agent_pos.to(torch.float32)                # [B,2]
                g_abs = env.gripper_closed.to(torch.float32).unsqueeze(-1)  # [B,1]
                robot_state = torch.cat([rob_pos, g_abs], dim=-1)        # [B,3]

                batch = {
                    "observation.images.camera1": frames,
                    "observation.state": robot_state,
                    "task": tasks,
                }

                proc = preprocessor(batch)
                action = policy.select_action(proc)
                action = postprocessor(action)                           # [B,3] on CPU
                action = action.to(torch.float32).to(env.device)

                dpos = action[..., :2]
                grip = action[..., 2:3].clamp(0.0, 1.0)

                u = kp * dpos - kd * env.agent_vel
                u = torch.clamp(u, -u_clip, u_clip)
                env.step(torch.cat([u, grip], dim=-1))

                # Update event trackers
                ever_picked = ever_picked | env.gripper_closed
                ever_delivered = ever_delivered | env.delivered

                if env.delivered.all():
                    break

            delivered = env.delivered.to(torch.float32).to("cpu")
            # Discrete score: 1.0 if delivered, else 0.0
            disc = delivered                                            # [B]

            # Continuous score: up to 0.5 for pick, 0.5 for place
            ag_T = env.agent_pos.to(torch.float32)
            obj_T = env.object_pos.to(torch.float32)
            goal_T = env.goal_pos.to(torch.float32)

            d_pick0 = torch.linalg.vector_norm(init_agent - init_object, dim=-1) / sqrt2
            d_pickT = torch.linalg.vector_norm(ag_T - obj_T, dim=-1) / sqrt2
            d_place0 = torch.linalg.vector_norm(init_object - init_goal, dim=-1) / sqrt2
            d_placeT = torch.linalg.vector_norm(obj_T - goal_T, dim=-1) / sqrt2

            eps = 1e-6
            # Pick half: full if ever picked, else proportional distance reduction
            ever_picked_f = ever_picked.to(torch.float32)
            pick_improvement = torch.clamp((d_pick0 - d_pickT) / torch.clamp(d_pick0, min=eps), 0.0, 1.0)
            half1 = torch.where(ever_picked_f > 0.5, torch.full_like(d_pickT, 0.5), 0.5 * pick_improvement)

            # Place half: full if ever delivered, else proportional reduction
            ever_delivered_f = ever_delivered.to(torch.float32)
            place_improvement = torch.clamp((d_place0 - d_placeT) / torch.clamp(d_place0, min=eps), 0.0, 1.0)
            half2 = torch.where(ever_delivered_f > 0.5, torch.full_like(d_placeT, 0.5), 0.5 * place_improvement)

            cont = (half1 + half2).to("cpu")                              # [B]

            discrete_scores = [float(x) for x in disc.tolist()]
            continuous_scores = [float(x) for x in cont.tolist()]

            if args.save_video is True:
                out_dir = Path(args.model_path) / "eval_videos"
                out_dir.mkdir(parents=True, exist_ok=True)
                env.save_video(str(out_dir / f"{args.save_prefix}rollout_obj{oid}_goal{gid}.mp4"))

            row = [oid, gid] + discrete_scores + continuous_scores
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(row)
            print(env.picked, env.delivered)
            print(cont)
            disc_success_rate = float(delivered.mean().item()) * 100.0
            mean_cont = float(cont.mean().item())
            print(f"object: {oid} goal: {gid} discrete success rate: {disc_success_rate:.1f}% mean continuous: {mean_cont:.3f}")


if __name__ == "__main__":
    main()
