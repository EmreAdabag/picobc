import sys
import json as _json
from pathlib import Path
import numpy as np
import torch
from env import PickAndPlaceEnv
from expert_rollout import ExpertController
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse

def collect_lerobot_dataset(
    episodes_per_task: int,
    root: str | Path,
    tasks: list,
    seed: int = 0,
    max_distractors: int = 0,
    image_size: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bootstrap env to read rendering params
    dt = 0.01
    env_boot = PickAndPlaceEnv(device=device, record_video=False, dt=dt, image_size=image_size, max_distractors=max_distractors)
    fps = int(1.0 / float(env_boot.dt))
    H, W = int(env_boot.H), int(env_boot.W)

    robot_state_dim = 3

    # Common vector features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (robot_state_dim,),
            "names": [
                "agent.pos.x",
                "agent.pos.y",
                "gripper"
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["ux", "uy", "gripper"],
        },
    }

    # Vision-only, videos
    features["observation.images.camera1"] = {
        "dtype": "video",
        "shape": (H, W, 3),
        "names": ["height", "width", "channels"],
    }

    root = Path(root)
    ds = LeRobotDataset.create(
        repo_id='null',
        fps=fps,
        features=features,
        root=root,
        use_videos=True
    )

    max_steps = int(10.0 / float(env_boot.dt))
    episode_task_records = []

    for i, (obj_idx, goal_idx) in enumerate(tasks):
        print(f"\n\nstarting task {i}\n")

        env = PickAndPlaceEnv(device=device, record_video=False, dt=dt, seed=seed, image_size=image_size, max_distractors=max_distractors)

        for _ in range(episodes_per_task):
            # Use goal_idx to render the goal with another object sprite
            env.reset(obj_idx=obj_idx, goal_idx=goal_idx)
            task_str = env.command()

            ctrl = ExpertController()
                
            steps = 0
            steps_after_success = 0
            while steps < max_steps:
                xk = env.agent_pos.clone()
                
                env.step(ctrl.act(env))

                img = env.current_frame().to("cpu").numpy()[0]
                rob_pos = env.agent_pos.to(torch.float32).to("cpu").numpy().astype(np.float32)[0]
                dpos = (env.agent_pos - xk).to("cpu").numpy().astype(np.float32)[0]
                g_abs = float(env.gripper_closed[0].item())
                act = np.array([dpos[0], dpos[1], g_abs], dtype=np.float32)
                
                frame = {
                    "observation.images.camera1": img,
                    "observation.state": np.array([rob_pos[0], rob_pos[1], g_abs], dtype=np.float32),
                    "action": act,
                    "task": task_str,
                }
                ds.add_frame(frame)
                steps_after_success += env.success[0].item()
                
                if steps_after_success > 50:
                    break
                steps += 1

            ds.save_episode()
            episode_task_records.append({
                "object_id": int(env.object_id[0].item()),
                "goal_id": int(env.goal_id),
                "task": task_str,
            })

    ds.finalize()
    
    # Save episode-level task strings next to the dataset
    sidecar_path = Path(root) / "episode_tasks.json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        _json.dump(episode_task_records, fh, indent=2)
    print(f"LeRobot dataset recorded: root='{root}', episodes={episodes_per_task*len(tasks)}, fps={fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pick+place demos for a fixed object over a goal range")
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--goal_id_s", type=int, required=True)
    parser.add_argument("--goal_id_e", type=int, required=True)
    parser.add_argument("--root", type=str, default="datasets/tmp")
    parser.add_argument("--episodes_per_task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_distractors", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    # Build tasks: fixed object against goal range, skipping goal==object
    tasks = [(args.object_id, g) for g in range(args.goal_id_s, args.goal_id_e) if g != args.object_id]

    collect_lerobot_dataset(
        episodes_per_task=args.episodes_per_task,
        root=args.root,
        tasks=tasks,
        seed=args.seed,
        max_distractors=args.max_distractors,
        image_size=args.image_size,
    )
