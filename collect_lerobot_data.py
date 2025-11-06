import sys
from pathlib import Path
import numpy as np
import torch
from env import PickAndPlaceEnv
from expert_rollout import ExpertController
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse

def collect_lerobot_dataset(
    episodes_per_task: int,
    repo_id: str,
    root: str | Path,
    tasks: list,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bootstrap env to read rendering params
    dt = 0.01
    env_boot = PickAndPlaceEnv(task_type="place", device=device, record_video=False, dt=dt)
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
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        batch_encoding_size=1,
    )

    max_steps = int(10.0 / float(env_boot.dt))

    for i, (obj_idx, task_type) in enumerate(tasks):
        print(f"\n\nstarting task {i}\n")

        env = PickAndPlaceEnv(task_type=task_type, device=device, record_video=False, dt=dt, seed=seed)
        info = env.command_info(0)
        if (task_type=='place'):
            task_str = f"place the {info['object_name']} on the gold star"
        else:
            task_str = f"pick up the {info['object_name']}"

        for _ in range(episodes_per_task):
            env.reset(task_type=task_type, obj_idx=obj_idx)
            ctrl = ExpertController(task_type)
                
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

    ds.finalize()
    print(f"LeRobot dataset recorded: repo_id='{repo_id}', root='{(root / repo_id)}', episodes={episodes_per_task*len(tasks)}, fps={fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll out expert policy and save a video")
    parser.add_argument("--obj_id_s", type=int, default=0)
    parser.add_argument("--obj_id_e", type=int, default=1)
    parser.add_argument("--root", type=str, default="datasets/tmp")
    parser.add_argument("--episodes_per_task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = [(obj_id, tsk) for obj_id in range(args.obj_id_s, args.obj_id_e) for tsk in ['pick']]

    collect_lerobot_dataset(
        episodes_per_task=args.episodes_per_task,
        repo_id="null",
        root=args.root,
        tasks=tasks,
        seed=args.seed
    )
