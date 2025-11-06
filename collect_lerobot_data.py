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
    B: int = 8,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert episodes_per_task % B == 0
    episodes_per_task = episodes_per_task // B

    # Bootstrap env to read rendering params
    dt = 0.01
    env_boot = PickAndPlaceEnv(batch_size=B, task_type="place", device=device, record_video=False, dt=dt)
    fps = int(1.0 / float(env_boot.dt))
    H, W = int(env_boot.H), int(env_boot.W)

    robot_state_dim = 2

    # Common vector features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (robot_state_dim,),
            "names": [
                "agent.pos.x",
                "agent.pos.y",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["ux", "uy"],
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
        image_writer_threads=32,
        batch_encoding_size=1,
    )

    max_steps = int(10.0 / float(env_boot.dt))

    for i, (obj_idx, task_type) in enumerate(tasks):
        print(f"\n\nstarting task {i}\n")
        env = PickAndPlaceEnv(batch_size=B, task_type=task_type, device=device, record_video=False, dt=dt, seed=seed)
        for _ in range(episodes_per_task):
            env.reset(task_type=task_type, obj_idx=obj_idx)
            env.object_id = torch.full((B,), int(obj_idx), device=env.device, dtype=torch.long)
            ctrl = ExpertController(task_type)

            info = env.command_info(0)
            if (task_type=='place'):
                task_str = f"place the {info['object_name']} on the gold star"
            else:
                task_str = f"pick up the {info['object_name']}"
                
            imgs_steps = []
            robot_steps = []
            act_steps = []
            succ_step = np.full((B,), max_steps)

            steps = 0
            while steps < max_steps:
                act = ctrl.act(env)
                xk = env.agent_pos.clone()
                env.step(act)
                pos_delta = (env.agent_pos - xk).to("cpu").numpy().astype(np.float32)

                imgs = env.current_frame().to("cpu").numpy()
                robots = env.agent_pos.to(torch.float32).to("cpu").numpy().astype(np.float32)
                succ = env.success.detach().to("cpu").numpy().astype(bool)
                
                imgs_steps.append(imgs)
                robot_steps.append(robots)
                act_steps.append(pos_delta)
                just_succeeded = (succ_step==max_steps) & succ
                endstep = min(steps+100, max_steps-1) # buffer for 50 steps after success
                succ_step = np.where(just_succeeded, endstep, succ_step).astype(int)

                steps += 1
                if (steps > succ_step).all():
                    break
            
            for b in range(B):
                for t in range(succ_step[b]):
                    img = imgs_steps[t][b]
                    rob = robot_steps[t][b]
                    act = np.zeros_like(act_steps[t][b], dtype=np.float32)

                    frame = {
                        "observation.images.camera1": img,
                        "observation.state": rob,
                        "action": act,
                        "task": task_str,
                    }
                    ds.add_frame(frame)
                ds.save_episode()

    ds.finalize()
    print(f"LeRobot dataset recorded: repo_id='{repo_id}', root='{(root / repo_id)}', episodes={episodes_per_task*len(tasks)}, fps={fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll out expert policy and save a video")
    parser.add_argument("--obj_id_s", type=int, default=0)
    parser.add_argument("--obj_id_e", type=int, default=1)
    parser.add_argument("--root", type=str, default="datasets/tmp")
    parser.add_argument("--episodes_per_task", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tasks = [(obj_id, tsk) for obj_id in range(args.obj_id_s, args.obj_id_e) for tsk in ['pick', 'place']]

    collect_lerobot_dataset(
        episodes_per_task=args.episodes_per_task,
        repo_id="null",
        root=args.root,
        tasks=tasks,
        B=args.batch_size,
        seed=args.seed
    )
