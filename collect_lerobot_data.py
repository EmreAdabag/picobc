import sys
from pathlib import Path
import numpy as np
import torch
from env import PickAndPlaceEnv
from expert_rollout import ExpertController

# Ensure we import the local LeRobot sources in ./lerobot/src before any installed package
_ROOT_DIR = Path(__file__).parent
_LR_SRC = _ROOT_DIR / "lerobot" / "src"
if str(_LR_SRC) not in sys.path:
    sys.path.insert(0, str(_LR_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def collect_lerobot_dataset(
    episodes: int,
    repo_id: str,
    root: str | Path,
    task_ids: list,
    batch_size: int = 8,
):
    # sys.path is already updated at import time to prefer local lerobot

    assert task_ids is not None
    assert len(task_ids) > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = int(batch_size)

    # Two tasks per object: pick and place
    combos = [(oid, t) for oid in task_ids for t in ("pick", "place")]
    assert int(episodes) % (len(combos) * B) == 0
    episodes_per_combo = int(episodes) // len(combos)
    runs_per_combo = episodes_per_combo // B

    # Bootstrap env to read rendering params
    env_boot = PickAndPlaceEnv(batch_size=B, task_type="place", device=device, record_video=False)
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
    batch_encode = max(1, min(8, int(episodes)))
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        batch_encoding_size=batch_encode,
    )

    max_steps = int(10.0 / float(env_boot.dt))

    for (obj_idx, task_type) in combos:
        for _ in range(runs_per_combo):
            env = PickAndPlaceEnv(batch_size=B, task_type=task_type, device=device, record_video=False)
            env.reset(task_type=task_type, obj_idx=obj_idx)
            # Ensure obj_idx==0 is honored
            env.object_id = torch.full((B,), int(obj_idx), device=env.device, dtype=torch.long)
            ctrl = ExpertController(task_type)

            info = env.command_info(0)
            task_str = f"{task_type} the {info['object_name']}"

            imgs_steps = []
            robot_steps = []
            act_steps = []

            steps = 0
            while steps < max_steps:
                act = ctrl.act(env)
                xk = env.agent_pos.clone()
                env.step(act)
                pos_delta = (env.agent_pos - xk).to("cpu").numpy().astype(np.float32)

                imgs = env.current_frame().to("cpu").numpy()
                robots = env.agent_pos.to(torch.float32).to("cpu").numpy().astype(np.float32)

                imgs_steps.append(imgs)
                robot_steps.append(robots)
                act_steps.append(pos_delta)

                steps += 1
                if bool(env.success.all()):
                    break

            T = len(imgs_steps)
            for b in range(B):
                for t in range(T):
                    frame = {
                        "observation.images.camera1": imgs_steps[t][b],
                        "observation.state": robot_steps[t][b],
                        "action": act_steps[t][b],
                        "task": task_str,
                    }
                    ds.add_frame(frame)
                ds.save_episode()

    ds.finalize()
    print(
        f"LeRobot dataset recorded: repo_id='{repo_id}', root='{(root / repo_id)}', episodes={episodes}, fps={fps}"
    )


if __name__ == "__main__":
    collect_lerobot_dataset(
        episodes=32,
        repo_id="null",
        root="datasets_smolvla_twotask",
        task_ids=[0, 1],
        batch_size=8,
    )
