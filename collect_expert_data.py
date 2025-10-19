import torch
from env import PickAndPlaceEnv, ExpertController, collect_expert_dataset


def run_episode(batch_size: int = 4, episodes: int = 1):
    # Always batched env; all tensors have leading batch dim
    env = PickAndPlaceEnv(batch_size=batch_size, multi_task=False)
    ctrl = ExpertController()
    for ep in range(episodes):
        env.reset(record_video=True)
        steps = 0
        while True:
            u = ctrl.act(env)                  # [B,2]
            out = env.step(u)
            steps += 1
            if bool(out["success"].all()):
                break

        # Save one concatenated video for this episode across the batch
        path = f"episode_{ep}.mp4"
        env.save_video(path)
        # Report per-batch success flags
        print(f"Episode {ep}: success={env.success.tolist()}, steps={steps}, video={path}")


if __name__ == "__main__":
    # run_episode()
    if 1:
        collect_expert_dataset(1000, "expert_1k.h5")
