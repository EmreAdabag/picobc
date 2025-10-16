import torch
from env import PickAndPlaceEnv, ExpertController, collect_expert_dataset


def run_episode(max_steps: int = 500):
    env = PickAndPlaceEnv()
    ctrl = ExpertController()
    for ep in range(10):
        env.reset(record_video=True)
        for t in range(max_steps):
            u = ctrl.act(env)
            out = env.step(u)
            if out["success"]:
                break

        # Save video for this episode
        path = f"episode_{ep}.mp4"
        env.save_video(path, fps=30)
        print(f"Episode {ep}: success={env.success}, steps={t+1}, video={path}")


if __name__ == "__main__":
    run_episode()
    if 1:
        collect_expert_dataset(2000, "expert_multitask.h5")