import torch
from env import PickAndPlaceEnv
import numpy as np
import zarr
from tqdm import tqdm

class ExpertController:
    """
    Batched PD controller using full state knowledge.
    Phases per batch item: to object until picked, then to goal until delivered.
    """

    def __init__(self):
        self.kp = 10.0
        self.kd = 4.0
        self.kp_loaded = 10.0
        self.kd_loaded = 8.0
        self.u_clip = 4.0

    def act(self, env: PickAndPlaceEnv) -> torch.Tensor:
        target = torch.where((~env.picked).unsqueeze(-1), env.object_pos, env.goal_center)  # [B,2]
        kp = torch.where(env.picked, self.kp_loaded, self.kp).unsqueeze(1)
        kd = torch.where(env.picked, self.kd_loaded, self.kd).unsqueeze(1)
        pos = env.agent_pos
        vel = env.agent_vel
        u = kp * (target - pos) - kd * vel
        
        u = torch.clamp(u, min=-self.u_clip, max=self.u_clip)
        return u


def collect_expert_dataset(N: int, path: str, multi_task: bool):
    B = 20
    assert int(N) % B == 0
    env = PickAndPlaceEnv(batch_size=B, multi_task=multi_task)
    ctrl = ExpertController()
    max_steps = int(10.0 / float(env.dt))
    
    H, W = int(env.H), int(env.W)
    frames_all = torch.empty(max_steps, B, H, W, 3, dtype=torch.uint8)
    poss_all = torch.empty(max_steps, B, 2, dtype=torch.float32)
    u_all = torch.empty(max_steps, B, 2, dtype=torch.float32)
    obj_pos_all = torch.empty(max_steps, B, 2, dtype=torch.float32)

    root = zarr.open_group(path, mode='w')
    img_arr = root.create_array('data/img', shape=(0, H, W, 3), dtype=np.uint8, chunks=(1000, H, W, 3))
    state_arr = root.create_array('data/state', shape=(0, 2), dtype=np.float32, chunks=(10000, 2))
    cmd_arr = root.create_array('data/cmd', shape=(0, 2), dtype=np.float32, chunks=(10000, 2))
    action_arr = root.create_array('data/action', shape=(0, 2), dtype=np.float32, chunks=(10000, 2))
    obj_pos_arr = root.create_array('meta/obj_pos', shape=(0, 2), dtype=np.float32, chunks=(10000, 2))
    
    episode_ends = []
    goal_positions = []
    total_samples = 0

    for start in tqdm(range(0, int(N), B)):
        env.reset(record_video=False)
        env.step(torch.zeros(B, 2))
        
        for b in range(B):
            goal_idx = int(env._target_goal_idx[b].item())
            goal_pos = env.goals_center[b, goal_idx].to(torch.float32).cpu().numpy().copy()
            goal_positions.append(goal_pos)
        
        active = torch.ones(B, dtype=torch.bool)
        done_step = torch.full((B,), -1, dtype=torch.long)
        t = 0
        frames_all[t] = env.current_frame().to("cpu")
        poss_all[t] = env.agent_pos.to("cpu")
        u_all[t] = torch.zeros(B, 2)
        for b in range(B):
            obj_idx = int(env._target_object_idx[b].item())
            obj_pos_all[t, b] = env.objects_pos[b, obj_idx].to(torch.float32).to("cpu")
        t += 1
        
        while bool(active.any()) and t < max_steps:
            u = ctrl.act(env)
            out = env.step(u)
            frames_all[t] = env.current_frame().to("cpu")
            poss_all[t] = env.agent_pos.to("cpu")
            u_all[t] = u.to("cpu")
            for b in range(B):
                obj_idx = int(env._target_object_idx[b].item())
                obj_pos_all[t, b] = env.objects_pos[b, obj_idx].to(torch.float32).to("cpu")
            succ_cpu = out["success"].to("cpu")
            newly_done = active & succ_cpu
            done_step[newly_done] = t
            active[newly_done] = False
            t += 1

        for b in range(B):
            t_end = int(done_step[b].item()) + 1 if int(done_step[b].item()) >= 0 else t
            cmd = env.command[b].to(torch.float32).cpu().numpy().copy()
            
            img_arr.resize((total_samples + t_end, H, W, 3))
            state_arr.resize((total_samples + t_end, 2))
            cmd_arr.resize((total_samples + t_end, 2))
            action_arr.resize((total_samples + t_end, 2))
            obj_pos_arr.resize((total_samples + t_end, 2))
            
            img_arr[total_samples:total_samples+t_end] = frames_all[:t_end, b].numpy()
            state_arr[total_samples:total_samples+t_end] = poss_all[:t_end, b].numpy()
            cmd_arr[total_samples:total_samples+t_end] = np.tile(cmd, (t_end, 1))
            action_arr[total_samples:total_samples+t_end] = u_all[:t_end, b].numpy()
            obj_pos_arr[total_samples:total_samples+t_end] = obj_pos_all[:t_end, b].numpy()
            total_samples += t_end
            episode_ends.append(total_samples)
    
    root['meta/episode_ends'] = np.array(episode_ends)
    root['meta/goal_pos'] = np.array(goal_positions)
    root['meta/dt'] = env.dt.numpy().reshape(1)
    root['meta/multi_task'] = np.array(env.multi_task).reshape(1)
    print(f"wrote {N} demonstrations totalling {total_samples} datapoints")

def generate_expert_demo_video(batch_size: int = 4, episodes: int = 1, multi_task=False):
    # Always batched env; all tensors have leading batch dim
    env = PickAndPlaceEnv(batch_size=batch_size, multi_task=multi_task)
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
    multi_task=False
    demos = 1000
    # generate_expert_demo_video(multi_task=multi_task)
    if 1:
        collect_expert_dataset(demos, f"datasets/expert_{demos}_singletask.zarr", multi_task=multi_task)
