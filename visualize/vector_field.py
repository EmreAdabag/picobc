import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from bc_model import BCModel, normalize, unnormalize
from env import PickAndPlaceEnv

@torch.no_grad()
def visualize_vector_field(
    model,
    output_path: str = "vector_field.png",
    grid_size: int = 20,
    num_conditions: int = 5
):
    device = next(model.parameters()).device
    model.eval()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, num_conditions, figsize=(5 * num_conditions, 5))
    env = PickAndPlaceEnv(batch_size=1, device=device, multi_task=False)
    
    for cond_idx in range(num_conditions):
        env.reset(record_video=False)
        
        agent_pos_init = torch.rand(2).tolist()
        
        env.agent_pos[0] = torch.tensor(agent_pos_init, device=device, dtype=env.dtype)
        arangeB = torch.arange(1, device=device)
        env.object_pos = env.objects_pos[arangeB, env._target_object_idx]
        env.goal_center = env.goals_center[arangeB, env._target_goal_idx]
        
        grid_x = np.linspace(0.05, 0.95, grid_size)
        grid_y = np.linspace(0.05, 0.95, grid_size)
        
        deltas_x = np.zeros((grid_size, grid_size))
        deltas_y = np.zeros((grid_size, grid_size))
        
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                env.agent_pos[0] = torch.tensor([x, y], device=device, dtype=env.dtype)
                env.agent_vel[0] = torch.zeros(2, device=device, dtype=env.dtype)
                
                frame = env.current_frame().to(torch.uint8)
                img = frame.permute(0, 3, 1, 2).float().div_(255.0).to(device)
                state = env.agent_pos.to(device)
                cmd = env.command.to(device)
                
                state_norm = normalize(state, model.state_min, model.state_max)
                dpos_n = model(img, state_norm, cmd)
                dpos = unnormalize(dpos_n, model.dpos_min, model.dpos_max)
                dpos *= 100 # simulated kp gain
                
                deltas_x[i, j] = dpos[0, 0].cpu().item()
                deltas_y[i, j] = dpos[0, 1].cpu().item()
        
        env.agent_pos[0] = torch.tensor(agent_pos_init, device=device, dtype=env.dtype)
        frame = env.current_frame()[0].cpu().numpy()
        ax, ay = env.agent_pos[0, 0], env.agent_pos[0, 1]
        agent_radius_world = env._world_radius_to_px(env.agent_radius_px)
        mask_agent = ((env.grid_x - ax) ** 2 + (env.grid_y - ay) ** 2 <= agent_radius_world ** 2).cpu().numpy()
        frame[mask_agent] = np.array([255, 255, 255])

        ax_subplot = axes[cond_idx]
        ax_subplot.imshow(frame, extent=[0, 1, 1, 0], origin='upper')
        
        X, Y = np.meshgrid(grid_x, grid_y)
        ax_subplot.quiver(X, Y, deltas_x, deltas_y, angles='xy', color='red', alpha=0.8, scale=5.0, width=0.003)
        
        ax_subplot.set_xlim(0, 1)
        ax_subplot.set_ylim(1, 0)
        ax_subplot.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved vector field visualization to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="vector_field.png")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--num_conditions", type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCModel().to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    
    visualize_vector_field(
        model,
        args.output,
        args.grid_size,
        args.num_conditions
    )

