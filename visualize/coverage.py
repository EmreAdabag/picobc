
import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import argparse
import os
import zarr

def visualize_agent_trajectories(dataset_path: str, output_path: str = None, num_demos: int = 0):
    assert os.path.exists(dataset_path)
    print(f"Loading dataset from {dataset_path}...")

    root = zarr.open_group(dataset_path, mode='r')
    states = root['data']['state'][:]
    cmds = root['data']['cmd'][:]
    episode_ends = root['meta']['episode_ends'][:]
    obj_positions = root['meta']['obj_pos'][:]
    goal_positions = root['meta']['goal_pos'][:]
    
    if num_demos > 0:
        assert num_demos <= len(episode_ends)
        episode_ends = episode_ends[:num_demos]
        states = states[:episode_ends[-1]]

    resolution = 512
    canvas = torch.zeros(resolution, resolution, 3, dtype=torch.uint8)
    
    obj_colors = [torch.tensor([69, 123, 157], dtype=torch.uint8), torch.tensor([255, 140, 0], dtype=torch.uint8)]
    goal_colors = [torch.tensor([29, 185, 84], dtype=torch.uint8), torch.tensor([155, 89, 182], dtype=torch.uint8)]
    
    total_positions = 0

    for i in range(len(episode_ends)):
        if i % 100 == 0:
            print(f"Processing episode {i}/{len(episode_ends)}...")
        
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        agent_pos = torch.from_numpy(states[start_idx:end_idx])
        total_positions += len(agent_pos)
        
        cmd = cmds[start_idx]
        obj_idx, goal_idx = int(cmd[0]), int(cmd[1])
        obj_pos = torch.from_numpy(obj_positions[i])
        goal_pos = torch.from_numpy(goal_positions[i])
        
        obj_px = int(obj_pos[0].item() * (resolution - 1))
        obj_py = int(obj_pos[1].item() * (resolution - 1))
        goal_px = int(goal_pos[0].item() * (resolution - 1))
        goal_py = int(goal_pos[1].item() * (resolution - 1))
        
        obj_half, goal_half = 4, 3
        canvas[max(0,obj_py-obj_half):min(resolution,obj_py+obj_half+1), 
               max(0,obj_px-obj_half):min(resolution,obj_px+obj_half+1)] = obj_colors[obj_idx]
        canvas[max(0,goal_py-goal_half):min(resolution,goal_py+goal_half+1),
               max(0,goal_px-goal_half):min(resolution,goal_px+goal_half+1)] = goal_colors[goal_idx]

        pixel_y = (agent_pos[:, 1] * (resolution - 1)).round().to(torch.int32)
        pixel_x = (agent_pos[:, 0] * (resolution - 1)).round().to(torch.int32)
        pixel_y = torch.clamp(pixel_y, 0, resolution - 1)
        pixel_x = torch.clamp(pixel_x, 0, resolution - 1)

        for t in range(len(pixel_x)):
            x, y = pixel_x[t].item(), pixel_y[t].item()
            canvas[y, x] = torch.tensor([255, 0, 0], dtype=torch.uint8)

    print(f"Processed {total_positions} total positions across {len(episode_ends)} episodes")

    if output_path is None:
        base_name = os.path.splitext(dataset_path)[0]
        suffix = f"_trajectories_{num_demos}demos" if num_demos > 0 else "_trajectories_all"
        output_path = f"{base_name}{suffix}.png"

    # Save as image
    from PIL import Image
    img = Image.fromarray(canvas.numpy())
    img.save(output_path)
    print(f"Saved agent trajectory visualization to {output_path}")

    return canvas, output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--num_demos', type=int, default=0)
    args = parser.parse_args()
    visualize_agent_trajectories(args.dataset_path, args.output, args.num_demos)
