import os, sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import zarr
import argparse
import torch
from torchvision.io import write_video

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--num_demos', type=int, default=1)
parser.add_argument('--subsample', type=int, default=1)
args = parser.parse_args()

root = zarr.open_group(args.dataset, mode='r')
# dt = root['meta']['dt'][0]
dt = 0.005
imgs = root['data']['img']
episode_ends = root['meta']['episode_ends']

end_idx = episode_ends[args.num_demos - 1]
frames = torch.from_numpy(imgs[:end_idx:args.subsample])

output = args.dataset.rstrip('/').replace('.zarr', f'_{args.num_demos}demos.mp4')
write_video(output, frames, fps=int(1/dt))
print(f"Saved video to {output}")

