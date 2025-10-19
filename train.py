import os
import math
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from bc_model import BCModel
from rollout import rollout_batch as eval_rollout
import wandb

USE_WANDB=1
def wandb_log_maybe(log, step=None):
    if USE_WANDB:
        wandb.log(log, step=step)    


class HDF5DeltaDataset(Dataset):
    """Loads all transitions from HDF5 into RAM for fast access.

    Preloads all data as numpy arrays (uint8 for images to save memory).
    Converts to tensors on-the-fly in __getitem__.
    """
    def __init__(self, path: str, num_demos: int = 0):
        super().__init__()
        self.data = []  # List of (img_uint8, pos_t, pos_tp1, cmd) tuples

        groups = None
        if num_demos > 0:
            with h5py.File(path, 'r') as f:
                all_groups = list(f.keys())
            assert num_demos <= len(all_groups)
            rng = np.random.default_rng()
            rng.shuffle(all_groups)
            groups = all_groups[:num_demos]

        # Load all data into memory
        print(f"Loading HDF5 file into memory: {path}")
        with h5py.File(path, 'r') as f:
            keys = list(f.keys()) if groups is None else list(groups)
            for k in tqdm(keys, desc="Loading episodes"):
                g = f[k]
                if 'images' not in g or 'agent_pos' not in g or 'command' not in g:
                    continue
                
                # Load entire episode at once
                images = np.array(g['images'])       # TxHxWx3 uint8
                agent_pos = np.array(g['agent_pos']) # Tx2 float
                command = np.array(g['command'])     # 2 float
                
                T = images.shape[0]
                if T < 2:
                    continue
                
                # Store all transitions
                for t in range(T - 1):
                    self.data.append((
                        images[t],        # HxWx3 uint8
                        agent_pos[t],     # 2
                        agent_pos[t + 1], # 2
                        command           # 2
                    ))
        
        print(f"Loaded {len(self.data)} transitions into memory")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, pos_t, pos_tp1, cmd = self.data[idx]
        
        # Convert from numpy to tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).contiguous()   # 3xHxW
        state_t = torch.from_numpy(pos_t).float().contiguous()                              # 2
        cmd_t = torch.from_numpy(cmd).float().contiguous()                                  # 2
        dpos = torch.from_numpy(pos_tp1 - pos_t).float().contiguous()                       # 2

        return img_t, state_t, cmd_t, dpos


@torch.no_grad()
def _eval_closed_loop_using_rollout(model: BCModel, episodes: int = 10, timeout_s: float = 5.0) -> tuple[int, int]:
    model.eval()
    successes = eval_rollout(model, episodes=episodes, timeout_s=timeout_s)
    model.train()
    return successes, episodes


def train(dataset_path: str, out_dir: str, steps: int, batch_size: int, lr: float, resume: str | None, pretrained_backbone: str | None, num_demos: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    eval_every_n_steps = 1000

    if USE_WANDB:
        wandb.init(project=os.environ.get('WANDB_PROJECT', 'mcdonalds'), config={
            'data': dataset_path,
            'out': out_dir,
            'steps': steps,
            'batch_size': batch_size,
            'lr': lr,
            'resume': resume,
            'pretrained_backbone': pretrained_backbone,
            'num_demos': num_demos,
            'device': str(device),
        })

    os.makedirs(out_dir, exist_ok=True)

    ds = HDF5DeltaDataset(dataset_path, num_demos=num_demos)
    N = len(ds)
    val_size = max(1, int(round(0.1 * N))) if N > 0 else 0
    train_size = max(0, N - val_size)
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    # Data is in memory, use fewer workers to avoid serialization overhead
    num_workers = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                          pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                        pin_memory=torch.cuda.is_available())

    model = BCModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"dataset size: {N} | train: {train_size} | val: {val_size} | batch size: {batch_size}")
    print(f"trainable params: {n_params}")
    if USE_WANDB:
        wandb.config.update({'dataset_size': N, 'train_size': train_size, 'val_size': val_size}, allow_val_change=True)
        wandb.config.update({'trainable_params': int(n_params)}, allow_val_change=True)
    
    if pretrained_backbone:
        b_state = torch.load(pretrained_backbone, map_locatio=device)
        filtered = {k[len('backbone.'):]: v for k, v in b_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(filtered, strict=True)
        print(f"initialized backbone from {pretrained_backbone}")

    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state)
        print(f"loaded checkpoint from {resume}")
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # compute normalization stats on training split (per-dim percentiles)
    tr_idx = train_ds.indices
    state_arr = np.stack([ds.data[i][1] for i in tr_idx], axis=0)
    dpos_arr = np.stack([ds.data[i][2] - ds.data[i][1] for i in tr_idx], axis=0)
    s_p02, s_p98 = np.percentile(state_arr, [2, 98], axis=0)
    d_p02, d_p98 = np.percentile(dpos_arr, [2, 98], axis=0)
    model.state_p02.copy_(torch.tensor(s_p02, dtype=torch.float32, device=device))
    model.state_p98.copy_(torch.tensor(s_p98, dtype=torch.float32, device=device))
    model.dpos_p02.copy_(torch.tensor(d_p02, dtype=torch.float32, device=device))
    model.dpos_p98.copy_(torch.tensor(d_p98, dtype=torch.float32, device=device))
    print(f"norm: state p02={s_p02.tolist()} p98={s_p98.tolist()} dpos p02={d_p02.tolist()} p98={d_p98.tolist()}")

    model.train()
    best_val = float('inf')
    global_step = 0
    steps_per_epoch = math.ceil(train_size / batch_size)
    epochs = int(batch_size * steps // train_size)
    for ep in range(epochs):
        global_ep = ep + 1
        total = 0.0
        count = 0
        for img, state, cmd, dpos in tqdm(train_dl, desc=f"epoch {global_ep}/{epochs}"):
            img = img.to(device)
            state = state.to(device)
            cmd = cmd.to(device)
            dpos = dpos.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(img, state, cmd)
            dpos_n = (2.0 * (dpos - model.dpos_p02) / (model.dpos_p98 - model.dpos_p02) - 1.0).clamp(-1.5, 1.5)
            loss = loss_fn(pred, dpos_n)
            loss.backward()
            opt.step()

            total += loss.item() * img.size(0)
            count += img.size(0)

            global_step += 1
            if global_step % 5 == 0:
                wandb_log_maybe({'train/loss': loss.item(), 'epoch': global_ep}, step=global_step)

        train_loss = total/max(1,count)
        print(f"epoch {global_ep}/{epochs} | loss={train_loss:.6f}")
        
        # Validation
        if len(val_dl) > 0:
            model.eval()
            with torch.no_grad():
                v_total, v_count = 0.0, 0
                for img, state, cmd, dpos in val_dl:
                    img = img.to(device)
                    state = state.to(device)
                    cmd = cmd.to(device)
                    dpos = dpos.to(device)
                    pred = model(img, state, cmd)
                    dpos_n = (2.0 * (dpos - model.dpos_p02) / (model.dpos_p98 - model.dpos_p02) - 1.0).clamp(-1.5, 1.5)
                    v_total += nn.functional.mse_loss(pred, dpos_n, reduction='sum').item()
                    v_count += img.size(0)
                val_loss = v_total / max(1, v_count)
            print(f"val_loss={val_loss:.6f}")
            wandb_log_maybe({'val/loss': val_loss, 'epoch': global_ep}, step=global_step)
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(out_dir, "best.pt")
                torch.save(model.state_dict(), best_path)
        model.train()
        
        # sim eval
        if (global_step) // eval_every_n_steps != (global_step - steps_per_epoch) // eval_every_n_steps:
            ckpt_path = os.path.join(out_dir, f"checkpoint_ep{global_ep}.pt")
            torch.save(model.state_dict(), ckpt_path)
            s, tot = _eval_closed_loop_using_rollout(model, episodes=100, timeout_s=5.0)
            rate = s / max(1, tot)
            print(f"closed-loop eval @epoch {global_ep}: success {s}/{tot} (rate={rate:.2f})")
            wandb_log_maybe({'eval/success_rate': rate, 'epoch': global_ep}, step=global_step)

    final_path = os.path.join(out_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"saved final checkpoint to {final_path}")
    if USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--out', type=str, default='ckpts', help='Output directory for checkpoints')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to initialize from')
    parser.add_argument('--pretrained_backbone', type=str, default=None, help='Checkpoint for backbone')
    parser.add_argument('--num_demos', type=int, default=0, help='Number of demonstrations to use before split (0 = all)')
    args = parser.parse_args()

    train(args.data, args.out, steps=args.steps, batch_size=args.batch_size, lr=args.lr, resume=args.resume, pretrained_backbone=args.pretrained_backbone, num_demos=args.num_demos)
