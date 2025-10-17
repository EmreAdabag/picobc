import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from bc_model import BCModel
from rollout import rollout as eval_rollout


class HDF5DeltaDataset(Dataset):
    """Streams transitions from an HDF5 file without preloading everything into RAM.

    Builds an index of (group_key, t) pairs where each item corresponds to a
    transition (t -> t+1). Each __getitem__ opens the HDF5 file and reads just
    the necessary slices to construct a single sample.
    """
    def __init__(self, path: str, num_demos: int = 0):
        super().__init__()
        self.path = path
        self._index: list[tuple[str, int]] = []

        groups = None
        if num_demos > 0:
            with h5py.File(path, 'r') as f:
                all_groups = list(f.keys())
            assert num_demos <= len(all_groups)
            rng = np.random.default_rng()
            rng.shuffle(all_groups)
            groups = all_groups[:num_demos]

        # Pre-scan groups to build a flat index of transitions
        with h5py.File(self.path, 'r') as f:
            keys = list(f.keys()) if groups is None else list(groups)
            for k in keys:
                g = f[k]
                if 'images' not in g or 'agent_pos' not in g or 'command' not in g:
                    continue
                T = int(g['images'].shape[0])
                if T < 2:
                    continue
                # For episode group k, valid transition timesteps are 0..T-2
                for t in range(T - 1):
                    self._index.append((k, t))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_key, t = self._index[idx]
        # Read exactly one transition worth of data
        with h5py.File(self.path, 'r') as f:
            g = f[ep_key]
            # Single frame and state at t, and next pos at t+1
            img = g['images'][t]                 # HxWx3 uint8
            pos_t = g['agent_pos'][t]            # 2
            pos_tp1 = g['agent_pos'][t + 1]      # 2
            cmd = g['command'][...]              # 2

        # Prepare tensors
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).contiguous()   # 3xHxW
        state_t = torch.from_numpy(pos_t).float().contiguous()                              # 2
        cmd_t = torch.from_numpy(cmd).float().contiguous()                                  # 2
        dpos = torch.from_numpy(pos_tp1 - pos_t).float().contiguous()                       # 2

        return img_t, state_t, cmd_t, dpos


@torch.no_grad()
def _eval_closed_loop_using_rollout(model: BCModel, episodes: int = 10, max_steps: int = 300) -> tuple[int, int]:
    # Directly evaluate the in-memory model without saving, no videos
    successes = eval_rollout(model, episodes=episodes, out_dir='ckpts', max_steps=max_steps, render_video=False)
    # Ensure training mode after eval
    model.train()
    return successes, int(episodes)


def train(dataset_path: str, out_path: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-4, resume: str | None = None, backbone_ckpt: str | None = None, num_demos: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = HDF5DeltaDataset(dataset_path, num_demos=num_demos)
    N = len(ds)
    val_size = max(1, int(round(0.1 * N))) if N > 0 else 0
    train_size = max(0, N - val_size)
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    # Use conservative num_workers to avoid h5py file handle issues
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"dataset size: {N} | train: {train_size} | val: {val_size} | batch size: {batch_size}")

    model = BCModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_params}")
    
    if backbone_ckpt:
        b_state = torch.load(backbone_ckpt, map_location=device)
        filtered = {k[len('backbone.'):]: v for k, v in b_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(filtered, strict=True)
        print(f"initialized backbone from {backbone_ckpt}")

    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state)
        print(f"loaded checkpoint from {resume}")
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    best_val = float('inf')
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
            loss = loss_fn(pred, dpos)
            loss.backward()
            opt.step()

            total += loss.item() * img.size(0)
            count += img.size(0)
        print(f"epoch {global_ep}/{epochs} | loss={total/max(1,count):.6f}")
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
                    v_total += nn.functional.mse_loss(pred, dpos, reduction='sum').item()
                    v_count += img.size(0)
                val_loss = v_total / max(1, v_count)
            print(f"           val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_path = f"best_{out_path}"
                torch.save(model.state_dict(), best_path)
        model.train()
        if (global_ep) % 10 == 0:
            ckpt_path = f"ckpts/checkpoint_ep{global_ep}.pt"
            torch.save(model.state_dict(), ckpt_path)
            # Closed-loop evaluation (no video), report success rate
            s, tot = _eval_closed_loop_using_rollout(model, episodes=10, max_steps=300)
            rate = s / max(1, tot)
            print(f"closed-loop eval @epoch {global_ep}: success {s}/{tot} (rate={rate:.2f})")

    torch.save(model.state_dict(), out_path)
    print(f"saved model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--out', type=str, default='bc_model.pt', help='Output path for model weights')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to initialize from')
    parser.add_argument('--backbone_ckpt', type=str, default=None, help='Checkpoint for backbone')
    parser.add_argument('--num_demos', type=int, default=0, help='Number of demonstrations to use before split (0 = all)')
    args = parser.parse_args()

    print(f"dataset: {args.data} | out: {args.out} | epochs: {args.epochs} | batch_size: {args.batch_size} | resume: {args.resume} | backbone_ckpt: {args.backbone_ckpt} | num_demos: {args.num_demos}")
    os.makedirs('ckpts', exist_ok=True)

    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size, resume=args.resume, backbone_ckpt=args.backbone_ckpt, num_demos=args.num_demos)
