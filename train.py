import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from bc_model import BCModel
from rollout_bc import rollout as eval_rollout


class HDF5DeltaDataset(Dataset):
    """Preloads all samples into memory for fast iteration."""
    def __init__(self, path: str):
        super().__init__()
        imgs = []
        states = []
        dposes = []

        with h5py.File(path, 'r') as f:
            for k in f.keys():
                g = f[k]
                images = g['images'][...]          # TxHxWx3 uint8
                pos = g['agent_pos'][...]          # Tx2
                vel = g['agent_vel'][...]          # Tx2
                T = images.shape[0]
                if T < 2:
                    continue
                # Build (t -> t+1) pairs
                imgs.append(images[:-1])           # (T-1)xHxWx3
                # st = np.concatenate([pos[:-1], vel[:-1]], axis=-1)  # (T-1)x4
                st = pos[:-1]
                states.append(st)
                dposes.append(pos[1:] - pos[:-1])  # (T-1)x2

        imgs = np.concatenate(imgs, axis=0)            # NxHxWx3
        states = np.concatenate(states, axis=0)        # Nx4
        dposes = np.concatenate(dposes, axis=0)        # Nx2

        # Convert once to tensors
        self.imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        self.states = torch.from_numpy(states).float().contiguous()
        self.dpos = torch.from_numpy(dposes).float().contiguous()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, idx):
        return self.imgs[idx], self.states[idx], self.dpos[idx]


@torch.no_grad()
def _eval_closed_loop_using_rollout(model: BCModel, episodes: int = 10, max_steps: int = 300) -> tuple[int, int]:
    # Directly evaluate the in-memory model without saving, no videos
    successes = eval_rollout(model, episodes=episodes, out_dir='ckpts', max_steps=max_steps, render_video=False)
    # Ensure training mode after eval
    model.train()
    return successes, int(episodes)


def train(dataset_path: str, out_path: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-4, resume: str | None = None, backbone_ckpt: str | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = HDF5DeltaDataset(dataset_path)
    N = len(ds)
    val_size = max(1, int(round(0.1 * N))) if N > 0 else 0
    train_size = max(0, N - val_size)
    if N > 0 and train_size == 0:
        train_size = N - 1
        val_size = 1
    if N > 0:
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, ds

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"dataset size: {N} | train: {train_size} | val: {val_size} | batch size: {batch_size}")

    model = BCModel().to(device)
    # Optionally initialize only the ResNet backbone from a prior BCModel checkpoint saved by this script.
    # This assumes keys are prefixed with 'backbone.'. If format mismatches, let it crash.
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
        for img, state, dpos in tqdm(train_dl, desc=f"epoch {global_ep}/{epochs}"):
            img = img.to(device)
            state = state.to(device)
            dpos = dpos.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(img, state)
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
                for img, state, dpos in val_dl:
                    img = img.to(device)
                    state = state.to(device)
                    dpos = dpos.to(device)
                    pred = model(img, state)
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
    args = parser.parse_args()

    os.makedirs('ckpts', exist_ok=True)

    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size, resume=args.resume, backbone_ckpt=args.backbone_ckpt)
