import os
import math
import argparse
import zarr
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from bc_model import BCVisionModel, BCStateModel, normalize, unnormalize
from rollout import batch_eval as eval_rollout
from visualize.vector_field import visualize_vector_field
import wandb


USE_WANDB=0
VALIDATE=0

def save_checkpoint(model, optimizer, epoch, step, out_dir, filename):
    path = os.path.join(out_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }, path)
def wandb_log_maybe(log, step=None):
    if USE_WANDB:
        wandb.log(log, step=step)    


class ZarrDeltaDataset(Dataset):
    def __init__(self, path: str, num_demos: int = 0, mode: str = 'vision'):
        super().__init__()
        assert mode in ['state', 'vision']
        self.mode = mode
        root = zarr.open_group(path, mode='r')
        
        imgs = root['data']['img'][:] if mode == 'vision' else None
        states = root['data']['state'][:]
        cmds = root['data']['cmd'][:]
        obj_positions = root['meta']['obj_pos'][:] if mode == 'state' else None
        goal_positions = root['meta']['goal_pos'][:]
        episode_ends = root['meta']['episode_ends'][:]
        
        if num_demos > 0:
            assert num_demos <= len(episode_ends)
            episode_ends = episode_ends[:num_demos]
            end_idx = episode_ends[-1]
            if imgs is not None:
                imgs = imgs[:end_idx]
            states = states[:end_idx]
            cmds = cmds[:end_idx]
            if obj_positions is not None:
                obj_positions = obj_positions[:end_idx]
            goal_positions = goal_positions[:num_demos]
        
        indices = []
        for i in range(len(episode_ends)):
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            for t in range(start_idx, end_idx - 1):
                indices.append(t)
        
        self.imgs = imgs
        self.cmds = cmds
        self.obj_positions = obj_positions
        self.goal_positions = goal_positions
        self.episode_ends = episode_ends
        self.indices = indices
        
        dpos_all = states[1:] - states[:-1]
        _state_arr = states[indices]
        _dpos_arr = dpos_all[indices]
        
        self.state_min = _state_arr.min(axis=0)
        self.state_max = _state_arr.max(axis=0)
        self.dpos_min = _dpos_arr.min(axis=0)
        self.dpos_max = _dpos_arr.max(axis=0)
        
        self.states_norm = normalize(states, self.state_min, self.state_max)
        self.dpos_norm = normalize(dpos_all, self.dpos_min, self.dpos_max)
        self.goals_norm = normalize(goal_positions, self.state_min, self.state_max)
        self.obj_positions_norm = normalize(obj_positions, self.state_min, self.state_max)
        
        print(f"Loaded {len(indices)} transitions")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        dpos_norm = torch.from_numpy(self.dpos_norm[t]).float()
        
        if self.mode == 'vision':
            img = torch.from_numpy(self.imgs[t]).permute(2, 0, 1).float().div_(255.0)
            state_t_norm = torch.from_numpy(self.states_norm[t]).float()
            cmd_t = torch.from_numpy(self.cmds[t]).float()
            return img, state_t_norm, cmd_t, dpos_norm
        else:
            state_t_norm = torch.from_numpy(self.states_norm[t]).float()
            obj_pos_t = torch.from_numpy(self.obj_positions_norm[t]).float()
            ep_idx = next(i for i, end in enumerate(self.episode_ends) if t < end)
            goal_pos_t = torch.from_numpy(self.goals_norm[ep_idx]).float()
            cmd_t = torch.from_numpy(self.cmds[t]).float()
            obs = torch.cat([state_t_norm, obj_pos_t, goal_pos_t, cmd_t])
            return obs, dpos_norm



def train(dataset_path: str, out_dir: str, steps: int, batch_size: int, lr: float, resume: str | None, pretrained_backbone: str | None, num_demos: int, device: str, mode: str):
    eval_every_n_steps = 20000
    assert mode in ['state', 'vision']

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
            'mode': mode,
        })

    os.makedirs(out_dir, exist_ok=True)

    ds = ZarrDeltaDataset(dataset_path, num_demos=num_demos, mode=mode)
    N = len(ds)
    val_size = max(1, int(round(0.1 * N))) if N > 0 else 0
    train_size = max(0, N - val_size)
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    num_workers = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model = (BCVisionModel() if mode == 'vision' else BCStateModel()).to(device)
    
    model.state_min.copy_(torch.from_numpy(ds.state_min).float().to(model.state_min.device))
    model.state_max.copy_(torch.from_numpy(ds.state_max).float().to(model.state_max.device))
    model.dpos_min.copy_(torch.from_numpy(ds.dpos_min).float().to(model.dpos_min.device))
    model.dpos_max.copy_(torch.from_numpy(ds.dpos_max).float().to(model.dpos_max.device))
    print(f"state min: {ds.state_min} | state max: {ds.state_max} | dpos min: {ds.dpos_min} | dpos max: {ds.dpos_max}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"dataset size: {N} | train: {train_size} | val: {val_size} | batch size: {batch_size}")
    print(f"trainable params: {n_params}")
    if USE_WANDB:
        wandb.config.update({'dataset_size': N, 'train_size': train_size, 'val_size': val_size}, allow_val_change=True)
        wandb.config.update({'trainable_params': int(n_params)}, allow_val_change=True)
    
    if pretrained_backbone:
        b_state = torch.load(pretrained_backbone, map_location=device)
        filtered = {k[len('backbone.'):]: v for k, v in b_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(filtered, strict=True)
        print(f"initialized backbone from {pretrained_backbone}")
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    start_epoch = 0
    global_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('step', 0)
        print(f"loaded checkpoint from {resume} (epoch={start_epoch}, step={global_step})")

    model.train()
    best_val = float('inf')
    steps_per_epoch = math.ceil(train_size / batch_size)
    epochs = int(batch_size * steps // train_size)
    global_ep = start_epoch
    for ep in range(start_epoch, epochs):
        global_ep = ep + 1
        total = 0.0
        count = 0
        for batch in tqdm(train_dl, desc=f"epoch {global_ep}/{epochs}"):
            if mode == 'vision':
                img, state, cmd, dpos = batch
                img = img.to(device)
                state = state.to(device)
                cmd = cmd.to(device)
                dpos = dpos.to(device)
                pred = model(img, state, cmd)
                batch_size_actual = img.size(0)
            else:
                obs, dpos = batch
                obs = obs.to(device)
                dpos = dpos.to(device)
                pred = model(obs)
                batch_size_actual = obs.size(0)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(pred, dpos)
            loss.backward()
            opt.step()

            total += loss.item() * batch_size_actual
            count += batch_size_actual

            global_step += 1
            if global_step % 5 == 0:
                wandb_log_maybe({'train/loss': loss.item(), 'epoch': global_ep}, step=global_step)

        train_loss = total/max(1,count)
        print(f"epoch {global_ep}/{epochs} | loss={train_loss:.6f}")
        
        # Validation
        if VALIDATE and len(val_dl) > 0:
            model.eval()
            with torch.no_grad():
                v_total, v_count = 0.0, 0
                for batch in val_dl:
                    if mode == 'vision':
                        img, state, cmd, dpos = batch
                        img = img.to(device)
                        state = state.to(device)
                        cmd = cmd.to(device)
                        dpos = dpos.to(device)
                        pred = model(img, state, cmd)
                        batch_size_actual = img.size(0)
                    else:
                        obs, dpos = batch
                        obs = obs.to(device)
                        dpos = dpos.to(device)
                        pred = model(obs)
                        batch_size_actual = obs.size(0)
                    v_total += nn.functional.mse_loss(pred, dpos, reduction='sum').item()
                    v_count += batch_size_actual
                val_loss = v_total / max(1, v_count)
            print(f"val_loss={val_loss:.6f}")
            wandb_log_maybe({'val/loss': val_loss, 'epoch': global_ep}, step=global_step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, opt, global_ep, global_step, out_dir, "best.pt")
            model.train()
        
        # sim eval
        if (global_step) // eval_every_n_steps != (global_step - steps_per_epoch) // eval_every_n_steps:
            save_checkpoint(model, opt, global_ep, global_step, out_dir, f"checkpoint_ep{global_ep}.pt")
            
            model.eval()
            eps = 100
            successes = eval_rollout(model, episodes=eps, timeout_s=5., mode=mode)
            rate = successes / max(1, eps)
            print(f"closed-loop eval @epoch {global_ep}: success {successes}/{eps} (rate={rate:.2f})")
            wandb_log_maybe({'eval/success_rate': rate, 'epoch': global_ep}, step=global_step)
            
            vector_field_path = os.path.join(out_dir, f"vector_field_ep{global_ep}.png")
            visualize_vector_field(model, vector_field_path, mode=mode)
            model.train()

    save_checkpoint(model, opt, global_ep, global_step, out_dir, "final.pt")
    if USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--out', type=str, default='ckpts', help='Output directory for checkpoints')
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to initialize from')
    parser.add_argument('--pretrained_backbone', type=str, default=None, help='Checkpoint for backbone')
    parser.add_argument('--num_demos', type=int, default=0, help='Number of demonstrations to use before split (0 = all)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (e.g. cuda:0)')
    parser.add_argument('--mode', type=str, default='vision', choices=['state', 'vision'], help='Training mode: state or vision')
    args = parser.parse_args()

    train(args.data, args.out, steps=args.steps, batch_size=args.batch_size, lr=args.lr, resume=args.resume, pretrained_backbone=args.pretrained_backbone, num_demos=args.num_demos, device=torch.device(args.device), mode=args.mode)
