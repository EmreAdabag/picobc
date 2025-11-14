import argparse
import pickle
import h5py
import numpy as np
import torch
from env import PickAndPlaceEnv
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--obj_lo", type=int, default=0)
    parser.add_argument("--obj_hi", type=int, default=2)
    parser.add_argument("--goal_lo", type=int, default=0)
    parser.add_argument("--goal_hi", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output", type=str, default="probe_data.h5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    cfg.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(args.model_path, config=cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    agent_positions = []
    object_positions = []
    goal_positions = []
    picked_flags = []
    object_ids = []
    goal_ids = []
    image_activations = []
    lang_activations = []
    lang_masks = []
    env_states = []
    
    max_lang_len = 0

    for oid in range(args.obj_lo, args.obj_hi):
        for gid in range(args.goal_lo, args.goal_hi):
            print(oid, gid)
            if oid == gid:
                continue
            
            for batch_idx in range(args.num_batches):
                env = PickAndPlaceEnv(batch_size=args.batch_size, device=device, record_video=False, seed=args.seed)
                env.reset(obj_idx=oid, goal_idx=gid)
                policy.reset()
                task_str = env.command()
                tasks = [task_str for _ in range(args.batch_size)]

                frames = env.current_frame().to(torch.float32) / 255.0
                frames = frames.permute(0, 3, 1, 2).contiguous()
                rob_pos = env.agent_pos.to(torch.float32)
                g_abs = env.gripper_closed.to(torch.float32).unsqueeze(-1)
                robot_state = torch.cat([rob_pos, g_abs], dim=-1)

                batch = {
                    "observation.images.camera1": frames,
                    "observation.state": robot_state,
                    "task": tasks,
                }

                proc = preprocessor(batch)
                policy.select_action(proc)
                prefix_embeds = policy.model.prefix_output_embeds
                boundaries = policy.model.prefix_boundaries
                
                # Extract image and language embeddings separately
                img_embeds = prefix_embeds[:, :boundaries['images_end'], :]
                lang_embeds = prefix_embeds[:, boundaries['images_end']:boundaries['lang_end'], :]
                
                # Track max language length
                lang_len = lang_embeds.shape[1]
                max_lang_len = max(max_lang_len, lang_len)

                agent_positions.append(env.agent_pos.cpu().numpy())
                object_positions.append(env.object_pos.cpu().numpy())
                goal_positions.append(env.goal_pos.cpu().numpy())
                picked_flags.append(env.picked.cpu().numpy())
                object_ids.append(np.full(args.batch_size, oid, dtype=np.int32))
                goal_ids.append(np.full(args.batch_size, gid, dtype=np.int32))
                image_activations.append(img_embeds.cpu().float().numpy())
                lang_activations.append(lang_embeds.cpu().float().numpy())
                
                env_state = env.get_env_state()
                env_states.append(env_state)

    # Pad language activations to max length
    padded_lang_activations = []
    for lang_act in lang_activations:
        batch_size, curr_len, hidden_dim = lang_act.shape
        if curr_len < max_lang_len:
            pad_len = max_lang_len - curr_len
            padding = np.zeros((batch_size, pad_len, hidden_dim), dtype=lang_act.dtype)
            padded = np.concatenate([lang_act, padding], axis=1)
            mask = np.concatenate([np.ones((batch_size, curr_len)), np.zeros((batch_size, pad_len))], axis=1)
        else:
            padded = lang_act
            mask = np.ones((batch_size, curr_len))
        padded_lang_activations.append(padded)
        lang_masks.append(mask)
    
    agent_positions = np.concatenate(agent_positions, axis=0)
    object_positions = np.concatenate(object_positions, axis=0)
    goal_positions = np.concatenate(goal_positions, axis=0)
    picked_flags = np.concatenate(picked_flags, axis=0)
    object_ids = np.concatenate(object_ids, axis=0)
    goal_ids = np.concatenate(goal_ids, axis=0)
    image_activations = np.concatenate(image_activations, axis=0)
    lang_activations = np.concatenate(padded_lang_activations, axis=0)
    lang_masks = np.concatenate(lang_masks, axis=0)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("agent_pos", data=agent_positions)
        f.create_dataset("object_pos", data=object_positions)
        f.create_dataset("goal_pos", data=goal_positions)
        f.create_dataset("picked", data=picked_flags)
        f.create_dataset("object_id", data=object_ids)
        f.create_dataset("goal_id", data=goal_ids)
        f.create_dataset("image_activations", data=image_activations)
        f.create_dataset("lang_activations", data=lang_activations)
        f.create_dataset("lang_masks", data=lang_masks)

    pickle_path = args.output.replace(".h5", "_env_states.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(env_states, f)

    print(f"Saved {len(agent_positions)} samples to {args.output}")
    print(f"Saved {len(env_states)} environment states to {pickle_path}")


if __name__ == "__main__":
    main()

