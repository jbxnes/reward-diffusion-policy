# Usage: python collect_rewards.py --n_inits 300 --model_path ./data/checkpoints/pusht_state_policy_ep100_pretrained42.ckpt --seed 42

import torch
import random
import numpy as np
from tqdm import tqdm
import argparse
import collections

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.dataset.pusht_dataset import PushTDataset
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.environment.pusht_env import PushTEnv
from diffusion_policy.dataset.normalization import normalize_data, unnormalize_data


def collect_rewards(n_inits, model_path, seed):
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # define variables
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 20
    action_dim = 2
    
    max_steps = 304
    num_diffusion_iters = 100
    
    reward_train_start_seed = 200000
    
    device = torch.device('cuda')
    
    # get pre-trained model
    noise_pred_net = ConditionalUnet1D(input_dim=action_dim, 
                                       global_cond_dim=obs_dim * obs_horizon)
    
    state_dict = torch.load(model_path, map_location='cuda')
    noise_pred_net.load_state_dict(state_dict)
    noise_pred_net.to(device)
    
    # get noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # get dataset stats
    stats = PushTDataset(
        dataset_path='./data/pusht/pusht_cchi_v7_replay.zarr',
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon).stats
    
    # get environment
    env = PushTEnv(kp_obs=True)
    
    # define data array that we will populate
    action_reward_data = np.zeros(((max_steps // action_horizon) * n_inits, 
                                   obs_dim + pred_horizon * action_dim + 2), 
                                  dtype=np.float32) # (38 * n_inits, 54)

    # collect data by doing n_init episodes
    n_policy_evals = 0
    for i in range(n_inits):
        
        # initalize environment 
        env.seed(reward_train_start_seed + i)
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Collecting Rewards for Init {i}") as pbar:
            while step_idx < max_steps:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque) # (2, 20)
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, pred_horizon, action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy() # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action']) # (pred_horizon, action_dim)

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:] # (action_horizon, action_dim)

                # collect action data 
                action_reward_data[n_policy_evals, :obs_dim] = obs_seq[-1]
                action_reward_data[n_policy_evals, obs_dim:-2] = np.array(action_pred).flatten()

                # execute action_horizon number of steps
                # without replanning
                dense_reward0 = env.dense_reward()
                for j in range(len(action)):
                    # stepping env
                    obs, reward, dense_reward, done, _, info = env.step(action[j])

                    # save observations
                    obs_deque.append(obs)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        break
                    
                # collect reward data labels 
                dense_rewardT = env.dense_reward()
                action_reward_data[n_policy_evals, -2] = dense_rewardT
                action_reward_data[n_policy_evals, -1] = dense_rewardT - dense_reward0
                n_policy_evals += 1
            
        # logging 
        wandb.log({'inits_progress': i / n_inits})
        wandb.log({'n_policy_evals': n_policy_evals})
        wandb.log({'step_idx': n_policy_evals})

    np.save(f'./data/reward/action_reward_data_{n_inits}inits.npy', action_reward_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_inits', default=300, type=int)
    parser.add_argument('--model_path', default='./data/checkpoints/pusht_state_policy_ep100_pretrained42.ckpt', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    import wandb
    wandb.init(project="reward_data_collection", config=vars(parsed_args))
    
    # experiment inputs
    collect_rewards(n_inits=parsed_args.n_inits,
                    model_path=parsed_args.model_path, 
                    seed=parsed_args.seed)
