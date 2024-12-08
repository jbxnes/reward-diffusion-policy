import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import collections
import random

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.pusht_dataset import PushTDataset
from diffusion_policy.environment.pusht_env import PushTEnv
from diffusion_policy.dataset.normalization import normalize_data, unnormalize_data


def collect_rewards(num_eps, model_path):
    """Collect synthetic data for reward model training by running the pre-trained policy
    for num_eps episodes. 
    
    The collected data is stored in a numpy array with the following columns
    - observation (obs_dim).
    - action sequence generated by policy (pred_horizon * action_dim).
    - ground truth reward after executing action_horizon actions (1).
    - difference in reward after executing action_horizon actions (1).
    
    Arguements:
        num_eps (int): Number of episodes to run the policy for.
        model_path (str): Path to the pre-trained policy.
    """
    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # define variables
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 20
    action_dim = 2
    
    max_steps = 304
    num_diffusion_iters = 100
    
    reward_data_start_seed = 200000
    
    device = torch.device('cuda')
    
    # get dataset stats
    pusht_stats = PushTDataset(
        dataset_path='./data/pusht/pusht_cchi_v7_replay.zarr',
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon).stats
    
    # get environment
    envs = [PushTEnv(kp_obs=True) for _ in range(num_eps)]
    
    # get pre-trained noise prediction model
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

    # collect data by doing n_init episodes
    with tqdm(total=max_steps, desc="Collecting Rewards") as pbar:
        
        # initialize environments
        for i in range(num_eps):
            envs[i].seed(reward_data_start_seed + i)
        
        # get inital observations
        obs = [envs[i].reset()[0] for i in range(num_eps)]
        obs = np.stack(obs) # (num_eps, 20)
        
        # keep a queue of the last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon) # 2 * (num_eps, 20)
        
        # define data array that we will populate
        action_reward_data = np.zeros(((max_steps // action_horizon) * num_eps, 
                                    obs_dim + pred_horizon * action_dim + 2), 
                                    dtype=np.float32) # (38 * num_eps, 54)
        
        
        step_idx = 0
        n_policy_evals = 0
        
        # generate data
        while step_idx < max_steps:
            B = num_eps
            
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque, axis=1) # (B, obs_horizon, obs_dim)
            nobs = normalize_data(obs_seq, stats=pusht_stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32) 
            
            # infer action
            with torch.no_grad():
                obs_cond = nobs.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
                
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action
                
                noise_scheduler.set_timesteps(num_diffusion_iters)
                
                for k in noise_scheduler.timesteps:
                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy() # (B, pred_horizon, action_dim)
                action_pred = unnormalize_data(naction, stats=pusht_stats['action']) # (B, pred_horizon, action_dim)
                
                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[:, start:end, :] # (B, action_horizon, action_dim)
                
                # collect reward data inputs,
                action_reward_data[n_policy_evals:n_policy_evals + B, :obs_dim] = obs_seq[:, -1]
                action_reward_data[n_policy_evals:n_policy_evals + B, obs_dim:-2] = action_pred.reshape(B, -1)
                
                # compute initial reward
                dense_rewards0 = [envs[i].dense_reward() for i in range(num_eps)]
                dense_rewards0 = np.stack(dense_rewards0) # (num_eps,)
                
                # execute action_horizon number of steps 
                for i in range(action_horizon):
                    obs = [envs[j].step(action[j][i])[0] for j in range(num_eps)]
                    obs = np.stack(obs) # (num_eps, 20)
                    obs_deque.append(obs)
                    
                    step_idx += 1
                    pbar.update(1)
                    if step_idx > max_steps:
                        break
                    
                dense_rewardsT = [envs[i].dense_reward() for i in range(num_eps)]
                dense_rewardsT = np.stack(dense_rewardsT) # (num_eps,)
                
                action_reward_data[n_policy_evals:n_policy_evals + B, -2] = dense_rewardsT
                action_reward_data[n_policy_evals:n_policy_evals + B, -2] = dense_rewardsT - dense_rewards0
                    
                n_policy_evals += B
    
    # save data 
    if not os.path.exists('./data/reward'):
        os.makedirs('./data/reward')
    
    np.save(f'./data/reward/action_reward_data_{num_eps}eps.npy', action_reward_data)
    print(f"REWARD DATA SAVED to ./data/reward/action_reward_data_{num_eps}eps.npy")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_eps', default=300, type=int)
    parser.add_argument('--model_path', default='./data/checkpoints/policy_pretrained.ckpt', type=str)
    parsed_args = parser.parse_args()
    
    # experiment inputs
    collect_rewards(num_eps=parsed_args.num_eps,
                    model_path=parsed_args.model_path)
