# Usage: python collect_reward_data.py --n_inits 300 --model_path ./data/checkpoints/pusht_state_policy_ep100_pretrained42.ckpt --seed 42

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
    random.seed(seed) 
    
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
    
    # define data arrays that we will populate 
    state_action_data = np.zeros(((max_steps // action_horizon) * n_inits, obs_dim + action_dim * action_horizon), dtype=np.float32) # (38 * n_inits, 36)
    reward_data = np.zeros((max_steps * n_inits, 2), dtype=np.float32) # (38 * n_inits, 2)

    # collect data by doing n_init episodes
    for i in range(n_inits):
        
        # initalize environment 
        env.seed(reward_train_start_seed + i)
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        
        done = False
        step_idx = 0
        n_policy_evals = 0

        with tqdm(total=max_steps, desc=f"Collecting Rewards for Init {i}") as pbar:
            while not done:
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
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action']) # list of shape (16, 2)

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:] # (action_horizon, action_dim)

                # collect reward data inputs 
                state_action_data[n_policy_evals, :20] = obs_seq[-1]
                state_action_data[n_policy_evals, 20:] = np.array(action).flatten()

                # execute action_horizon number of steps
                # without replanning
                dense_reward0 = env.dense_reward()
                dense_rewardT = 0
                for j in range(len(action)):
                    # stepping env
                    obs, reward, dense_reward, done, _, info = env.step(action[j])
                    dense_rewardT = dense_reward

                    # save observations
                    obs_deque.append(obs)

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
                    
                # collect reward data labels 
                reward_data[n_policy_evals, 0] = dense_rewardT
                reward_data[n_policy_evals, 1] = dense_rewardT - dense_reward0
                n_policy_evals += 1
                
        wandb.log({'progress': i / n_inits})

    np.save('./data/reward/state_action_data.npy', state_action_data)
    np.save('./data/reward/reward_data.npy', reward_data) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_inits', default=1000, type=int)
    parser.add_argument('--model_path', default='./data/checkpoints/pusht_state_policy_ep100_pretrained42.ckpt', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    import wandb
    from datetime import datetime
    run_name = f"run_seed{parsed_args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="reward_data_collection", name=run_name, config=vars(parsed_args)) 
    
    print(f"RUNNING: {run_name}")
    
    # experiment inputs
    collect_rewards(n_inits=parsed_args.n_inits,
                    model_path=parsed_args.model_path,
                    seed=parsed_args.seed)
