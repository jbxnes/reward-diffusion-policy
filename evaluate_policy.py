import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import collections
from skvideo.io import vwrite
import wandb

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.pusht_dataset import PushTDataset
from diffusion_policy.environment.pusht_env import PushTEnv
from diffusion_policy.dataset.normalization import normalize_data, unnormalize_data


def evaluate(num_eps, model_path, save_vids, log_wandb=False):
    """Evaluate the policy on num_eps unseen environment initializations.
    
    Arguements: 
        num_eps (int): Number of environment initializations to evaluate the policy on.
        model_path (str): Path to the pre-trained noise prediction model.
        save_vids (bool): Whether to save videos of the evaluation.
        log_wandb (bool): Whether to log results to Weights & Biases.
    """
    
    # set seed
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
    
    eval_start_seed = 400000
    
    device = torch.device('cuda')
    
    # get dataset stats
    stats = PushTDataset(
        dataset_path='./data/pusht/pusht_cchi_v7_replay.zarr',
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon).stats
    
    # get environments
    env = PushTEnv(kp_obs=True)
    
    # get pre-trained noise prediction model 
    noise_pred_net = ConditionalUnet1D(input_dim=action_dim,
                                       global_cond_dim=obs_dim * obs_horizon) # base model
    state_dict = torch.load(model_path, map_location='cuda')
    noise_pred_net.load_state_dict(state_dict)
    noise_pred_net.to(device)
    noise_pred_net.eval()
    
    # get noise scheduler and alphas 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # eval loop 
    with tqdm(range(num_eps), desc='Evaluate') as tglobal: 
        max_rewards = []
        max_dense_rewards = []
        for ep_idx in tglobal:
            
            env.seed(eval_start_seed + ep_idx)
            obs, info = env.reset()

            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            
            if save_vids:
                imgs = [env.render(mode='rgb_array')]
                
            rewards = []
            dense_rewards = []
            step_idx = 0
            done = False
        
            with tqdm(total=max_steps, desc=f"Eval") as pbar:
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
                        naction = naction.detach().to('cpu').numpy() # (B, pred_horizon, action_dim)
                        naction = naction[0]
                        action_pred = unnormalize_data(naction, stats=stats['action']) # (pred_horizon, action_dim)

                        # only take action_horizon number of actions
                        start = obs_horizon - 1
                        end = start + action_horizon
                        action = action_pred[start:end,:] # (action_horizon, action_dim)
                        
                        for j in range(len(action)):
                            # stepping env
                            obs, reward, dense_reward, done, _, info = env.step(action[j])

                            # save observations
                            obs_deque.append(obs)
                            
                            # track results 
                            rewards.append(reward)
                            dense_rewards.append(dense_reward)
                            if save_vids:
                                imgs.append(env.render(mode='rgb_array'))

                            # update progress bar
                            step_idx += 1
                            pbar.update(1)
                            pbar.set_postfix(reward=reward)
                            if step_idx > max_steps:
                                done = True
                            if done:
                                break
                            
                    # track results
                    max_rewards.append(np.max(rewards))
                    max_dense_rewards.append(np.max(dense_rewards))
                    
                    if log_wandb:
                        wandb.log({
                            'coverage': np.max(rewards),
                            'running_avg_coverage': np.mean(max_rewards),
                            'reward': np.max(dense_rewards),
                            'running_avg_reward': np.mean(max_dense_rewards),
                            'episode': ep_idx,
                            'num_steps': step_idx
                        })
                        
                    # save video
                    if save_vids:
                        if not os.path.exists('./data/media'):
                            os.makedirs('./data/media')
                            
                        vwrite(f'./data/media/vis{ep_idx}.mp4', imgs)
                        print(f"SAVED MEDIA TO ./data/media/vis*.mp4")
                 
    print('AVERAGE COVERAGE:', np.mean(max_rewards))
    print('AVERAGE REWARD:', np.mean(max_dense_rewards))
    return np.mean(max_rewards), np.mean(max_dense_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_eps', default=50, type=int)
    parser.add_argument('--model_path', default='./data/checkpoints/policy_pretrained.ckpt', type=str)
    parser.add_argument('--save_vids', action='store_true')
    parsed_args = parser.parse_args()
    
    wandb.init(project="evaluate_policy", config=vars(parsed_args))
    
    # experiment inputs
    evaluate(num_eps=parsed_args.num_eps,
             model_path=parsed_args.model_path,
             save_vids=parsed_args.save_vids,
             log_wandb=True)