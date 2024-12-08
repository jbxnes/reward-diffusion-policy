import os 
import torch
import torch.nn as nn
import numpy as np 
import argparse
from tqdm import tqdm
import collections
import wandb

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.reward_dataset import RewardDataset
from diffusion_policy.reward_model import RewardModel
from diffusion_policy.dataset.pusht_dataset import PushTDataset, state_to_kps
from diffusion_policy.dataset.normalization import normalize_data, unnormalize_data
from diffusion_policy.environment.pusht_env import PushTEnv
from diffusion_policy.adjoint_matching_utils import solve_lean_adjoint, adjoint_matching_loss


def adjoint_matching(num_ft_iters, num_eps, warmup_steps, lr, reward_diff, noise_pred_model_path, 
                     reward_model_path, action_reward_path, clip_adjoint, log_wandb=False):
    """Finetune the diffusion policy using the Adjoint Matching algorithm. 
    Domingo-Enrich et al. (2024): https://arxiv.org/abs/2409.08861, Algorithm 2
    
    Arguments:
        num_ft_iters (int): Number of finetuning iterations.
        num_eps (int): Number of episodes to run in parallel.
        warmup_steps (int): Number of warmup steps for the noise scheduler.
        lr (float): Learning rate.
        reward_diff (bool): Whether the reward model should predict reward difference.
        noise_pred_model_path (str): Path to the pre-trained noise prediction model.
        reward_model_path (str): Path to the pre-trained reward model.
        action_reward_path (str): Path to the action reward data.
        clip_adjoint (bool): Whether to clip the lean adjoint samples.
        log_wandb (bool): Whether to log training results to Weights & Biases. 
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
    
    reward_ft_start_seed = 300000
    
    device = torch.device('cuda')
    
    # get dataset stats
    pusht_stats = PushTDataset(
        dataset_path='./data/pusht/pusht_cchi_v7_replay.zarr',
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon).stats
    
    reward_stats = RewardDataset(
        action_reward_path=action_reward_path, 
        train=True,
        reward_diff=reward_diff).stats
    
    # get environments
    envs = [PushTEnv(kp_obs=True) for _ in range(num_eps)]
    
    # load pre-trained reward model
    reward_model = RewardModel(reward_diff=reward_diff)
    state_dict = torch.load(reward_model_path, map_location='cuda')
    reward_model.load_state_dict(state_dict)
    reward_model.to(device)
    
    # load pre-trained noise prediction model
    # we need to make two copies, one of which will be finetuned 
    noise_pred_net_pt = ConditionalUnet1D(input_dim=action_dim, # base model
                                          global_cond_dim=obs_dim * obs_horizon) 
    noise_pred_net_ft = ConditionalUnet1D(input_dim=action_dim, # finetuned model
                                          global_cond_dim=obs_dim * obs_horizon) 
    
    state_dict = torch.load(noise_pred_model_path, map_location='cuda')
    
    noise_pred_net_pt.load_state_dict(state_dict)
    noise_pred_net_pt.to(device)
    
    noise_pred_net_ft.load_state_dict(state_dict)
    noise_pred_net_ft.to(device)
    
    # get noise scheduler and alphas 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    alphas = noise_scheduler.alphas
    
    # define optimizer
    # notice that we are only updating parameters of the finetuned model
    optimizer = torch.optim.AdamW(
        params=noise_pred_net_ft.parameters(),
        lr=lr, weight_decay=1e-6)
    
    # get LR scheduler 
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_ft_iters,
    ) 
    
    # training loop
    with tqdm(range(num_ft_iters), desc='Train Iter') as tglobal: 
        for iter_idx in tglobal:
            with tqdm(total=max_steps, desc=f"Running episode") as pbar:
            
                # initalize environments
                for i in range(num_eps):
                    envs[i].seed(reward_ft_start_seed + i)
                
                # get inital observations
                obs = [envs[i].reset()[0] for i in range(num_eps)] 
                obs = np.stack(obs) # (num_eps, 20)
                
                # keep a queue of last 2 steps of observations
                obs_deque = collections.deque(
                    [obs] * obs_horizon, maxlen=obs_horizon) # 2 * (num_eps, 20)
                
                ### 1. GENERATE TRAJECTORIES ### 
                # define data arrays that we will populate 
                trajectories = torch.zeros((num_eps * (max_steps // action_horizon), 
                                            num_diffusion_iters + 1, pred_horizon, action_dim))
                obs_action_data = torch.zeros((num_eps * (max_steps // action_horizon), 
                                                obs_horizon * obs_dim + pred_horizon * action_dim))
                
                step_idx = 0
                n_policy_evals = 0

                while step_idx < max_steps:
                    B = num_eps
                    
                    # stack the last obs_horizon (2) number of observations
                    obs_seq = np.stack(obs_deque, axis=1) # (B, obs_horizon, obs_dim)
                    nobs = normalize_data(obs_seq, stats=pusht_stats['obs'])
                    nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32) 
                    
                    # collect trajectories for this policy evaluation 
                    num_traj = []
                    
                    with torch.no_grad():
                        obs_cond = nobs.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
                        
                        noisy_action = torch.randn(
                            (B, pred_horizon, action_dim), device=device) # (B, pred_horizon, action_dim)
                        num_traj.append(noisy_action) 
                        
                        noise_scheduler.set_timesteps(num_diffusion_iters)
                        
                        for k in noise_scheduler.timesteps:
                            noise_pred = noise_pred_net_ft(
                                sample=num_traj[-1],
                                timestep=k,
                                global_cond=obs_cond
                            )
                            
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=num_traj[-1]
                            ).prev_sample
                            
                            num_traj.append(naction)
                        
                        # collect trajectories from policy eval 
                        trajectories[n_policy_evals:n_policy_evals + B, :] = torch.stack(num_traj, dim=1) # (B, num_diffusion_iters + 1, pred_horizon, action_dim)
                        
                        # unnormalize action
                        naction = num_traj[-1].detach().to('cpu').numpy() # (B, pred_horizon, action_dim)
                        action_pred = unnormalize_data(naction, stats=pusht_stats['action']) # (B, pred_horizon, action_dim)

                        # only take action_horizon number of actions
                        start = obs_horizon - 1
                        end = start + action_horizon
                        action = action_pred[:, start:end,:] # (B, action_horizon, action_dim)

                        # collect reward data
                        # (num_eps * (max_steps // action_horizon), obs_horizon * obs_dim + pred_horizon * action_dim)
                        obs_action_data[n_policy_evals:n_policy_evals + B, :obs_horizon * obs_dim] = obs_cond # (B, obs_horizon * obs_dim)
                        obs_action_data[n_policy_evals:n_policy_evals + B, obs_horizon * obs_dim:] = torch.from_numpy(action_pred).flatten(start_dim=1) # (B, pred_horizon * action_dim)
                        
                        n_policy_evals += B
                        
                        # execute action_horizon number of steps
                        for i in range(action_horizon):
                            obs = [envs[j].step(action[j][i])[0] for j in range(num_eps)]
                            obs = np.stack(obs) # (num_eps, 20)
                            obs_deque.append(obs)
                            
                            step_idx += 1
                            pbar.update(1)
                            if step_idx >= max_steps:
                                break
            
            ### 2. SOLVE LEAN ADJOINT ODE ###
            # predict rewards
            obs_action_input = np.array(obs_action_data[:, obs_dim:].cpu()) # (num_eps * (max_steps // action_horizon), obs_dim)
            obs_action_input = normalize_data(np.array(obs_action_input), stats=reward_stats)
            obs_action_input = torch.from_numpy(obs_action_input).to(device, dtype=torch.float32)
            obs_action_input.requires_grad = True
            
            rewards = reward_model(obs_action_input) # (num_eps * (max_steps // action_horizon),)
            
            # get gradient of rewards wrt actions
            rewards.sum().backward()
            reward_gradient = obs_action_input.grad[:, obs_dim:] # (num_eps * (max_steps // action_horizon), pred_horizon * action_dim)
            
            # solve lean adjoint ode 
            obs_cond = obs_action_data[:, :obs_horizon * obs_dim] # (num_eps * (max_steps // action_horizon), obs_horizon * obs_dim)  
            obs_cond = obs_cond.to(device)
            trajectories = trajectories.to(device)
            alphas = alphas.to(device)
        
            lean_adjoint = solve_lean_adjoint(trajectories=trajectories, 
                                              obs_cond=obs_cond, 
                                              alphas=alphas, 
                                              reward_gradient=reward_gradient, 
                                              noise_pred_net=noise_pred_net_pt, 
                                              clip_adjoint=clip_adjoint,
                                              device=device)
            # (num_eps * (max_steps // action_horizon), num_diffusion_iters, pred_horizon, action_dim)
            
            ### 3. COMPUTE ADJOINT MATCHING LOSS LOSS ###
            loss = adjoint_matching_loss(trajectories=trajectories, 
                                         lean_adjoint=lean_adjoint, 
                                         obs_cond=obs_cond, 
                                         noise_pred_net_ft=noise_pred_net_ft, 
                                         noise_pred_net_pt=noise_pred_net_pt, 
                                         alphas=alphas, 
                                         device=device)
            
            # optimize 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # logging 
            train_loss = loss.item()
            print(f"\nTRAIN LOSS | ITER {iter_idx}: {train_loss}\n")
            
            if log_wandb:
                wandb.log({'train_loss': train_loss,
                            'train_iter': iter_idx,
                            'warmup_steps': warmup_steps,
                            'inital_lr': lr,
                            'clip_adjoint': int(clip_adjoint),
                            'lr': optimizer.param_groups[0]['lr'],
                            'reward_difference': int(reward_diff)})

    # save model
    if not os.path.exists('./data/checkpoints'):
        os.makedirs('./data/checkpoints')
        
    torch.save(noise_pred_net_ft.state_dict(), f'./data/checkpoints/policy_finetuned_warmup{warmup_steps}_rd{int(reward_diff)}_clip{int(clip_adjoint)}.ckpt')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_ft_iters', default=100, type=int)
    parser.add_argument('--num_eps', default=10, type=int)
    parser.add_argument('--warmup_steps', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--reward_diff', action='store_true')
    parser.add_argument('--noise_pred_model_path', default='./data/checkpoints/policy_pretrained.ckpt', type=str)
    parser.add_argument('--reward_model_path', default='./data/checkpoints/reward_model_rd0.ckpt', type=str)
    parser.add_argument('--action_reward_path', default='./data/reward/action_reward_data_300eps.npy', type=str)
    parser.add_argument('--clip_adjoint', action='store_true')
    parsed_args = parser.parse_args()
    
    print(parsed_args.reward_diff)
    
    wandb.init(project="reward_finetuning", config=vars(parsed_args))
    
    # experiment inputs
    adjoint_matching(num_ft_iters=parsed_args.num_ft_iters,
                     num_eps=parsed_args.num_eps,
                     warmup_steps=parsed_args.warmup_steps,
                     lr=parsed_args.lr,
                     reward_diff=parsed_args.reward_diff,
                     noise_pred_model_path=parsed_args.noise_pred_model_path,
                     reward_model_path=parsed_args.reward_model_path,
                     action_reward_path=parsed_args.action_reward_path,
                     clip_adjoint=parsed_args.clip_adjoint,
                     log_wandb=True)
    