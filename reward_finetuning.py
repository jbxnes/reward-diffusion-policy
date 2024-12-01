# Usage: python reward_finetuning.py --num_epochs 100 --num_eps 10 --lr 1e-2 --reward_diff --noise_pred_model_path ./data/checkpoints/policy_kps_pretrained_seed42.ckpt --reward_model_path ./data/checkpoints/reward_model_kps_rd0_seed42.ckpt --action_reward_path ./data/reward/action_reward_data_300inits.npy --seed 42

import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
import argparse
import collections

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.dataset.pusht_dataset import PushTDataset, state_to_kps
from diffusion_policy.dataset.reward_dataset import RewardDataset
from diffusion_policy.model.reward import RewardModel
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalization import normalize_data, unnormalize_data
from diffusion_policy.environment.pusht_env import PushTEnv


def adjoint_matching_loss(trajectories, lean_adjoint_trajectories, 
                          obs_cond, noise_pred_net_ft, noise_pred_net_pt, alphas, device):
    # trajectories: (num_eps * num_policy_evals, num_diffusion_iters, pred_horizon, action_dim)
    # alphas: (num_diffusion_iters,)
    # reward_action_gradients: (num_eps * num_policy_evals, pred_horizon * action_dim)
    # obs_cond: (num_eps * num_policy_evals, obs_horison * obs_dim)
    num_diffusion_iters = trajectories.shape[1]
    
    loss = 0
    
    for t in range(1, num_diffusion_iters - 1):
        # 1. compute alphas, betas
        prev_t = t - 1
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_prod_t = alpha_cumprod[t] # alpha bar k
        alpha_prod_t_prev = alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0) # alpha bar k+1
        
        # compute coefficient for noise prediction difference
        noise_coeff = alpha_prod_t_prev / (alpha_prod_t * (1 - alpha_prod_t_prev))
        noise_coeff = noise_coeff * (1 - (alpha_prod_t / alpha_prod_t_prev))
        noise_coeff = noise_coeff ** (0.5)
        
        # compute noise prediction difference between ft and pt models
        noise_diff = noise_pred_net_ft(sample=trajectories[:, t],
                                       timestep=t,
                                       global_cond=obs_cond)
        noise_diff -= noise_pred_net_pt(sample=trajectories[:, t],
                                        timestep=t,
                                        global_cond=obs_cond)
        noise_diff = noise_diff.flatten(start_dim=1)
         
        # compute coefficient for lean adjoint trajectories
        lean_adjoint_coeff = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        lean_adjoint_coeff = lean_adjoint_coeff * (1 - (alpha_prod_t / alpha_prod_t_prev))
        lean_adjoint_coeff = lean_adjoint_coeff ** (0.5)

        print(lean_adjoint_trajectories[:, t].max(), lean_adjoint_trajectories[:, t].min())
        
        # add loss term
        loss_term = (noise_coeff * noise_diff - lean_adjoint_coeff * lean_adjoint_trajectories[:, t]) ** 2
        loss += loss_term.sum()
        
    return loss
        

# def remove_noise(noise_pred, timestep, sample, alphas):
#     # alphas go from small to large here 
#     t = timestep 
#     prev_t = t - 1
    
#     # compute alphas, betas
#     alpha_cumprod = torch.cumprod(alphas, dim=0)
#     alpha_prod_t = alpha_cumprod[t]
#     alpha_prod_t_prev = alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
#     beta_prod_t = 1 - alpha_prod_t
#     beta_prod_t_prev = 1 - alpha_prod_t_prev
#     current_alpha_t = alpha_prod_t / alpha_prod_t_prev
#     current_beta_t = 1 - current_alpha_t
    
#     # compute predicted original sample from predicted noise
#     pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

#     # clip pred_original_sample
#     # TODO should i be clipping here?
#     pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)
    
#     # compute coefficients for pred_original_sample x_0 and current sample x_t
#     pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
#     current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

#     # compute predicted previous sample µ_t
#     pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    
#     return pred_prev_sample

    
def lean_adjoint(trajectories, obs_cond, alphas, reward_gradient, noise_pred_net, clamp_adjoint, device):
    # trajectories: (num_eps * num_policy_evals, num_diffusion_iters, pred_horizon, action_dim)
    # alphas: (num_diffusion_iters,)
    # reward_action_gradients: (num_eps * num_policy_evals, pred_horizon * action_dim)
    # obs_cond: (num_eps * num_policy_evals, obs_horison * obs_dim)
    
    num_traj = trajectories.shape[0]
    num_diffusion_iters = trajectories.shape[1]
    
    # initialize a_k array which we will populate 
    # (num_eps * num_policy_evals, num_diffusion_iters, pred_horizon * action_dim)
    lean_adjoint_traj = torch.zeros((num_traj, num_diffusion_iters, reward_gradient.shape[-1]), device=device) 
    lean_adjoint_traj[:, -1] = reward_gradient
    
    for k in range(num_diffusion_iters - 2, -1, -1): 
        
        # define function to help with gradient computation 
        # def func(sample):
        #     noise_pred = noise_pred_net(
        #         sample=sample, # (num_eps * num_policy_evals, pred_horizon, action_dim)
        #         timestep=k,
        #         global_cond=obs_cond # (num_eps * num_policy_evals, obs_horizon * obs_dim)
        #     )
            
        #     return remove_noise(noise_pred, k, sample, alphas) - sample 
        
        def func(sample):
            alpha_cumprod = torch.cumprod(alphas, dim=0)
            alpha_prod_t = alpha_cumprod[t]
            alpha_prod_t_prev = alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # compute gradient 
        noised_action_k = trajectories[:, k] # (num_eps * num_policy_evals, pred_horizon, action_dim)
        noised_action_k.requires_grad = True
        
        output = func(noised_action_k)
        
        grad_outputs = torch.ones_like(output)
        action_gradient_k = torch.autograd.grad(outputs=output, inputs=noised_action_k, grad_outputs=grad_outputs)[0] # (num_eps * num_policy_evals, pred_horizon, action_dim)
        action_gradient_k = action_gradient_k.flatten(start_dim=1) # (num_eps * num_policy_evals, pred_horizon * action_dim)
        
        # compute lean adjoint at time k 
        lean_adjoint_traj[:, k] = lean_adjoint_traj[:, k+1] + (lean_adjoint_traj[:, k+1] * action_gradient_k).sum(dim=-1, keepdim=True) # (num_eps * num_policy_evals,)
        if clamp_adjoint:
            lean_adjoint_traj[:, k] = lean_adjoint_traj[:, k].clamp(-1.0, 1.0)
    
    return lean_adjoint_traj


def adjoint_matching(num_epochs, num_eps, lr, reward_diff, 
                     noise_pred_model_path, reward_model_path, action_reward_path, 
                     clamp_adjoint, seed):
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
    noise_pred_net_pt = ConditionalUnet1D(input_dim=action_dim, 
                                          global_cond_dim=obs_dim * obs_horizon) # base model
    noise_pred_net_ft = ConditionalUnet1D(input_dim=action_dim, 
                                          global_cond_dim=obs_dim * obs_horizon) # finetuned model
    
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
        num_warmup_steps=500,
        num_training_steps=num_epochs * (max_steps // action_horizon),
    )
    
    # training loop
    trajectory_seed = 0
    with tqdm(range(num_epochs), desc='Epoch') as tglobal: 
        for epoch_idx in tglobal:
            
            # initalizing environments
            for i in range(num_eps):
                envs[i].seed(reward_ft_start_seed + trajectory_seed)
                trajectory_seed += 1
            
            # getting inital observations
            obs = [envs[i].reset()[0] for i in range(num_eps)] # num_eps * (20)
            obs = np.stack(obs) # (num_traj, 20)
            
            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon) # 2 * (num_eps, 20)
            
            ### GENERATE TRAJECTORIES ### 
            # define data arrays that we will populate 
            trajectories = torch.zeros((num_eps * (max_steps // action_horizon), 
                                        num_diffusion_iters + 1, pred_horizon, action_dim))
            obs_action_data = torch.zeros((num_eps * (max_steps // action_horizon), 
                                            obs_horizon * obs_dim + pred_horizon * action_dim))
            
            step_idx = 0
            n_policy_evals = 0

            with tqdm(total=max_steps, desc=f"Running episode") as pbar:
                    while step_idx < max_steps:
                        B = num_eps
                        
                        # stack the last obs_horizon (2) number of observations
                        obs_seq = np.stack(obs_deque, axis=1) # (B, 2, 20)
                        nobs = normalize_data(obs_seq, stats=pusht_stats['obs'])
                        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32) # (B, 2, 20)
                        
                        # generate set of trajectories 
                        num_traj = []
                        
                        with torch.no_grad():
                            obs_cond = nobs.flatten(start_dim=1) # (B, 2*20)
                            noisy_action = torch.randn(
                                (B, pred_horizon, action_dim), device=device) # (B, pred_horizon, action_dim)
                            
                            num_traj.append(noisy_action) # list of (B, pred_horizon, action_dim)
                            
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
                            
                            trajectories[n_policy_evals:n_policy_evals + B, :] = torch.stack(num_traj, dim=1) # (B, num_diffusion_iters + 1, pred_horizon, action_dim)
                            
                            # unnormalize action
                            naction = naction.detach().to('cpu').numpy() # (B, pred_horizon, action_dim)
                            action_pred = unnormalize_data(naction, stats=pusht_stats['action']) # (B, pred_horizon, action_dim)

                            # only take action_horizon number of actions
                            start = obs_horizon - 1
                            end = start + action_horizon
                            action = action_pred[:, start:end,:] # (B, action_horizon, action_dim)

                            # collect reward data inputs 
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
            ### END GENERATE TRAJECTORIES ###
            
            ### COMPUTE REWARD GRADIENTS ###
            obs_action_input = np.array(obs_action_data[:, obs_dim:].cpu()) # (num_eps * (max_steps // action_horizon), obs_dim)
            obs_action_input = normalize_data(np.array(obs_action_input), stats=reward_stats)
            obs_action_input = torch.from_numpy(obs_action_input).to(device, dtype=torch.float32)
            obs_action_input.requires_grad = True
            
            rewards = reward_model(obs_action_input) # (num_eps * (max_steps // action_horizon),)
            
            grad_outputs = torch.ones_like(rewards)
            
            # get gradient of rewards wrt actions
            reward_gradient = torch.autograd.grad(outputs=rewards, inputs=obs_action_input, grad_outputs=grad_outputs)[0] # (num_eps * num_policy_evals, pred_horizon, action_dim)        
            reward_gradient = reward_gradient[:, obs_dim:]
            ### END COMPUTE REWARD GRADIENTS ###
            
            ### SOLVE LEAN ADJOINT ODE ###
            obs_cond = obs_action_data[:, :obs_horizon * obs_dim] # (num_eps * (max_steps // action_horizon), obs_horizon * obs_dim)  
            obs_cond = obs_cond.to(device)
            trajectories = trajectories.to(device)
            alphas = alphas.to(device)
        
            lean_adjoint_traj = lean_adjoint(trajectories=trajectories, 
                                             obs_cond=obs_cond, 
                                             alphas=alphas, 
                                             reward_gradient=reward_gradient, 
                                             noise_pred_net=noise_pred_net_pt, 
                                             clamp_adjoint=clamp_adjoint,
                                             device=device)
            # (num_eps * (max_steps // action_horizon), num_diffusion_iters, pred_horizon, action_dim)
            ### END SOLVE LEAN ADJOINT ODE ###
            
            ### COMPUTE LOSS ###
            optimizer.zero_grad()
            
            adjoint_loss = adjoint_matching_loss(trajectories=trajectories, 
                                                lean_adjoint_trajectories=lean_adjoint_traj, 
                                                obs_cond=obs_cond, 
                                                noise_pred_net_ft=noise_pred_net_ft, 
                                                noise_pred_net_pt=noise_pred_net_pt, 
                                                alphas=alphas, 
                                                device=device)
            adjoint_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            loss_cpu = adjoint_loss.item()
            train_loss = loss_cpu / (n_policy_evals * num_eps)
            print(f"\nTRAIN LOSS | EPOCH {epoch_idx}: {train_loss}\n")
            
            # log results
            wandb.log({'train_loss': train_loss,
                       'epoch': epoch_idx,
                       'num_eps': num_eps,
                       'inital_lr': lr,
                       'reward_difference': int(reward_diff)})
    
    # save model
    torch.save(noise_pred_net_ft.state_dict(), f'./data/checkpoints/policy_kps_finetuned_seed{seed}_rd{int(reward_diff)}.ckpt')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_eps', default=10, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--reward_diff', action='store_true')
    parser.add_argument('--noise_pred_model_path', default='./data/checkpoints/policy_kps_pretrained_seed42.ckpt', type=str)
    parser.add_argument('--reward_model_path', default='./data/checkpoints/reward_model_kps_rd0_seed42.ckpt', type=str)
    parser.add_argument('--action_reward_path', default='./data/reward/action_reward_data_300inits.npy', type=str)
    parser.add_argument('--clamp_adjoint', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    import wandb
    wandb.init(project="reward_finetuning", config=vars(parsed_args))
    
    # experiment inputs
    adjoint_matching(num_epochs=parsed_args.num_epochs,
                     num_eps=parsed_args.num_eps,
                     lr=parsed_args.lr,
                     reward_diff=parsed_args.reward_diff,
                     noise_pred_model_path=parsed_args.noise_pred_model_path,
                     reward_model_path=parsed_args.reward_model_path,
                     action_reward_path=parsed_args.action_reward_path,
                     clamp_adjoint=parsed_args.clamp_adjoint,
                     seed=parsed_args.seed)
    