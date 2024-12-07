import torch


def adjoint_matching_loss(trajectories, lean_adjoint, obs_cond, noise_pred_net_ft, noise_pred_net_pt, alphas, device):
    """Compute the adjoint matching objective.
    Domingo-Enrich et al. (2024): https://arxiv.org/abs/2409.08861
    
    Arguments:
        trajectories (torch.Tensor): A batch of trajectories from the diffusion process. (num_traj, num_diffusion_iters, pred_horizon, action_dim)
        lean_adjoint (torch.Tensor): The lean adjoint samples. (num_traj, num_diffusion_iters, pred_horizon * action_dim)
        obs_cond (torch.Tensor): A batch of observations to condition the diffusion process. (num_traj, obs_horizon * obs_dim)
        noise_pred_net_ft (torch.nn.Module): The noise prediction network at the fine-tuning step.
        noise_pred_net_pt (torch.nn.Module): The noise prediction network at the pre-training step.
        alphas (torch.Tensor): The alphas used in the diffusion process. (num_diffusion_iters,)
        device (torch.device): The device to run the computation on.
        
    Returns:
        loss (torch.Tensor): The adjoint matching loss. 
    """
    num_diffusion_iters = 100
    loss = 0
    
    # alphas 1 -> 0
    # alphas = torch.flip(alphas, dims=[0])  
    alpha_cumprod = torch.flip(torch.cumprod(alphas, dim=0), dims=[0])
    
    for t in range(0, num_diffusion_iters): # 0 - 99
        # 1. compute alphas, betas
        next_t = t + 1
        
        alpha_prod_t = alpha_cumprod[t] 
        alpha_prod_t_next = alpha_cumprod[next_t] if next_t < num_diffusion_iters else torch.tensor(0.99999)   
        
        # compute coefficient for noise prediction difference
        noise_coeff = alpha_prod_t_next / (alpha_prod_t * (1 - alpha_prod_t_next))
        noise_coeff = noise_coeff * (1 - (alpha_prod_t / alpha_prod_t_next))
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
        lean_adjoint_coeff = (1 - alpha_prod_t_next) / (1 - alpha_prod_t)
        lean_adjoint_coeff = lean_adjoint_coeff * (1 - (alpha_prod_t / alpha_prod_t_next))
        lean_adjoint_coeff = lean_adjoint_coeff ** (0.5)
        
        # add loss term
        loss_term = (noise_coeff * noise_diff - lean_adjoint_coeff * lean_adjoint[:, t]) ** 2
        loss += loss_term.mean()
        
    return loss
        

def diffusion_step(noise_pred, timestep, sample, alphas):
    """Perform a single diffusion step to compute the predicted sample at the previous timestep. 
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
    
    Arguments:
        noise_pred (torch.Tensor): Output from the learned diffusion model. (num_traj, pred_horizon, action_dim)
        timestep (float): The current timestep in the diffusion chain. 
        sample (torch.Tensor): A current instance of a sample created by the diffusion process. (num_traj, pred_horizon, action_dim)
        alphas (torch.Tensor): The alphas used in the diffusion process. (num_diffusion_iters,)
        
    Returns: 
        pred_prev_sample (torch.Tensor): The predicted sample at the previous timestep. (num_traj, pred_horizon, action_dim)
    """
    # alphas go from small to large here 
    t = timestep 
    prev_t = t - 1

    # compute alphas, betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_prod_t = alpha_cumprod[t]
    alpha_prod_t_prev = alpha_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # compute predicted original sample from predicted noise
    pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # clip pred_original_sample
    pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

    # compute coefficients for pred_original_sample x_0 and current sample x_t
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # compute predicted previous sample µ_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    return pred_prev_sample

    
def solve_lean_adjoint(trajectories, obs_cond, alphas, reward_gradient, noise_pred_net, clip_adjoint, device):
    """Solve the learn adjoint ODE backwards from timestep k = K to 0. 
    Domingo-Enrich et al. (2024): https://arxiv.org/abs/2409.08861
    
    Arguments:
        trajectories (torch.Tensor): A batch of trajectories from the diffusion process. (num_traj, num_diffusion_iters + 1, pred_horizon, action_dim)
        obs_cond (torch.Tensor): A batch of observations to condition the diffusion process. (num_traj, obs_horizon * obs_dim)
        alphas (torch.Tensor): The alphas used in the diffusion process. (num_diffusion_iters,)
        reward_gradient (torch.Tensor): The gradient of the reward with respect to the actions. (num_traj, pred_horizon * action_dim)
        noise_pred_net (torch.nn.Module): The noise prediction network.
        clip_adjoint (bool): Whether to clip the lean adjoint samples for numerical stability.
        device (torch.device): The device to run the computation on.
        
    Returns:   
        lean_adjoint (torch.Tensor): The lean adjoint samples. (num_traj, num_diffusion_iters + 1, pred_horizon * action_dim)
    """    
    num_traj = trajectories.shape[0]
    num_diffusion_iters = trajectories.shape[1]
    
    # initialize lean_adjoint array which we will populate 
    lean_adjoint = torch.zeros((num_traj, num_diffusion_iters, reward_gradient.shape[-1]), device=device) 
    lean_adjoint[:, -1] = reward_gradient
    
    for k in range(num_diffusion_iters - 2, -1, -1): 
        
        # define function to help with gradient computation 
        def func(sample):
            noise_pred = noise_pred_net(
                sample=sample, # (num_traj, pred_horizon, action_dim)
                timestep=k,
                global_cond=obs_cond # (num_traj, obs_horizon * obs_dim)
            )
            
            return diffusion_step(noise_pred, k, sample, alphas) - sample 
            
        # compute gradient 
        noised_action_k = trajectories[:, k] # (num_traj, pred_horizon, action_dim)
        noised_action_k.requires_grad = True
        
        output = func(noised_action_k)
        output.sum().backward()
        
        action_gradient_k = noised_action_k.grad.flatten(start_dim=1) # (num_traj, pred_horizon * action_dim)
        
        # compute lean adjoint at time k 
        lean_adjoint_k = (lean_adjoint[:, k+1] * action_gradient_k).sum(dim=-1, keepdim=True)
        if clip_adjoint: 
            lean_adjoint_k = lean_adjoint_k.clamp(-1.0, 1.0)
        lean_adjoint[:, k] = lean_adjoint_k + lean_adjoint[:, k+1]
    
    return lean_adjoint