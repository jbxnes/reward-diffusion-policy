import os
import torch
import torch.nn as nn 
import numpy as np 
import argparse
from tqdm import tqdm
import wandb

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.pusht_dataset import PushTDataset


def train(dataset_path, seed, log_wandb=False):
    """Pre-train the noise prediction model. The model is trained to replicate the behavior 
    demonstrated in expert demonstrations using behaviour cloning. 
    
    The policy resulting from the noise prediction model is designed to predict a sequence 
    of prediction_horizon actions, given a sequence of obs_horizon observations. 
    
    The observations used are 9D keypoints obtained from the ground truth pose of the T block, 
    with proprioception for end effector location. 
    
    Arguements:
        dataset_path (str): Path to the file containing expert demonstration data.
        seed (int): Random seed for reproducibility. 
        log_wandb (bool): Whether to log training results to Weights & Biases.   
    """
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # define variables 
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 20
    action_dim = 2
    
    num_epochs = 100
    num_diffusion_iters = 100
    
    device = torch.device('cuda')
    
    # get dataset
    dataset = PushTDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon)
    
    stats = dataset.stats
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    
    # get model 
    noise_pred_net = ConditionalUnet1D(input_dim=action_dim, 
                                       global_cond_dim=obs_dim * obs_horizon)
    noise_pred_net.to(device)
    
    # get noise scheduler 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # get optimizer
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # get LR scheduler 
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    
    # training 
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = 0
            n_samples = 0
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    obs_cond = nobs[:,:obs_horizon,:] # (B, obs_horizon, obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    )

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration (forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise, reduction='sum')

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss += loss_cpu
                    n_samples += B 
                    tepoch.set_postfix(loss=loss_cpu / B)
            
            # log results 
            epoch_avg_loss = epoch_loss / n_samples
            if log_wandb
                wandb.log({'train_loss': epoch_avg_loss,
                        'epoch': epoch_idx})
            tglobal.set_postfix(loss=epoch_avg_loss)

    # save_model
    if not os.path.exists('./data/checkpoints'):
        os.makedirs('./data/checkpoints')
    
    torch.save(noise_pred_net.state_dict(), f'./data/checkpoints/policy_pretrained_seed{seed}.ckpt')
    print(f"MODEL SAVED TO ./data/checkpoints/policy_pretrained_seed{seed}.ckpt")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_path', default='./data/pusht/pusht_cchi_v7_replay.zarr', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    wandb.init(project="diffusion_policy_pretraining", config=vars(parsed_args)) 
    
    # experiment inputs
    train(dataset_path=parsed_args.dataset_path,
          seed=parsed_args.seed,
          log_wandb=True)
    