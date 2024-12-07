import os
import torch
import torch.nn as nn
import numpy as np 
import argparse
from tqdm import tqdm
import wandb

from diffusers.optimization import get_scheduler
from diffusion_policy.reward_model import RewardModel
from diffusion_policy.dataset.reward_dataset import RewardDataset


def train(num_epochs, lr, batch_size, reward_diff, action_reward_path, seed, log_wandb=False):
    """Train the reward model.
    
    Given an observation and sequence of pred_horizon actions, the model is trained to predict
    the reward after exectuting action_horizon actions. 
    
    Arguments:
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        reward_diff (bool): Whether to use reward difference as the target.
        action_reward_path (str): Path to the file containing reward data.
        seed (int): Random seed for reproducibility.
        log_wandb (bool): Whether to log training results to Weights & Biases.
    """
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # device
    device = torch.device('cuda')
    
    # load data
    train_dataset = RewardDataset(
        action_reward_path=action_reward_path,
        train=True,
        reward_diff=reward_diff)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    
    test_dataset = RewardDataset(
        action_reward_path=action_reward_path,
        train=False,
        reward_diff=reward_diff)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    
    # get reward model 
    reward_model = RewardModel(reward_diff=reward_diff)
    reward_model.to(device)
    
    # get optimizer
    optimizer = torch.optim.Adam(
        params=reward_model.parameters(),
        lr=lr)

    # get LR scheduler 
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    
    # training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            
            # train epoch
            reward_model.train()
            train_loss = 0
            n_samples = 0
            with tqdm(train_dataloader, desc='Train', leave=False) as tepoch:
                for obs_actions, rewards in tepoch:
                    obs_actions = obs_actions.to(device)
                    rewards = rewards.to(device)
                    
                    # predicted rewards
                    pred_rewards = reward_model(obs_actions)
                    
                    # L2 loss
                    loss = nn.functional.mse_loss(pred_rewards, rewards, reduction='sum')
                    
                    # optimize 
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    
                    loss_cpu = loss.item()
                    train_loss += loss_cpu
                    n_samples += rewards.shape[0]
                    tepoch.set_postfix(loss=loss_cpu / rewards.shape[0])
            
            # print results
            avg_train_loss = train_loss / n_samples
            print(f"\nTRAIN LOSS | EPOCH {epoch_idx}: {avg_train_loss}\n")
            tglobal.set_postfix({'train_loss': avg_train_loss})
            
            # test epoch
            reward_model.eval()
            test_loss = 0
            n_samples = 0
            with tqdm(test_dataloader, desc='Test', leave=False) as tepoch:
                for obs_actions, rewards in tepoch:
                    obs_actions = obs_actions.to(device)
                    rewards = rewards.to(device)
                    
                    with torch.no_grad():
                        # predicted rewards
                        pred_rewards = reward_model(obs_actions)
                        
                        # L2 loss 
                        loss = nn.functional.mse_loss(pred_rewards, rewards, reduction='sum')
                        
                        loss_cpu = loss.item()
                        test_loss += loss_cpu
                        n_samples += rewards.shape[0]
                        tepoch.set_postfix(loss=loss_cpu / rewards.shape[0])
                    
            # print results 
            avg_test_loss = test_loss / n_samples
            print(f"\nTEST LOSS | EPOCH {epoch_idx}: {avg_test_loss}\n")
            
            # log results
            if log_wandb:
                wandb.log({'train_loss': avg_train_loss,
                        'test_loss': avg_test_loss,
                        'epoch': epoch_idx,
                        'inital_lr': lr,
                        'reward_difference': int(reward_diff)})
            
    # save model 
    if not os.path.exists('./data/checkpoints'):
        os.makedirs('./data/checkpoints')
        
    torch.save(reward_model.state_dict(), f'./data/checkpoints/reward_model_rd{int(reward_diff)}_seed{seed}.ckpt')
    print(f"MODEL SAVED TO ./data/checkpoints/reward_model_rd{int(reward_diff)}_seed{seed}.ckpt")
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--reward_diff', action='store_true')
    parser.add_argument('--action_reward_path', default='./data/reward/action_reward_data_300eps.npy', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    
    wandb.init(project="reward_model_training", config=vars(parsed_args)) 
    
    train(num_epochs=parsed_args.num_epochs,
          lr=parsed_args.lr,
          batch_size=parsed_args.batch_size,
          reward_diff=parsed_args.reward_diff,
          action_reward_path=parsed_args.action_reward_path,
          seed=parsed_args.seed,
          log_wandb=True)