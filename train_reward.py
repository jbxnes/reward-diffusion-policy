# Usage: python train_reward.py --num_epochs 100 --lr 1e-4 --batch_size 256 --reward_diff --obs_actions_path ./data/reward/obs_action_data_300inits.npy --rewards_path ./data/reward/reward_data_300inits.npy --seed 42
import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
import argparse

from diffusers.optimization import get_scheduler
from diffusion_policy.dataset.reward_dataset import RewardDataset
from diffusion_policy.model.reward import RewardModel


def train(num_epochs, lr, batch_size, reward_diff, obs_actions_path, rewards_path, seed):
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda')
    
    # load data
    train_dataset = RewardDataset(
        obs_actions_path=obs_actions_path, 
        rewards_path=rewards_path, 
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
        obs_actions_path=obs_actions_path, 
        rewards_path=rewards_path, 
        train=False,
        reward_diff=reward_diff)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True)
    
    # initialize model 
    reward_model = RewardModel(reward_diff=reward_diff)
    reward_model.to(device)
    
    # init optimizer
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
                    
                    optimizer.zero_grad()
                    pred_rewards = reward_model(obs_actions)
                    loss = nn.functional.mse_loss(pred_rewards, rewards, reduction='sum')
                    loss.backward()
                    
                    optimizer.step()
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
                        pred_rewards = reward_model(obs_actions)
                        
                        loss = nn.functional.mse_loss(pred_rewards, rewards, reduction='sum')
                        
                        loss_cpu = loss.item()
                        test_loss += loss_cpu
                        n_samples += rewards.shape[0]
                        tepoch.set_postfix(loss=loss_cpu / rewards.shape[0])
                    
            # print results 
            avg_test_loss = test_loss / n_samples
            print(f"\nTEST LOSS | EPOCH {epoch_idx}: {avg_test_loss}\n")
            
            # log results
            wandb.log({'train_loss': avg_train_loss,
                       'test_loss': avg_test_loss,
                       'epoch': epoch_idx,
                       'inital_lr': lr,
                       'reward_difference': int(reward_diff)})
            
    # save model 
    torch.save(reward_model.state_dict(), f'./data/checkpoints/reward_model_rd{int(reward_diff)}_seed{seed}.ckpt')
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-22, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--reward_diff', action='store_true')
    parser.add_argument('--obs_actions_path', default='./data/reward/obs_action_data_300inits.npy', type=str)
    parser.add_argument('--rewards_path', default='./data/reward/reward_data_300inits.npy', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parsed_args = parser.parse_args()
    
    import wandb
    wandb.init(project="reward_model_training", config=vars(parsed_args)) 
    
    train(num_epochs=parsed_args.num_epochs,
          lr=parsed_args.lr,
          batch_size=parsed_args.batch_size,
          reward_diff=parsed_args.reward_diff,
          obs_actions_path=parsed_args.obs_actions_path,
          rewards_path=parsed_args.rewards_path,
          seed=parsed_args.seed)