# Reward-based Fine-tuning of Diffusion Policies for Improved Robot Learning

This repository implements a novel reward-based fine-tuning method for [diffusion policies](https://arxiv.org/abs/2303.04137), a popular choice for policy in robot learning settings. Diffusion policies are typically trained in a supervised manner on expert demonstrations, e.g. Behavior Cloning. However, this method enables diffusion policy optimization using synthetic data. The approach adapts the [Adjoint Matching](https://arxiv.org/abs/2409.08861) algorithm, a reward-based fine-tuning method originally designed for text-to-image diffusion models, to the robot learning setting. 

The method is evaluated on the Push-T task from [Chi et al. (2024)](https://arxiv.org/abs/2303.04137). The code for the Push-T task and diffusion policy pre-training was adapted from their [repository](https://github.com/real-stanford/diffusion_policy).

## Setup

### Environment
To use the repository, install the conda environment on a Linux machine with Nvidia GPU. 

```console
conda create -n robodiff python=3.9
conda activate robodiff
pip install -r requirements.txt
```

### Download Push-T Dataset 
Under the repo root, create data subdirectory:
```console
mkdir data
```
Download the dataset from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/) and extract it:
```console
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -O ./data/pusht.zip
unzip ./data/pusht.zip -d ./data  && rm -f ./data/pusht.zip
```

### Weights & Biases
Optionally, you can login to [wandb](https://wandb.ai/home) to track results:
```console
wandb login
```

## Usage

If you haven't already, activate the environment:
```console
conda activate robodiff
```

For any further details about parameters of the following scripts, see the documentation.

### Diffusion Policy Pre-training
Pre-train the diffusion policy on expert demonstrations from the Push-T task:
```console
python pretrain_policy.py --dataset_path ./data/pusht/pusht_cchi_v7_replay.zarr
```
The pre-tained model will be saved to `./data/checkpoints/policy_pretrained.ckpt`.

### Collect Data for Reward Model Training

Collect data for reward model training but rolling out the pre-trained policy on `num_eps` unseen environment instances: 
```console
python collect_rewards.py --num_eps 300 --model_path ./data/checkpoints/policy_pretrained.ckpt
```
The collected data will be saved to `./data/reward/action_reward_data_300eps.npy`.

### Reward Model Training

Train the reward model on the collected data: 
```console
python train_reward.py \
    --num_epochs 300 \
    --lr 1e-2 \
    --batch_size 256 \
    --action_reward_path ./data/reward/action_reward_data_300eps.npy
```
The trained reward model will be saved to `./data/checkpoints/reward_model_rd0.ckpt`.

### Reward Fine-tuning the Diffusion Policy via Adjoint Matching
Fine-tune the pre-trained diffusion policy using the Adjoint Matching algorithm: 
```console
python reward_finetuning.py \
    --num_ft_iters 200 \
    --num_eps 3 \
    --warmup_steps 10 \
    --lr 1e-4 \
    --clip_adjoint \
    --noise_pred_model_path ./data/checkpoints/policy_pretrained.ckpt \
    --reward_model_path ./data/checkpoints/reward_model_rd0.ckpt \
    --action_reward_path ./data/reward/action_reward_data_300eps.npy
```
The fine-tuned model will be saved to `./data/checkpoints/policy_finetuned_rd0.ckpt`.

### Evaluation
Evaluate the diffusion policy on `num_eps` unseen environment instances:
```console
python evaluate_policy.py \
    --num_eps 50 \
    --model_path ./data/checkpoints/policy_finetuned_rd0.ckpt \
    --save_vids
```
The `save_vids` parameter will save videos of the policy evaluation. The videos are saved to `./data/media/vis*.mp4`.

## References
* [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
* [Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://arxiv.org/abs/2409.08861)