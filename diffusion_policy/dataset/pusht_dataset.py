import torch
import numpy as np
from typing import Dict
import zarr

from diffusion_policy.dataset.normalization import get_data_stats, normalize_data


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def state_to_kps(states):
    """Convert state observations to keypoint observations. 

    Arguements:
        states (np.ndarray): A NumPy array of shape (N, 5), where each row contains:
                            - [agent_x, agent_y, block_x, block_y, block_angle].

    Returns:
        np.ndarray: A NumPy array of shape (N, 20), where each row contains:
                    [agent_x, agent_y, kp_x1, kp_y1, ..., kp_x9, kp_y9].
    """
    # state (N, 5)
    keypoints = states[:, :-1] # (N, 4)
    angles = states[:, -1] # (N,) 

    # construct rotation matrices
    sin = np.sin(angles) 
    cos = np.cos(angles) 
    rot_mat = np.stack([cos, sin, -sin, cos], axis=1).reshape(-1, 2, 2) # (N, 2, 2)

    # local vertices of keypoints 
    local_vertices = np.array([[-60.0, 0.0], [60.0, 0.0], 
                               [60.0, 30.0], [-60.0, 30.0], 
                               [-15.0, 30.0], [15.0, 30.0], 
                               [15.0, 120.0], [-15.0, 120.0]]) # (8, 2)

    # get global vertices 
    global_vertices = local_vertices @ rot_mat # (N, 8, 2)
    global_vertices += keypoints[:, 2:].reshape(-1, 1, 2) # (N, 8, 2)

    # concatenate keypoints 
    global_vertices = global_vertices.reshape(-1, 16) # (N, 16)
    keypoints = np.concatenate([keypoints, global_vertices], axis=1) # (N, 20)

    return keypoints.astype(np.float32)


class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': state_to_kps(dataset_root['data']['state'][:])
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample