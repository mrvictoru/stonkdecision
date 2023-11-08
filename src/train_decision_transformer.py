import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
# get the start time to calculate training time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets.load import load_dataset
from torch.utils.data import Dataset, DataLoader

from cust_transf import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utility function to compute the discounted cumulative sum of a vector
def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

# tensor version of discount_cumsum
def discount_cumsum_torch(x, gamma):
    disc_cumsum = torch.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

def compute_rtg(data, gamma, rtg_scale):
    rtg = []
    for reward in data['reward'].apply(lambda x: discount_cumsum(np.array(x, dtype=np.float32), gamma)/rtg_scale):
        rtg.append(reward)
    return rtg

def compute_rtg_torch(reward, gamma, rtg_scale):
    rtgs = torch.zeros_like(reward)
    for i,reward in enumerate(torch.unbind(reward, dim=0)):
        rtg = discount_cumsum_torch(reward, gamma)/rtg_scale
        rtgs[i,:,:] = rtg
    return rtgs

# define a custom dataset class which loads the data, modifies the reward to be the discounted cumulative sum and apply trajectory masking
class CustomTrajDataset(Dataset):
    def __init__(self, file_name, context_len, gamma, rtg_scale, device):
        self.gamma = gamma
        self.context_len = context_len
        self.device = device
        self.data = load_dataset("json", data_files = file_name, field = 'data')['train']

        # check if the data is homogeneous
        self.homogeneous = True
        try:
            check = np.array(self.data['state'])

        except Exception as e:
            self.homogeneous = False
            print("Data might not homogeneous, see error below.")
            print(e)

        if self.homogeneous:
            print("Processing data as torch tensor.")
            self.data = self.data.with_format('torch')

            # calculate mean and std of states
            states = self.data['state'].to(self.device)
            states = states.reshape((states.shape[0]*states.shape[1], states.shape[2]))
            self.state_mean, self.state_std = torch.mean(states, dim=0), torch.std(states, dim=0) + 1e-6

            # calculate rtg
            self.rtg = compute_rtg_torch(self.data['reward'].to(self.device), gamma, rtg_scale)
            self.stateshape = self.data['state'].shape[0]

            # move the data to the device
            self.state = self.data['state'].to(self.device)
            self.action = self.data['action'].to(self.device)

        
        else:
            print("Processing data as pandas dataframe.")
            self.data = self.data.with_format('pandas')
            #min_len = 10**6

            # calculate mean and std of states
            states = []

            for traj in self.data['state']:
                states.append(traj)
                #if len(traj) < min_len:
                #    min_len = len(traj)
            states = np.concatenate(states, axis=0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

            # calculate rtg
            self.rtg = pd.Series(compute_rtg(self.data, gamma, rtg_scale))

            # get the len of the dataset
            self.stateshape = len(self.data['state'])
            
        print("Dataset length: ", self.stateshape)


    def get_state_stats(self):
        return self.state_mean, self.state_std        

    def __len__(self):
        return self.stateshape

    def __getitem__(self, idx):

        # check if the data is homogeneous
    
        try:
            if self.homogeneous:
                state = self.state[idx]
                action = self.action[idx]
                rtg = self.rtg[idx]

            else:
                state = np.stack(self.data['state'][idx])
                action = np.stack(self.data['action'][idx])
                rtg = self.rtg[idx]

        except IndexError:
            # handle index out of range error
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.data['state'])}")


        data_len = state.shape[0]
        
        if data_len > self.context_len:
            # sample random start index
            start_idx = np.random.randint(0, data_len - self.context_len)
            
            if self.homogeneous:
                state = state[start_idx:start_idx+self.context_len]
                action = action[start_idx:start_idx+self.context_len]
                rtg = rtg[start_idx:start_idx+self.context_len]            
            else:

                # slice the data and convert to torch
                state = torch.from_numpy(state[start_idx:start_idx+self.context_len])
                action = torch.from_numpy(action[start_idx:start_idx+self.context_len])
                rtg = torch.from_numpy(rtg[start_idx:start_idx+self.context_len])
            timesteps = torch.arange(start=start_idx, end=start_idx + self.context_len, step=1)
            # trajectory mask
            mask = torch.ones(self.context_len, dtype=torch.long)
        else:
            padding_len = self.context_len - data_len
            if not self.homogeneous:
                # convert to torch
                state = torch.from_numpy(state)
                action = torch.from_numpy(action)
                rtg = torch.from_numpy(rtg)

            # pad the data with zeros
            state = torch.cat([state, torch.zeros((padding_len, *state.shape[1:]))], dim=0)
            action = torch.cat([action, torch.zeros((padding_len, *action.shape[1:]))], dim=0)
            rtg = torch.cat([rtg, torch.zeros((padding_len, *rtg.shape[1:]))], dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            # trajectory mask
            mask = torch.cat([torch.ones(data_len, dtype=torch.long), torch.zeros(padding_len, dtype=torch.long)], dim=0)

        """
        Exepected output type and shape
        state type:  torch.float32
        actions type:  torch.float32
        rtg type:  torch.float32
        timestep type:  torch.int64
        traj_mask type:  torch.int64
        shape state: torch.Size([32, 30, 13])
        shape rtg: torch.Size([32, 30, 1])
        shape timestep: torch.Size([32, 30])
        shape actions: torch.Size([32, 30, 2])
        """
        if self.homogeneous:
            return state.float(), action.float(), rtg, timesteps, mask
        else:
            return state.float(), action.float(), rtg.unsqueeze(-1), timesteps, mask

# create a combine dataset object from json files under a directory
def get_combinedataset(path, context_len = 60, gamma = 0.8, rtg_scale = 100, device = "cpu"):
    
    # read all json files under the directory
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))
    
    # create dataset objects from json files
    datasets = []
    for file in files:
        dataset = CustomTrajDataset(file, context_len, gamma, rtg_scale, device)
        print(f"{file} has {len(dataset)} trajectories")
        datasets.append(dataset)
    
    # combine the datasets
    combinedataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Combined dataset has {len(combinedataset)} trajectories")
    return combinedataset

def init_training_object(dataset, batch_size = 32, shuffle=True, lr = 1e-4, wt_decay = 1e-4, eps = 1e-6, warmup_steps = 1e5, growth_interval = 150 , context_len = 60, hp_scale = 2, drop_p = 0.2, device = "cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    # define model parameters
    # sample 1 batch from dataloader to get the shape of the data
    state, action, rtg, timesteps, mask = next(iter(dataloader))
    # use batch shape to determine the model parameters
    state_dim = state.shape[-1]
    act_dim = action.shape[-1]

    n_blocks = int(4*hp_scale)
    h_dim = int(96*hp_scale)
    n_heads = int(8*hp_scale)

    # create model
    model = DecisionTransformer(state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p).to(device)
    model.float()

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay, eps = eps)

    # create scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))

    # create a GradScaler object for mixed precision training
    scaler = torch.cuda.amp.GradScaler(growth_interval=growth_interval)

    # get the model parameters size
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of parameters: {n_params}")

    model_params = {
     'state_dim': state_dim,
     'act_dim': act_dim,
     'n_blocks': n_blocks,
     'h_dim': h_dim,
     'context_len': context_len,
     'n_heads': n_heads,
     'drop_p': drop_p,
    }

    return dataloader, model, optimizer, scheduler, scaler, model_params

# custom training function which take in the model, dataset, optimizer, scheduler, scaler, n_epochs, min_scale
def train_model(model, dataloader, optimizer, scheduler, scaler, n_epochs = 80, min_scale = 128):

    # record the start time
    start_time = datetime.datetime.now()

    # define training parameters
    log_action_losses = []

    # train model
    for epoch in range(n_epochs):
        model.train()

        for state, actions, rtg, timestep, traj_mask in tqdm(dataloader):
            # get batch data to device
            state = state.to(device)
            actions = actions.to(device)
            rtg = rtg.to(device).float()
            timestep = timestep.to(device)
            traj_mask = traj_mask.to(device)

            action_targets = torch.clone(actions).detach().to(device)

            # Zeroes out the gradients
            optimizer.zero_grad()

            # run forward pass with autocasting
            # disable autocasting for now to avoid mixed precision caused loss NaNs
            with torch.cuda.amp.autocast(enabled=False):
                _, _, act_preds = model.forward(state, rtg, timestep, actions)

                # consider only the action that are not padded
                act_preds = act_preds.view(-1, act_dim)[traj_mask.view(-1) > 0]
                action_targets = action_targets.view(-1, act_dim)[traj_mask.view(-1) > 0]

                # calculate losses just for actions
                loss = F.mse_loss(act_preds, action_targets, reduction='mean')

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # unscale the gradients
            scaler.unscale_(optimizer)
            # Clips the gradients by norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the learning rate according to the scheduler
            scheduler.step()
            # Updates the scale for next iteration.
            scaler.update()
            # enforce min scale to avoid mixed precision caused NaNs
            if scaler.get_scale() < min_scale:
                scaler._scale = torch.tensor(min_scale).to(scaler._scale)
        
            # append action loss to log
            log_action_losses.append(loss.detach().cpu().item())

        # print every 10 loss log
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f'Epoch {epoch}: Loss: {log_action_losses[-1]}')

    # record the end time
    end_time = datetime.datetime.now()
    print(f'Training time: {end_time - start_time}')
    
    return model, log_action_losses

def full_training_run(path = 'stock_trade_data'):
    comb_dset = get_combinedataset(path = path, device = device)
    dataloader, model, optimizer, scheduler, scaler, model_params= init_training_object(comb_dset, device = device)
    trained_model, log_action_loss = train_model(model, dataloader, optimizer, scheduler, scaler)

    plt.plot(log_action_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Log Action Loss')
    plt.yscale('log')
    plt.title('Log Action Loss vs Epoch')
    plt.show()

    return trained_model, model_params

def save_model(model, params, model_name, directory = 'trained_models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), os.path.join(directory, model_name+'.pt'))

    # write model parameters to a json file
    with open(os.path.join(directory, model_name+'_params.json'), 'w') as f:
        json.dump(params, f)