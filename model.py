import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from parameter import *
from net import Net
import wandb
from util import print_once

class Model:
    def __init__(self,global_model = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        if global_model:
            self.optimizer = optim.Adam(self.net.parameters(), lr=training_parameter.lr,eps=training_parameter.opti_eps,weight_decay=training_parameter.weight_decay)
            self.net_scaler = GradScaler()          
    # set weights of the network 
    def set_weights(self, weights):
        self.net.load_state_dict(weights)
        
    def step(self,matrix,hidden_state):
        matrix = torch.from_numpy(matrix).to(self.device)
        policy,value,hidden_state = self.net(matrix,hidden_state)
        policy = np.squeeze(policy.cpu().detach().numpy())
        value = value.cpu().detach().numpy()
        return policy,value,hidden_state
    
    def train(self, matrix,returns, old_ps, old_value, hidden_state):
        matrix = torch.from_numpy(matrix).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        old_ps = torch.from_numpy(old_ps).to(self.device)
        old_value = torch.from_numpy(old_value).to(self.device)
        hidden_state = torch.from_numpy(hidden_state).to(self.device)
        advantage = returns - old_value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        states_list = []
        with autocast():
            for i in range(old_ps.shape[0]):
                new_ps, new_v,  _ = self.net(matrix[i], hidden_state[i])
                ratio = torch.exp(torch.log(torch.clamp(new_ps, 1e-6, 1.0)) - torch.log(torch.clamp(old_ps[i], 1e-6, 1.0)))
                entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))
                
                # critic loss
                new_v = torch.squeeze(new_v)
                new_v_clipped = old_value[i] + torch.clamp(new_v - old_value[i], - training_parameter.clip_range, training_parameter.clip_range)
                value_losses1 = torch.square(new_v- returns)
                value_losses2 = torch.square(new_v_clipped - returns)
                critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

                # actor loss
                ratio = torch.squeeze(ratio)
                policy_losses = advantage[i] * ratio
                policy_losses2 = advantage[i] * torch.clamp(ratio, 1.0 - training_parameter.clip_range, 1.0 + training_parameter.clip_range)
                policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

                # total loss
                all_loss = -policy_loss - entropy * training_parameter.entropy_coef + \
                    training_parameter.value_coef * critic_loss

                self.net_scaler.scale(all_loss).backward()
                self.net_scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), training_parameter.max_grad_norm)
                self.net_scaler.step(self.optimizer)
                self.net_scaler.update()
                states_list.append([policy_loss.cpu().detach().numpy(),\
                                    critic_loss.cpu().detach().numpy(),\
                                    all_loss.cpu().detach().numpy(),\
                                    new_ps.cpu().detach().numpy(),\
                                    new_v.cpu().detach().numpy(),\
                                    grad_norm.cpu().detach().numpy()])
        if training_parameter.wandb:
            wandb.log({"policy_loss":np.mean([x[0] for x in states_list]),\
                        "critic_loss":np.mean([x[1] for x in states_list]),\
                            "total_loss":np.mean([x[2] for x in states_list]),\
                            "value":np.mean([x[4] for x in states_list]),\
                            "grad_norm":np.mean([x[5] for x in states_list]),\
                            "rewards":np.mean(returns.cpu().detach().numpy())})         
        return states_list