import numpy as np
import torch
from parameter import *
from model import Model
from custom_env import *
import ray

class Runner:
    def __init__(self,env_id) :
        self.env_id = env_id
        self.env =CustomCarRacing(render_mode="rgb_array")
        self.model = Model()
        
    def run(self):
        matrix = self.env.reset()
        hidden_state = None
        step = 0
        db_matrix = []
        db_vector = []
        db_policy = []
        db_value = []
        db_reward = []
        db_hidden_state = []
        db_done = []
        total_step = 0
        with torch.no_grad():
            while step < training_parameter.sync_windows and not self.env.done:
                db_matrix.append(matrix)
                policy,value,hidden_state = self.model.step(matrix,hidden_state)
                matrix, reward, done= self.env.step(policy)
                db_policy.append(policy)
                db_value.append(value)
                db_reward.append(reward)
                db_hidden_state.append(
                    [hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])
                db_done.append(done)
                if done:
                    matrix = self.env.reset()
                    hidden_state = None
                step += 1 
        db_matrix = np.array(db_matrix)     
        db_policy = np.concatenate(db_policy, axis=0)
        db_value = np.concatenate(db_value, axis=0)
        db_advs = np.zeros_like(db_reward)
        last_gaelam = 0
        total_step += step
        if self.env.done or total_step> training_parameter.max_step:
            matrix,_ = self.env.reset()

        for t in reversed(range(len(db_reward))):
            if t == len(db_reward) - 1:
                next_nonterminal = 1.0 - self.env.done
                next_values = 0
            else:
                next_nonterminal = 1.0- db_done[t + 1]
                next_values = db_value[t + 1]
            delta = db_reward[t] + training_parameter.gamma * next_values * next_nonterminal - db_value[t]
            db_advs[t] = last_gaelam = delta + training_parameter.gamma * training_parameter.lam * next_nonterminal * last_gaelam
        db_return = db_advs + db_value
        return  db_matrix,db_policy, db_value, db_advs, db_return, db_hidden_state

    def set_weights(self, weights):
        self.model.set_weights(weights)
        
@ray.remote(num_cpus = training_parameter.num_CPU, num_gpus = training_parameter.num_GPU)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)        
                
      