import numpy as np
import torch
from constants import *
from training_parameter import training_parameter
from model import Model
from cargo_moving_truck_env import cargo_moving_truck_env
import ray

class Runner:
    def __init__(self,env_id) :
        self.env_id = env_id
        self.env = cargo_moving_truck_env(render_mode="rgb_array")
        self.model = Model()
        
    def run(self):
        observation = self.env.reset()
        hidden_state = None
        step = 0
        data_buffer = []
        db_state = []
        db_matrix = []
        db_vector = []
        db_policy = []
        db_value = []
        db_reward = []
        db_hidden_state = []
        db_done = []
        with torch.no_grad():
            while step < training_parameter.sync_windows and not self.env.done:
                db_state.append(observation['state'])
                db_matrix.append(observation['box'])
                db_vector.append(observation['vector'])
                policy,value,hidden_state = self.model.step(observation['state'],observation['box'],self['vector'],hidden_state)
                observation, reward, done, _ = self.env.step(policy)
                db_policy.append(policy)
                db_value.append(value)
                db_reward.append(reward)
                db_hidden_state.append(hidden_state)
                db_done.append(done)
                step += 1
                
        db_state = np.concatenate(db_state, axis=0)
        db_matrix = np.concatenate(db_matrix, axis=0)
        db_vector = np.concatenate(db_vector, axis=0)
        db_policy = np.concatenate(db_policy, axis=0)
        db_value = np.concatenate(db_value, axis=0)
        db_reward = np.concatenate(db_reward, axis=0)
        db_advs = np.zeros_like(db_reward)
        last_gaelam = 0
        last_values = np.zeros((db_value.shape[1]), dtype=np.float32)

        for t in reversed(range(len(db_reward))):
            if t == len(db_reward) - 1:
                next_nonterminal = 1.0 - self.map_done
                next_values = last_values
            else:
                next_nonterminal = 1.0- db_done[t + 1]
                next_values = db_value[t + 1]
            delta = db_reward[t] + training_parameter.gamma * next_values * next_nonterminal - db_value[t]
            db_advs[t] = last_gaelam = delta + training_parameter.gamma * training_parameter.lam * next_nonterminal * last_gaelam
        db.return_ = db_advs + db_value
        return db_state, db_matrix, db_vector, db_policy, db_value, db_advs, db_return, db_hidden_state


@ray.remote(num_cpus = 1, num_gpus = 0)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)        
                
      