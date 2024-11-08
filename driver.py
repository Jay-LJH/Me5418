import numpy as np
import torch
import ray
from model import Model
import os
from runner import RLRunner
import datetime
import wandb
from util import print_once
from parameter import *

def main():
    if training_parameter.wandb:
        wandb.util.generate_id()
        wandb.init(project="me5418")
    global_model =  Model(global_model=True)
    envs = [RLRunner.remote(i + 1) for i in range(training_parameter.num_envs)]
    total_steps = 0
    curr_steps = curr_episodes = map_update=last_model_t = 0
    update_done = True
    job_list = []
    while total_steps < training_parameter.max_step:
        if update_done:
                map_update += 1
                for i, env in enumerate(envs):
                    job_list.append(env.run.remote())
        done_id, job_list = ray.wait(job_list, num_returns=training_parameter.num_envs)
        update_done = True if job_list == [] else False
        done_len = len(done_id)
        job_results = ray.get(done_id)
        data_buffer = {"matrix":[],  "policy": [],"values": [],"advantages":[], "returns": [],"hidden_state":[]}
        for results in range(done_len):
            for i, key in enumerate(data_buffer.keys()):
                data_buffer[key].append(job_results[results][i])
            for key in data_buffer.keys():
                data_buffer[key] = np.concatenate(data_buffer[key], axis=0)
        # training of reinforcement learning
        temp_step = data_buffer["matrix"].shape[0]
        db_loss = []
        inds = np.arange(temp_step)
        for _ in range(training_parameter.num_epochs):
            np.random.shuffle(inds)
            for start in range(0, temp_step, training_parameter.batch_size):
                end = start + training_parameter.batch_size
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in
                            (data_buffer["matrix"],data_buffer["returns"],data_buffer["policy"],data_buffer["values"],
                            data_buffer["hidden_state"]))
                db_loss.append(global_model.train(*slices))      
        data_buffer=None
        net_weights = global_model.net.state_dict()
        net_weights_id = ray.put(net_weights)
        weight_job=[]
        for i, env in enumerate(envs):
            weight_job.append(env.set_weights.remote(net_weights_id))
        ray.get(weight_job)
        curr_steps += temp_step
        
        # save model
        if (curr_steps - last_model_t) >= training_parameter.save_interval:
            last_model_t = curr_steps
            model_path = "model/" +str(datetime.datetime.now().date())+"/" +str(curr_steps)
            os.makedirs(model_path)
            path_checkpoint = model_path + "/map_net_checkpoint.pkl"
            net_checkpoint = {"model": global_model.net.state_dict(),
                                "optimizer": global_model.optimizer.state_dict()}
            torch.save(net_checkpoint, path_checkpoint)

if __name__ == "__main__":
    main()