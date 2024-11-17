'''
import numpy as np
import torch
from constants import constants
import ray
from model import Model
import os

def main():

    # Create the control model for the agent.
    global_model = Model(global_model = True)

    # Create environment(s), allow synchronous training for A3C.
    envs = [RL_runner.remote(i + 1) for i in range(constants.NUM_ENVS)]

    # Training loop, multiple Actor-Critic asynchronously, update gradients, save model.
    episodes_done = 0
    job_list = []
    while episodes_done < constants.MAX_EPISODES:
        if update_done:
            map_update += 1
            for i, env in enumerate(envs):
                job_list.append(env.run.remote())

        # Utilize Ray package to enable and parallize data collection from RL.
        done_id, job_list = ray.wait(job_list, num_returns = constants.NUM_ENVS)
        update_done = True if job_list == [] else False
        done_len = len(done_id)
        job_results = ray.get(done_id)
        data_buffer = {"matrix":[], "returns":[], "values":[], "action":[], "ps":[], "hidden_state":[]}
        for results in range(done_len):
            for i, key in enumerate(data_buffer.keys()):
                data_buffer[key].append(job_results[results][i])

            for key in data_buffer.keys():
                data_buffer[key] = np.concatenate(data_buffer[key], axis=0)

        # training of reinforcement learning
        temp_step = data_buffer["obs"].shape[0]
        db_loss = []
        inds = np.arange(temp_step)
        for _ in range(training_parameter.num_epochs):
            np.random.shuffle(inds)
            for start in range(0, temp_step, training_parameter.batch_size):
                end = start + training_parameter.batch_size
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in
                            (data_buffer["obs"],data_buffer["vector"],data_buffer["returns"],
                            data_buffer["values"],data_buffer["action"],data_buffer["ps"],data_buffer["hidden_state"]))
                db_loss.append(global_model.train(*slices))

        data_buffer = None
        net_weights = global_model.network.state_dict()
        net_weights_id = ray.put(net_weights)
        weight_job=[]
        for i, env in enumerate(envs):
            weight_job.append(env.set_map_weights.remote(net_weights_id))
        ray.get(weight_job)
        # save model
        if (curr_steps - last_model_t) / training_parameter.save_interval >= 1.0:
            last_model_t = curr_steps
            model_path = os.join("model/", "model_" + str(curr_episodes))
            os.makedirs(model_path)
            path_checkpoint = model_path + "/map_net_checkpoint.pkl"
            net_checkpoint = {"model": model.network.state_dict(),
                                "optimizer": model.net_optimizer.state_dict(),
                                "map_update": map_update,
                                "step": curr_steps,
                                "episode": curr_episodes}
            torch.save(net_checkpoint, path_checkpoint)

if __name__ == "__main__":
    main()
'''
