# How to run

This time, since we are doing machine learning and developing on different platforms, we partially shift to docker images, though we still have conda configrations.
There is a debate among us betwween different versions of environtments, because of different typing mechanisms used between different Python versions.
All the docker image configuration files are found in 'docker configurations' folder. Inside we have configurations for gpu (CUDA) and cpu.
Required ports and tokens to connect to Jupyter server also can be found in the respective 'compose.yml' files.
To run Jupyter on the associated configurations, copy to root folder of this project 'CargoMovingTruck' and run 'docker compose up' in terminal.
We also have a previous version of this project inside 'insufficiently organized' but that is just kept for backward checking reasons. We have reorganized the code a lot to enhance easy reading.

# Individual file explanations

## docker configurations (not tested yet, we do not have time to synchronize everything within our group)

As mentioned above, all the configuration files required to build relevent Jupyter servers.

## Still using Conda environment control and activation (do this for now)

conda env create -f environment.yml

## constants.py

All the parameters that configurates how objects work, modes in which the environment runs, how the environment is rendered, and how rewards are defined.
Specifically, 'FULLY_OBSERVABLE' controls how states are returned. 'True' for easier direct metadata information of all the objects (including those outside render screen), 'False' for harder, render screen state so that the agent has to explore objects outside renderer and recognize all the objects.

## cargo_moving_truck_env.py

The environment definition following OpenAI Gymnasium specifications.

## cargo.py, truck.py, destination.py

The objects/entities related to our problem.

## utils.py, headless_renderer.py

The pygame polygon drawer and 'matplotlib' renderer on headless server IDEs such as Jupyter.

## logger.py, main.py

Deprecated for now. We might recover interactive mode in the future.

## unit_tests.ipynb

Test whether our gymnasium environment is well-defined. Not an exhaustive test yet.
