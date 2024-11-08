# how to run

## create environment and activate it

conda env create -f environment.yml

## human control

python custom_env.py

use Arrow keys to control the car

## model training

python driver.py

you can custom the env_num, cpu_num and gpu_num at parameter.py

if you don't want to use wandb to record training parameter, pls mark it to False

## model usage

python custom <path-to-model-file>

# file explaination

## custom_env.py

define our custom environment, including box and gym environment.
there is two types of render mode, include [human,rgb-array]

## driver.py

main process of training,create multiply ray process

## runner.py

subprocess for environment interact and experience collection

## model.py

define model.py that convert observation in tensor and do forward & backward propagation

## net.py

define the structure of the neural network.
