# robotap

an unofficial implementation of robotap

We use the robosuite environment due to my limit desktop gpu. If you find some environments that don't need lots of computing resources, don't hesitate to tell me.

## Installation

First, you should prepare for the robosuite environment:
```
conda create -n tapnet python=3.11
pip install mujoco
pip install robosuite
```
Then, prepare for tapnet methods according to [the official repository](https://github.com/google-deepmind/tapnet):
```
git clone https://github.com/deepmind/tapnet.git
cd tapnet
pip install .
```
Then, download the pre-trained model:
```
mkdir checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```
Now, you've had the basic environment to run the code.

## Usage
The core contribution is the control part of robotap(the jacobi matrix to compute the dpos of the end effector) in `control.py`. So, here are steps to use my repository:
attention: you should select the task, the path of some documents carefully.
>First, use `collect_human_demonstration.py`（in robosuite package）to collect some eye-in-hand human demonstrations.

>Second, use tapir method or bootstap method from [the official repository](https://github.com/google-deepmind/tapnet) to track many points in these human demonstrations.

>Third, use `active_points.py`to extract active points from these tracks for the specific task.

>Fourth, use  `control.py` to track a hand-select human demonstration carefully.

