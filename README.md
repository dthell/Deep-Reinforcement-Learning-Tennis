[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, the work will be with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

3. Then create a new environment with Python. For a detailed view on installing this environment the other dependancies, please refer to the dependencies section of the DRLND Git Hub: [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)

4. Dependencies

The code uses the following dependencies:
- Unity ML-Agents (https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
- Numpy (http://www.numpy.org/ , numpy version 1.19.1)
- collections from which deque is imported
- PyTorch (torch version 0.4.0)
- random
- matplotlib.pyplot (matplotlib version 3.3.2)
- os and logging for creating folders and managing the log file
- maddpg.py from which Main Agent is imported (controlling the various collaborative agents)
- ddpg.py
- OUNoise.py to add noise to the initial exploration phase of the learning algorithm
- utilities.py to transpose shapes, create from or to tensors, apply soft_update to the target algorithms
- networkforall.py include the network architectures of the actor and critic networks

The versions are for indication only as the one used to develop the code but the code may run with other versions.


### Instructions

Follow the instructions in `Tennis.ipynb` to test a trained agent or launch a new training.  

There are 6 sections:
1. Starting the environment to import the required dependancies and start the Unity environment
2. The second section is dedicated to getting some knowledge about the state and action spaces
3. The 3rd section is testing random actions to see how to interact with the environment
4. The 4th section implements the training loop with specific hyperparameters
5. The 5th section shows the training scores graph
6. The last part tests a trained agent and see how it interacts better with the environment. At the end the environment is closed

Some of agent parameters (noise scale, learning update frequency, number of update loops at each learning step, max length of an episode or batch size) can be updated at the begining of section 4 to test other parameters. For the hyperparameters not exposed in the function, the network architecture hyperparameters are at the begining of the maddpg.py file, the learning rates in the ddpg.py file and the OU noise parameters (aside from the scale) are in the OUNoise.py file.

The trained weights used in section 6 are only valid for the hyperparameters initially given in the Tennis Jupyter notebook
