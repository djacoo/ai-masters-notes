# RL-Laboratory🤖

Code for the Reinforcement Learning lab of *Reinforcement Learning and Advanced programming for AI* course, MSc degree in Artificial Intelligence 2024/2025 at the University of Verona.

## First Set-Up (Conda)
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your System.

2.  Install Miniconda
	- On Linux/Mac 
		- Use *./Miniconda3-latest-Linux-{version}.sh* to install.
		- *sudo apt-get install git* (may be required).
	- On Windows
		- Double click the installer to launch.
		- *NB: Ensure to install "Anaconda Prompt" and use it for the other steps.*

3.  Set-Up conda environment:
	- *git clone https://github.com/Isla-lab/RL-lab*
	- *conda env create -f RL-Lab/tools/rl-lab-environment.yml*

## First Set-Up (Python Virtual Environments)
Python virtual environments users (venv) can avoid the Miniconda installation. The following package should be installed:
  - scipy, numpy, gym
  - jupyter, matplotlib, tqdm
  - tensorflow, keras

## Spinning Up Set-Up (Conda)
1. Ensure you already have Miniconda installed from the previous lessons

2. Set-up a new and separate conda environment for Spinning Up:
	- *conda create -n spinningup python=3.6*
	- *conda activate spinningup*
	- *sudo apt-get update && sudo apt-get install libopenmpi-dev*

3. Finally, install the Spinning Up dependencies:
	- navigate to RL-lab/spinningup/spinningup
	- *pip install opencv-python==4.1.2.30*
	- *pip install -e .*
	
## Spinning Up Usage
1. Remember to activate your miniconda environment: *conda activate spinningup*

2. To train a RL agent, run the *train.py* script located inside the *spinningup* folder using the following arguments:
	- *env*: the environment to train the RL agent on (required)
	- *algo*: the RL algorithm to be used during training (required)
	- *exp_name*: the name of the experiment, necessary to save the results and the agent weights (required)
	- *hid*: a list representing the neural network hidden sizes (default is [32, 32])
	- *epochs*: the number of training epochs (default is 50)

An example of usage to train VPG over the CartPole environment may be: *python train.py --env CartPole-v1 --algo vpg --exp_name first_experiment*.
Once the training is complete, a graph showing the performance will be visualized.

3. To test a RL agent, run the *test.py* script located inside the *spinningup* folder using the following arguments:
	- *exp_name*: the name of a past training experiment to be tested (required)

4. The available RL algorithms are: vpg, ddpg, ppo, sac (note that ddpg, ppo, and sac are to be completed as part of the lessons!)

5. The available environments are: CartPole-v1, LunarLander-v2, BipedalWalker-v3, Pendulum-v0, Acrobot-v1, MountainCar-v0, MountainCarContinuous-v0, FrozenLake-v0

An example of usage to test a previous training experiment may be: *python test.py --exp_name first_experiment*.

## Assignments
Following the link to the code snippets for the lessons:

**First Semester**
- [x] Lesson 1: MDP and Gym Environments [Slides](slides/slides_lesson_1.pdf), [Code](lessons/lesson_1_code.py), [Results](results/lesson_1_results.txt)
- [x] Lesson 2: Multi-Armed Bandit [Slides](slides/slides_lesson_2.pdf), [Code](lessons/lesson_2_code.py), [Results](results/lesson_2_results.txt)
- [x] Lesson 3: Monte Carlo RL methods [Slides](slides/slides_lesson_3.pdf), [Code](lessons/lesson_3_code.py), [Results](results/lesson_3_results.txt)
- [x] Lesson 4: Temporal difference methods [Slides](slides/slides_lesson_4.pdf), [Code](lessons/lesson_4_code.py), [Results](results/lesson_4_results.txt)
- [x] Lesson 5: Dyna-Q [Slides](slides/slides_lesson_5.pdf), [Code](lessons/lesson_5_code.py), [Results](results/lesson_5_results.txt)

**Second Semester**
- [x] Lesson 6: Tensorflow-PyTorch and Deep Neural Networks [Slides](slides/slides_lesson_6.pdf), [Code](lessons/lesson_6_code.py), [Results](results/lesson_6_results.txt)
- [x] Lesson 7: Deep Q-Network [Slides](slides/slides_lesson_7.pdf), [Code](lessons/lesson_7_code.py), [Results](results/lesson_7_results.txt)
- [x] Lesson 8: REINFORCE [Slides](slides/slides_lesson_8.pdf), [Code](lessons/lesson_8_code.py), [Results](results/lesson_8_result.png) 
- [x] Lesson 9: A2C [Slides](slides/slides_lesson_9.pdf), [Code](lessons/lesson_9_code.py), [Results](results/lesson_9_result.png)
- [x] Lesson 10: PPO [Slides](slides/slides_lesson_10.pdf), [Code](spinningup), [Results](results/lesson_10_spinningup-ppo.png)
- [ ] Lesson 11: DDPG [Slides](slides/slides_lesson_11.pdf), [Code](spinningup), [Results](results/lesson_11_spinningup-ddpg_MountainCar_v0.png)
- [ ] Lesson 12: SAC [Slides](slides/slides_lesson_12.pdf), [Code](spinningup), [Results](results/lesson_12_spinningup-sac_MountainCar_v0.png)

<!---  - [ ] Lesson 11: DRL in Practice [Code!](lessons/lesson_11_code.py) [Results 1!](results/lesson_11_result.png) [Results 2!](results/lesson_11_results_TB3.png) [Slides!](slides/slides_lesson_11.pdf) --->

**Extra exercise**
- [x] Lesson Extra: Value/Policy Iteration [Slides](slides/slides_lesson_extra.pdf), [Code](lessons/lesson_extra_code.py), [Results](results/lesson_extra_results.txt)

## Tutorials
This repo includes a set of introductory tutorials to help accomplish the exercises. In detail, we provide the following Jupyter notebook that contains the basic instructions for the lab:
- **Tutorial 1 - Gym Environment:** [Here!](tutorials/tutorial_environment.ipynb)
- **Tutorial 2 - TensorFlow (Keras):** [Here!](tutorials/tutorial_tensorflow.ipynb)
- **Tutorial 3 - PyTorch:** [Here!](tutorials/tutorial_pytorch.ipynb)


## Contact information
*  Teaching assistant: **Gabriele Roncolato** - gabriele.roncolato@univr.it
*  Professor: **Alberto Castellini** - alberto.castellini@univr.it
