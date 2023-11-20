import numpy as np
import torch

from Env import Env          # Environment of our project
from Buffer import Buffer    # Buffer of memory
from DDPG import DDPG        # DDPG algorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 501         # max episode iterations
MAX_STEPS = 1001           # max steps in one episode

S_DIM = 9                  # number of state dimensions
A_DIM = 3                  # number of action dimensions
A_MAX = [400, 500, 500]    # maximum amplitude for the actuation [-A_MAX, A_MAX]

torch.manual_seed(42)      # random seed to make sure reproducibility
np.random.seed(42)         # random seed to make sure reproducibility

print("-----------------------------------------------------------------------------")
print('State Dim:',S_DIM, 'Action Dim:',A_DIM,"Reforcement Learning Iteration Start.")
print("-----------------------------------------------------------------------------")

Test = False            # if True, do not add noise to explore
Render = False          # if True, plot the environment when training

env = Env(Render)                                         # initialize environemnt
ram = Buffer()                                            # initialize memory buffer
trainer = DDPG(S_DIM, A_DIM, A_MAX, ram, device, Test)    # initialize DDPG part

best_reward = 0         # used to record best reward, if a get a better reward, save the NN model
reward_nums = []        # the reward data of each episode, used to plot reward figure

for _ep in range(0, MAX_EPISODES):
    
    new_observation = env.get_state()   # state information

    have_collision_flag = False         # if ball have collided with plane or not
    now_collision_flag = False          # if ball is colliding or not
    collision_start_time = 0            # time step of ball colliding with plane 
    
    finish_reason = ""     # record the reason of end this episode

    states = []
    actions = []
    new_states = []

    for r in range(MAX_STEPS):
        state = np.float32(new_observation)
        observation = new_observation
        
        action = trainer.get_action(state, Test = Test)        # Get action
        action = [action[i] * A_MAX[i]  for i in range(3)]
        
        new_observation = env.step(action)          # Simulate next time step
        new_state = np.float32(new_observation)

        collision_info = env.detect_collision()     # Get collision information

        if collision_info != () and collision_info[0][4] == 5:     # collide with plane
            if not now_collision_flag:
                collision_start_time = r
                now_collision_flag = True

            if have_collision_flag == True:
                finish_reason = "Collide more than one time"
                reward = -100
                trainer.ram.add(state, action, reward, new_state)
                env.finish()
                break

        elif collision_info != () and collision_info[0][4] != 5:   # not collide with plane
            finish_reason = "Collide with other thing"
            reward = -100
            trainer.ram.add(state, action, reward, new_state)
            env.finish()
            break

        if collision_info == () and now_collision_flag:
            have_collision_flag = True
            now_collision_flag = False

        if now_collision_flag:
            if r - collision_start_time >= 10:
                finish_reason = "Stay in the plane too much time"
                reward = -100
                trainer.ram.add(state, action, reward, new_state)
                env.finish()
                break

        if new_observation[2] < 0.1:
            finish_reason = "Collide with floor"
            ball_pos = new_observation[:3]
            target_pos = new_observation[3:6]
            distance = (target_pos[0] - ball_pos[0]) ** 2 + (target_pos[1] - ball_pos[1]) ** 2
            reward = max(100 - distance * 5, -100)

            trainer.ram.add(state, action, reward, new_state)
            env.finish()
            break

        if (not have_collision_flag) or now_collision_flag:
            states.append(state)
            actions.append(action)
            new_states.append(new_state)

        # update critic and actor network each 5 steps
        if _ep >= 3 and r % 50 == 0:
            trainer.optimize(Test)

    rewards = [reward] * len(states)
    for i in range(len(states)):
        trainer.ram.add(states[i], actions[i], rewards[i], new_states[i])

    # Decrease the noise
    if trainer.noise_size > 0.05:
        trainer.noise_size /= 1.03
    else:
        trainer.noise_size = min((100 - reward) / 500, 0.05)

    print('EPISODE :-', _ep, ' rew:', np.float32(reward), " finish reason:", finish_reason
            ,' steps:', r, 'memory:', np.float32(trainer.ram.len/trainer.ram.maxSize*100),'% '
            ," noise_size:", trainer.noise_size )
    
    reward_nums.append(reward)

    if reward > best_reward and not Test:
        best_reward = reward
        trainer.save_best_models()
    if _ep % 50 ==0:
        trainer.save_cur_models()

np.savetxt('./Buffer/reward_nums.dat', reward_nums)    # save reward information

print("------------------")
print('Completed episodes')
print("------------------")
