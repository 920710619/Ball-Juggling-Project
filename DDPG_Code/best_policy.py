import numpy as np
import torch
import time
import imageio

from Env import Env          # Environment of our project
from Buffer import Buffer    # Buffer of memory
from DDPG import DDPG        # DDPG algorithm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 20         # max episode iterations
MAX_STEPS = 1001           # max steps in one episode

S_DIM = 9                  # number of state dimensions
A_DIM = 3                  # number of action dimensions
A_MAX = [400, 500, 500]    # maximum amplitude for the actuation [-A_MAX, A_MAX]

Test = True            # if True, do not add noise to explore
Render = True          # if True, plot the environment when training

env = Env(Render)                                         # initialize environemnt
ram = Buffer()                                            # initialize memory buffer
trainer = DDPG(S_DIM, A_DIM, A_MAX, ram, device, Test)    # initialize DDPG part

trainer.load_best_models()
# trainer.load_cur_models()

new_observation = env.get_state()

writer = imageio.get_writer("simulation.gif", mode='I')

for _ep in range(MAX_EPISODES):

    have_collision_flag = False         # if ball have collided with plane or not
    now_collision_flag = False          # if ball is colliding or not
    collision_start_time = 0            # time step of ball colliding with plane 

    for r in range(MAX_STEPS):
        state = np.float32(new_observation)
        observation = new_observation

        if (not have_collision_flag) or now_collision_flag:
            action = trainer.get_action(state, Test = Test)        # Get action
            action = [action[i] * A_MAX[i]  for i in range(3)]
        else: 
            action = [0, 0, 0]

        new_observation = env.step(action)          # Simulate next time step
        new_state = np.float32(new_observation)

        if new_observation[2] < 0.1: 
            ball_pos = new_observation[:3]
            target_pos = new_observation[3:6]

            distance = (target_pos[0] - ball_pos[0]) ** 2 + (target_pos[1] - ball_pos[1]) ** 2
            reward = max(100 - distance * 5, -100)

            env.finish()
            break

        collision_info = env.detect_collision()     # Get collision information

        if collision_info != () and collision_info[0][4] == 5:     # collide with plane
            if not now_collision_flag:
                collision_start_time = r
                now_collision_flag = True

        if collision_info == () and now_collision_flag:
            have_collision_flag = True
            now_collision_flag = False

        time.sleep(0.02)

    print(' rew:', np.float32(reward), ' steps:', r,)


# np.savetxt('./Buffer/fix_step_nums.dat', fix_step_nums)

print('Completed episodes')

