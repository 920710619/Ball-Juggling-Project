# Ball-Juggling-Project

The aim of this project is to employ a robotic arm to hit a small ball towards a target location. Therefore, the initial phase of our project involves the creation of a URDF model that will be utilized to read and write the environment in “PyBullet” (a Python-based physical simulation library).
The construction of the robot arm is a crucial element of the modelling process, as the arm's structure and degree of freedom design directly impacts the dimensionality of the action in subsequent reinforcement learning tasks.
![Robot_Arm](https://github.com/920710619/Ball-Juggling-Project/assets/67464174/c3a21a9a-9f4d-4d95-b6db-a247e21041af)

As depicted in Figure, the robotic arm comprises of three degrees of freedom (DOFs). DOF1 corresponds to the rotation around the axis at the base, while DOF2 and DOF3 govern the swinging motion of the two arms.


![DDPG_reward_figure](https://github.com/920710619/Ball-Juggling-Project/assets/67464174/bcc1c29d-189b-4bf0-a816-33661c4841b6)
To better illustrate the effectiveness of the DDPG algorithm, we plotted the reward figure for DDPG (Episode = 500).
As shown in Figure, In the first 200 episodes, the reward exhibited significant fluctuations, while the overall trend showed a steady increase. This indicates that, Because of the presence of noise, the agent continued to explore and gradually learn better strategies. After 200 episodes, the reward stabilized around 100, indicating that the reinforcement learning algorithm had converged, and the results were highly effective.


![BallVideo](https://github.com/920710619/Ball-Juggling-Project/assets/67464174/e80fbad5-3355-48ec-bdb3-a8144d4ee0f2)

