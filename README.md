# Ball-Juggling-Project

The aim of this project is to employ a robotic arm to hit a small ball towards a target location. Therefore, the initial phase of our project involves the creation of a URDF model that will be utilized to read and write the environment in “PyBullet” (a Python-based physical simulation library).

## Robot Arm Structure

The construction of the robot arm is a crucial element of the modelling process, as the arm's structure and degree of freedom design directly impacts the dimensionality of the action in subsequent reinforcement learning tasks.

<img src="Resurt-Presentation\Robot_Arm.png" width="400">

As depicted in Figure, the robotic arm comprises of three degrees of freedom (DOFs). DOF1 corresponds to the rotation around the axis at the base, while DOF2 and DOF3 govern the swinging motion of the two arms.

## DDPG parameters

<img src="Resurt-Presentation\Parameters.png" width="800">
We are utilizing the DDPG algorithm, and the specific parameters are shown in the table.

## Optimizaiton

Despite implementing the DDPG algorithm as described above, our reinforcement learning agent was found to perform poorly in the ball juggling task and often failed to converge. We believe that the reason for this poor performance is that the ball juggling task, as a specific problem, possesses certain unique characteristics that make it is difficult to achieve 
good results by simply using a generic reinforcement learning algorithm.
In order to address the unique features and challenges of the ball juggling task, we have made adaptations to several aspects of our reinforcement learning code

### Replay Buffer Filter Condition

In our previous design, we provide the reward based on the ball's landing position, assigning a large positive reward if it landed close to the target point, and a large negative reward otherwise. Additionally, we stored the state, action, next state, and reward for each time step in the replay buffer, using them to update the agent's reinforcement learning strategy.
However, this approach had a flaw: if the ball received a large positive reward upon landing, the agent would assume that its behavior during that time step was excellent, even though the robot arm's actions at that point had no impact on the ball's landing position. Since we associated the reward for landing the ball with the action taken at the same time, it affected the agent's learning and updating process.
To address this issue, we added a filter condition to the replay buffer experience, such that only experiences leading up to the slapping of the ball were added to the buffer. We bound the experience of the ball's landing position and corresponding reward to the final time step. This helps the agent learn more effectively and avoid being misled by rewards associated with meaningless behavior following the slapping of the ball.

### Reward Optimization

During our experiments, we observed that our agent would keep the action at a low level sometime, resulting in the platform above the robot arm remaining motionless and the ball remaining on the platform for extended periods of time. 
We consider that this behavior was due to the agent not learning to keep the ball as still as possible, which would result in the agent receiving a relatively large positive reward. This tendency to delay receiving negative rewards often caused the agent to become trapped in local optima, which hindered its ability to effectively learn optimal behavior.
We have also observed situations where the robotic arm taps the ball to the target point using a part other than the small platform above the arm (such as the arm's rod). 
To prevent such issues from leading to the agent becoming trapped in a local optimum or failing to converge, we have made adjustments to the REWARD function. Specifically, if the ball remains on the platform for an extended period of time or if it collides with any part of the arm outside the small platform on top, we assign a large negative reward (-100) to the agent and terminate the current episode.

## Result

<img src="Resurt-Presentation\DDPG_reward_figure.png" width="500">

To better illustrate the effectiveness of the DDPG algorithm, we plotted the reward figure for DDPG (Episode = 500). As shown in Figure, In the first 200 episodes, the reward exhibited significant fluctuations, while the overall trend showed a steady increase. This indicates that, Because of the presence of noise, the agent continued to explore and gradually learn better strategies. After 200 episodes, the reward stabilized around 100, indicating that the reinforcement learning algorithm had converged, and the results were highly effective.

<img src="Resurt-Presentation\BallVideo.gif" width="900">


