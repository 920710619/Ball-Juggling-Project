import pybullet as p
import pybullet_data
import time
import numpy as np

class Env():
    def __init__(self, Render) -> None:

        # Visualize or not
        if Render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10.0)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.target_point = p.loadURDF("./URDF/target_point.urdf", [3, 4, 0.01])
        self.init_env()

    def get_state(self):
        # return the state
        joint_state = [p.getJointState(self.jugglerId, 0)[0], p.getJointState(self.jugglerId, 1)[0], 
                p.getJointState(self.jugglerId, 3)[0]]

        ball_pos, ball_vel = p.getBasePositionAndOrientation(self.ballId)[0], p.getBaseVelocity(self.ballId)[0]

        target_pos = p.getBasePositionAndOrientation(self.target_point)[0]

        state = list(ball_pos) + list(target_pos) + joint_state
        # time.sleep(0.001)

        return state

    def step(self, action):

        # Force control of the joint
        p.setJointMotorControl2(self.jugglerId,0, p.TORQUE_CONTROL, force=action[0]) 
        p.setJointMotorControl2(self.jugglerId,1, p.TORQUE_CONTROL, force=action[1]) 
        p.setJointMotorControl2(self.jugglerId,3, p.TORQUE_CONTROL, force=action[2])

        # Step simulation
        p.stepSimulation()

        state = self.get_state()
        # time.sleep(0.002)

        return state

    def finish(self):
        # Restart the simulation
        p.removeBody(self.ballId)
        p.removeBody(self.jugglerId)
        self.init_env()

    def init_env(self):
        # Initialize the juggler and ball
        self.jugglerId = p.loadURDF("./URDF/juggler2.urdf", [0, 0, 0.1])
        self.ballId = p.loadURDF("./URDF/ball.urdf", [0, 0, 1.35])

        p.createConstraint(self.jugglerId, -1 , self.planeId, -1, 1, [1,0,0], [0,0,0], [0,0,0])
        p.setCollisionFilterPair(self.jugglerId, self.ballId, 12, 0, enableCollision=True)

    def detect_collision(self):
        # Detect the Collision
        contacts = p.getContactPoints(self.ballId, self.jugglerId)

        return contacts
