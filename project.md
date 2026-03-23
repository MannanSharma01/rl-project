Step 1 — Basic PyBullet Environment

Goal: launch PyBullet and load a robot.

Install:

pip install pybullet

Minimal environment:

import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")

robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

for _ in range(10000):
    p.stepSimulation()
    time.sleep(1/240)

Goal of this step:

Robot loads

simulation runs

gravity works

Step 2 — Add Object + Container

You now replicate the pick-and-place environment.

object_id = p.loadURDF("cube_small.urdf", [0.6,0,0.02])

container = p.loadURDF("tray/traybox.urdf", [0.3,0.4,0])

Now your environment has:

robot arm

object

target container

This matches the problem description in your slides.

Step 3 — Randomize Object Position

Important for RL.

import random

x = random.uniform(0.4,0.7)
y = random.uniform(-0.3,0.3)

p.resetBasePositionAndOrientation(
    object_id,
    [x,y,0.02],
    [0,0,0,1]
)

This implements:

randomized initial conditions

which your proposal explicitly states.

Step 4 — Extract the State Vector

Your state is defined as:

St = [joint angles,
      joint velocities,
      end effector position,
      object position,
      target position]

Example code:

num_joints = p.getNumJoints(robot)

joint_states = p.getJointStates(robot, range(num_joints))

joint_angles = [s[0] for s in joint_states]
joint_velocities = [s[1] for s in joint_states]

link_state = p.getLinkState(robot, num_joints-1)
ee_position = link_state[0]

obj_pos, _ = p.getBasePositionAndOrientation(object_id)
target_pos, _ = p.getBasePositionAndOrientation(container)

state = joint_angles + joint_velocities + list(ee_position) + list(obj_pos) + list(target_pos)

Now you have:

𝑠
𝑡
∈
𝑅
𝑛
s
t
	​

∈R
n

which matches your RL formulation.

Step 5 — Implement Action Commands

Initially you don't need RL.

Just test control.

Example:

p.setJointMotorControl2(
    robot,
    jointIndex=2,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0.5
)

or torque control.

This corresponds to:

𝑎
𝑡
∈
𝑅
𝑚
a
t
	​

∈R
m
2. Reward Function Prototype

Start simple.

Example reward:

reward =
- distance(EE, object)
+ grasp_bonus
+ distance(object, container)
+ success_reward

Example:

import numpy as np

dist1 = np.linalg.norm(np.array(ee_position) - np.array(obj_pos))
dist2 = np.linalg.norm(np.array(obj_pos) - np.array(target_pos))

reward = -dist1 - dist2

Later you add:

+10 for successful grasp
+50 for successful placement
