"""
pick_place_env.py
-----------------
Gymnasium environment for the pick-and-place task, NOW WITH A GRIPPER.

State  : [joint_angles(7), joint_velocities(7), ee_pos(3), obj_pos(3), target_pos(3), gripper_state(1)] = 24-dim
Action : [joint_targets(7), gripper_command(1)], all clipped to [-1, 1]
Reward : -dist(EE->obj) - dist(obj->tray)
         +10  on successful grasp (gripper activated while close to obj)
         +50  on success (obj within 0.10 m of tray centre)
Done   : success OR max_steps exceeded
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import random
from gymnasium import spaces

GRASP_THRESHOLD  = 0.05   # m — EE must be this close to object to engage suction/grasp
PLACE_THRESHOLD  = 0.10   # m — object must be this close to tray centre to count as placed
MAX_STEPS        = 500

class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._client = None

        # 8 actions: 7 joint targets + 1 gripper command, all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 24-dim state vector (added 1 dim for gripper state)
        obs_low  = np.full(24, -np.inf, dtype=np.float32)
        obs_high = np.full(24,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._step_count = 0
        self._grasp_given = False
        
        # Gripper tracking
        self._grasp_constraint = None
        self._gripper_active = False

    # ------------------------------------------------------------------
    def _connect(self):
        if self._client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.3, 0.0, 0.2],
                physicsClientId=self._client,
            )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)

        p.loadURDF("plane.urdf", physicsClientId=self._client)
        self._robot = p.loadURDF(
            "kuka_iiwa/model.urdf", basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self._client
        )
        self._num_joints = p.getNumJoints(self._robot, physicsClientId=self._client)

        # Randomize cube position
        x = random.uniform(0.4, 0.7)
        y = random.uniform(-0.3, 0.3)
        self._object_id = p.loadURDF(
            "cube_small.urdf", basePosition=[x, y, 0.02],
            physicsClientId=self._client
        )
        self._container = p.loadURDF(
            "tray/traybox.urdf", basePosition=[0.3, 0.4, 0.0],
            physicsClientId=self._client
        )

        self._step_count  = 0
        self._grasp_given = False
        self._grasp_constraint = None
        self._gripper_active = False

        # Warm-up so the scene settles
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        # Action slicing: first 7 are arm joints, 8th is gripper command
        arm_actions = action[:7]
        gripper_cmd = action[7]

        # 1. Apply joint position targets to the arm
        for j in range(self._num_joints):
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(arm_actions[j]),
                force=200,
                physicsClientId=self._client,
            )

        # 2. Step the simulation physics
        p.stepSimulation(physicsClientId=self._client)
        self._step_count += 1

        # 3. Handle Gripper Logic (Suction/Magnetic style)
        link_state = p.getLinkState(self._robot, self._num_joints - 1, physicsClientId=self._client)
        ee_pos = np.array(link_state[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self._object_id, physicsClientId=self._client)
        obj_pos = np.array(obj_pos)
        
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)

        # Turn Gripper ON
        if gripper_cmd > 0.0 and self._grasp_constraint is None:
            if dist_ee_obj < GRASP_THRESHOLD:
                # Lock the object to the end-effector using a fixed constraint
                self._grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=self._robot,
                    parentLinkIndex=self._num_joints - 1,
                    childBodyUniqueId=self._object_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                    physicsClientId=self._client
                )
                self._gripper_active = True

        # Turn Gripper OFF
        elif gripper_cmd <= 0.0 and self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint, physicsClientId=self._client)
            self._grasp_constraint = None
            self._gripper_active = False

        # 4. Gather final observations and rewards
        obs = self._get_obs()
        ee_pos_final   = obs[14:17]
        obj_pos_final  = obs[17:20]
        tray_pos_final = obs[20:23]

        reward, terminated = self._compute_reward(ee_pos_final, obj_pos_final, tray_pos_final)
        truncated = self._step_count >= MAX_STEPS

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        js = [p.getJointState(self._robot, i, physicsClientId=self._client)
              for i in range(self._num_joints)]
        joint_angles     = [s[0] for s in js]
        joint_velocities = [s[1] for s in js]

        link_state = p.getLinkState(self._robot, self._num_joints - 1,
                                    physicsClientId=self._client)
        ee_pos = list(link_state[0])

        obj_pos, _    = p.getBasePositionAndOrientation(
            self._object_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(
            self._container, physicsClientId=self._client)

        # Add the gripper state flag (1.0 for holding, -1.0 for empty)
        gripper_state = [1.0 if self._gripper_active else -1.0]

        obs = (joint_angles + joint_velocities
               + ee_pos + list(obj_pos) + list(target_pos) + gripper_state)
        
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self, ee_pos, obj_pos, tray_pos):
        dist_ee_obj   = np.linalg.norm(ee_pos   - obj_pos)
        dist_obj_tray = np.linalg.norm(obj_pos  - tray_pos)

        reward = -dist_ee_obj - dist_obj_tray

        # Grasp bonus (one-time) — no height check needed anymore!
        if not self._grasp_given and self._gripper_active:
            reward += 10.0
            self._grasp_given = True

        # Success
        terminated = False
        if dist_obj_tray < PLACE_THRESHOLD:
            reward     += 50.0
            terminated  = True

        return reward, terminated

    # ------------------------------------------------------------------
    def render(self):
        pass

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None