import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import pybullet as p
from gym_pybullet_drones.utils.specs import BoundedArray

class LandingAviary(BaseSingleAgentAviary):
    """Single agent RL problem: take-off."""
    
    ################################################################################
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([0,0,10]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW,
                 freq: int= 120,
                 aggregate_phy_steps: int=5,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.RGB,
                 act: ActionType=ActionType.VEL,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        #define vision and velocity as control inputs
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         )
    
    def video_camera(self):
        nth_drone = 0
        gv_pos = np.array(self._get_vehicle_position()[0])
        #### Set target point, camera view and projection matrices #
        target = gv_pos#np.dot(rot_mat, np.array([0, 0, -1000])) + np.array(self.pos[nth_drone, :])

        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, 0.5]) +np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[1, 0, 0],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0,
                                                      physicsClientId=self.CLIENT
                                                      )
        #SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        SEG_FLAG = True
        [w, h, rgb, dep, seg] = p.getCameraImage(width=128,
                                                 height=128,
                                                 shadow=0,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        return rgb
    
    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """
        vz_max = -0.5
        gv_pos = np.array(self._get_vehicle_position()[0])
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[0:3]
        drone_v = drone_state[10:13]
        error_xy = np.linalg.norm(drone_pos[0:2]-gv_pos[0:2])
        error_z = np.linalg.norm(drone_pos[2]-gv_pos[2])

        flag_land = drone_pos[2] >= 0.05 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()
        flag_crash = drone_pos[2] < 0.05 and p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()

        error_z = error_z + 4

        theta_l = 20
        theta = np.rad2deg(np.arctan2(abs(error_xy), error_z))
        r = (error_xy ** 2 + error_z ** 2) ** 0.5

        reward_r = 2 - r / 5
        reward_theta = 4 / (1 + np.e ** ((theta - theta_l) / 3)) + 2 / (1 + np.e**((theta-6)/2)) - 2

        if drone_v[2]>0:
            reward_vz = -1
        elif drone_v[2]>-0.3:
            reward_vz = -0.3
        elif drone_v[2]>-0.6:
            reward_vz = 1
        elif drone_v[2]>-0.8:
            reward_vz = -0.8
        elif np.linalg.norm(drone_v)/self.SPEED_LIMIT[2] > 1.2:
            reward_vz = -0.8
        else:
            reward_vz = -1.5


        reward = reward_r + reward_theta + reward_vz


        if flag_crash:
            print('crashed!')
            reward = -300

        if flag_land:
            print('landed!')
            reward = 5000 * np.e ** (-abs(drone_v[2] + 0.5))

        return reward

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ():
            return True
        
        return self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC

    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        drone_state = self._getDroneStateVector(0)
        drone_position = drone_state[0:3]
        UGV_pos = np.array(self._get_vehicle_position()[0])
        x_pos_error = np.linalg.norm(drone_position[0]-UGV_pos[0])
        y_pos_error = np.linalg.norm(drone_position[1]-UGV_pos[1])

        
        #episode end returns true when landing ends up on the ground i.e. the episode should truely finish
        episode_end = p.getContactPoints(bodyA=1, physicsClientId=self.CLIENT) != ()

        Landing_flag = drone_position[2] >= 0.275 and episode_end
        
        return {"landing": Landing_flag,
                "episode end flag": episode_end,
                "x error": x_pos_error,
                "y error": y_pos_error,
                } #### Calculated by the Deep Thought supercomputer in 7.5M years

    def _resetPosition(self):
        PYB_CLIENT = self.getPyBulletClient()
        DRONE_IDS = self.getDroneIds()
        p.resetBasePositionAndOrientation(DRONE_IDS[0],
                                  [0, 0, 1],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  physicsClientId= PYB_CLIENT
                                  )
        p.resetBaseVelocity(DRONE_IDS[0],
                    [0, 0, 0],
                    physicsClientId= PYB_CLIENT
                    )

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)
        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in TakeoffAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
    # for cross compatbility with dm gym
