from mimetypes import init

import numpy as np
import random

class SimpleController:

    tripod_gait = [	0.15, 0, 0.05, 0.5, 0.5, # leg 1
				0.15, 0, 0.05, 0.0, 0.5, # leg 2
				0.15, 0, 0.05, 0.5, 0.5, # leg 3
				0.15, 0, 0.05, 0.0, 0.5, # leg 4
				0.15, 0, 0.05, 0.5, 0.5, # leg 5
				0.15, 0, 0.05, 0.0, 0.5] # leg 6

    

    def __init__(self, params=[], body_height=0.15, velocity=0.46, crab_angle=-1.57, period=1.0, dt=1/240) -> None:
        self.l_1 = 0.05317
        self.l_2 = 0.10188
        self.l_3 = 0.14735

        self.body_height = body_height
        self.velocity = velocity
        self.crab_angle = crab_angle
        self.period = period
        self.duration = 5

        self.array_dim = int(np.around(period / dt))

        self.positions = np.empty((0, self.array_dim))
        self.velocities = np.empty((0, self.array_dim))

        self.angles = np.empty((0, self.array_dim))
        self.speeds = np.empty((0, self.array_dim))

        params = np.array(params).reshape(6, 5)
    
    def IMU_feedback(self, measured_attitude):
        return

    def joint_angles(self, t): # 0-1119
        # prev_angles = mysupg.jointangels(time-1) # super fancy 'feedback' bro
        angles = []
        for i in range(3*18):
            angles.append(random.random()*1.745)
        return np.array(angles)
    
    def joint_speeds(self, t):
        k = int(((t % self.period) / self.period) * self.array_dim)
        return self.speeds[:, k]
