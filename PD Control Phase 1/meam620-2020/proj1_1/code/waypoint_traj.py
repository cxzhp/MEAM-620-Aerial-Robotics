import numpy as np


class WaypointTraj(object):
    """

    """

    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """
        self.const_vel = 2 # 2  work is 1.6

        self.all_points = points
        # print(self.all_points[-1])

        self.time = [0.0]

        self.distance = []
        for pt in range(len(self.all_points) - 1):
            self.distance.append(np.linalg.norm(self.all_points[pt + 1] - self.all_points[pt]))

        for t in range(len(self.distance)):
            self.time.append(self.time[t] + self.distance[0] / self.const_vel)

        # print('time', self.time)
        #
        # print('distance', self.distance)

    def update(self, t):
        # """
        # Given the present time, return the desired flat output and derivatives.
        #
        # Inputs
        #     t, time, s
        # Outputs
        #     flat_output, a dict describing the present desired flat outputs with keys
        #         x,        position, m
        #         x_dot,    velocity, m/s
        #         x_ddot,   acceleration, m/s**2
        #         x_dddot,  jerk, m/s**3
        #         x_ddddot, snap, m/s**4
        #         yaw,      yaw angle, rad
        #         yaw_dot,  yaw rate, rad/s
        # """

        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HEre
        if t == 0:
            x = self.all_points[0]
            x_dot = np.zeros((3,))

        elif t > self.time[-1] :

            x = self.all_points[-1]
            x_dot = np.zeros((3,))
            x_ddot = np.zeros((3,))
            # print('im here')

        else:
            for i in range(len(self.time) - 1):
                if t < self.time[i+1]:

                    x = self.all_points[i]
                    # print('what is:', x)
                    x_dot = ((self.const_vel / self.distance[i]) * (self.all_points[i + 1] - self.all_points[i]) )/3.5   #/3.5
                    x_ddot = np.zeros((3,))
                    break






        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}

        return flat_output
