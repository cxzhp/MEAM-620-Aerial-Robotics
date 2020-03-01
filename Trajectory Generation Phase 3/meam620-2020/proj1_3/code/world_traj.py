
import numpy as np

from proj1_3.code.graph_search import graph_search


class WorldTraj(object):

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """
        self.resolution = np.array([0.125, 0.125, 0.125])  # resolution, the discretization of the occupancy grid in x,y,z[0.25, 0.25, 0.25]
        self.margin = 0.3
        # margin, the inflation radius used to create the configuration space (assuming a spherical drone)
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        # ------------------------ Reduce Points-----------------------------------
        # Reference:
        # https://stackoverflow.com/questions/26820714/python-find-vector-direction-between-two-3d-points?rq=1
        # I want to determine if two consecutive points have same direction
        self.points = np.zeros((1, 3))
        self.temp = np.zeros((1, 3))
        diff = np.diff(self.path, axis=0)  # find difference between two consecutive points
        norm = np.sum(diff ** 2, 1)
        norm = np.asarray(norm).reshape((len(norm), 1))
        direction = diff / norm  # Normalized Direction

        for i in range(len(direction) - 1):
            if not np.array_equal(direction[i + 1], direction[i]):
                self.temp = np.append(self.temp, [self.path[i]],
                                      axis=0)  # OnlyPoints with different direction are needed
        self.temp = self.temp[1:, :]

        for i in range(self.temp.shape[0] - 1):
            if np.linalg.norm(self.temp[i + 1] - self.temp[i]) > 0.25:  # remove points which are close to each other, 0.4
                self.points = np.append(self.points, [self.temp[i]], axis=0)
        self.points = np.append(self.points, [self.path[-1]], axis=0)
        self.points = self.points[1:, :]  # no need (0,0,0)
        # print(self.points)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        self.velocity = 0.6  # -------------------------------------------------============================
        self.time_interval = []

        self.length_interval = []  # list
        for i in range(self.points.shape[0] - 1):
            self.length_interval.append(np.linalg.norm(self.points[i + 1] - self.points[i]))

        print('points length: ', self.points.shape[0])
        print('length interval length: ', len(self.length_interval))
        self.time_interval = np.asarray(self.length_interval) / self.velocity  # ndarray
        self.time_interval = np.insert(self.time_interval, 0, 0)
        print(self.time_interval)
        self.total_time = np.sum(self.time_interval)

        self.accumulated_time = [0, self.time_interval[1]]

        for i in range(1, self.time_interval.shape[0] - 1):
            self.accumulated_time.append(self.accumulated_time[i] + self.time_interval[i+1])


        # Minimum Jerk Spline on Slides 13 Page 5
        unknowns = 6 *  (self.points.shape[0] - 1 )  # 6 * m unknowns (m is number of segments )
        A_Matrix = np.zeros((unknowns,unknowns))  # A need to be saqure for inverting
        b = np.zeros((unknowns,3))  # right hand side of Ax = b
        A_Matrix[0:3,0:6] = np.array([[0,0,0,0,0,1],
                                      [0,0,0,0,1,0],
                                      [0,0,0,2,0,0]])
        # Boundary Conditions at first point and last point # Page 5 of Slides 13
        A_Matrix[-3:,-6:] = np.array([[self.time_interval[-1] ** 5, self.time_interval[-1] ** 4,self.time_interval[-1] ** 3, self.time_interval[-1] ** 2,self.time_interval[-1], 1],
                                    [5 * self.time_interval[-1] ** 4, 4 * self.time_interval[-1] ** 3,3 * self.time_interval[-1] ** 2, 2 * self.time_interval[-1], 1, 0],
                                    [20 * self.time_interval[-1] ** 3, 12 * self.time_interval[-1] ** 2, 6 * self.time_interval[-1], 2, 0, 0]])

        b[0:3,:] = np.array([self.points[0], [0,0,0],[0,0,0]])   # Page 9: p1(0) = y0 p1dot(0) = ydot0  p1ddot(0) = yddot0
        b[-3:,:] = np.array([self.points[-1],[0,0,0],[0,0,0]])  # Page 11: p1(t) = y0 p1dot(t) = ydot0  p1ddot(t) = yddot0


        for i in range(1, self.points.shape[0] - 1):
            A_Matrix[-3 + 6 * i:3+6*i, -6+6*i:6 + 6*i] = np.array([
                                          [self.time_interval[i] ** 5,self.time_interval[i] ** 4, self.time_interval[i] ** 3,self.time_interval[i] ** 2,self.time_interval[i],1,0,0,0,0,0,0],
                                          [0,0,0,0,0,0,0,0,0,0,0,1],
                                          [5 * self.time_interval[i] ** 4, 4 * self.time_interval[i] ** 3, 3 * self.time_interval[i] ** 2, 2 * self.time_interval[i], 1, 0,0, 0, 0, 0, -1, 0],
                                          [20 * self.time_interval[i] ** 3, 12 * self.time_interval[i] ** 2, 6 * self.time_interval[i], 2, 0, 0, 0, 0, 0, -2, 0, 0],
                                          [60 * self.time_interval[i] ** 2, 24 * self.time_interval[i], 6,0, 0, 0, 0, 0, -6, 0, 0, 0],
                                          [120 * self.time_interval[i], 24, 0, 0, 0, 0, 0, -24, 0, 0, 0, 0],

                                          ])


            b[-3 + 6 * i:3 + 6 * i, :] = np.array([self.points[i], self.points[i], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])


        self.Coef = np.linalg.solve(A_Matrix,b)
        # print(self.Coef)
        print(self.Coef.shape)


        print('---------')
        # print(self.Coef)
        # print('selfpoints: ', self.points.shape[0])
        # print(self.time_interval.shape[0])
        # print('---------')
        # # print('time', self.time_interval)
        # print('accumulated time:  ', self.accumulated_time)
        print(len(self.accumulated_time))




    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))  # Position
        x_dot = np.zeros((3,))  # Velocity
        x_ddot = np.zeros((3,))  # Acceleration
        x_dddot = np.zeros((3,))  # Jerk
        x_ddddot = np.zeros((3,))  # Snap
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if t > self.total_time:
            x = self.points[-1]  # Position
            # print(self.points[-1].shape)
            x_dot = np.zeros((3,))  # Velocity
            x_ddot = np.zeros((3,))  # Acceleration
        else:
            for i in range(len(self.accumulated_time) - 1):
                if self.accumulated_time[i + 1] > t > self.accumulated_time[i]:
                    dt = t - self.accumulated_time[i]
                    [x, x_dot, x_ddot, x_dddot, x_ddddot] = np.array(
                        [[dt ** 5, dt ** 4, dt ** 3, dt ** 2, dt ** 1, 1],
                         [5 * dt ** 4, 4 * dt ** 3, 3 * dt ** 2, 2 * dt, 1, 0],
                         [20 * dt ** 3, 12 * dt ** 2, 6 * dt, 2, 0, 0],
                         [60 * dt ** 2, 24 * dt, 6, 0, 0, 0],
                         [120 * dt, 24, 0, 0, 0, 0]
                         ]) @ self.Coef[6 * i:6 * i + 6, :]
                    break



        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                                   'yaw': yaw, 'yaw_dot': yaw_dot}




                    # The following old approach is minimum jerk, stop at each waypoints

                    # print(self.accumulated_time[i+1])
                    # position = lambda time: self.Coef[i, 0] * np.power(time, 5) + self.Coef[i, 1] * np.power(time, 4) + \
                    #                      self.Coef[i, 2] * np.power(time, 3) + self.Coef[i, 3] * np.power(time, 2) + \
                    #                      self.Coef[i, 4] * np.power(time, 1) + self.Coef[i, 5] * np.power(time, 0)
                    #
                    # velocity = lambda time: 5 * np.power(time, 4) * self.Coef[i, 0] + 4 * np.power(time, 3) * self.Coef[i, 1] + 3 * np.power(time, 2) * self.Coef[i, 2] + 2 * np.power(time, 1) * self.Coef[i, 3] + self.Coef[i, 4]
                    # acceleration = lambda time: 20 * np.power(time, 3) * self.Coef[i, 0] + 12 * np.power(time, 2) * self.Coef[i, 1] + 6 * np.power(time, 1) * self.Coef[i, 2] + 2 * self.Coef[i, 3]
                    #
                    # temp_pos = position(t - self.accumulated_time[i])
                    # # print(t - self.accumulated_time[i])
                    #
                    # temp_vel = velocity(t - self.accumulated_time[i])
                    #
                    # temp_acc = acceleration(t - self.accumulated_time[i])



                    # direction = (self.points[i + 1] - self.points[i]) / np.linalg.norm(self.points[i + 1] - self.points[i])
                    # x = self.points[i] + direction * temp_pos
                    # x_dot = direction * temp_vel
                    # x_ddot = direction * temp_acc

        return flat_output
