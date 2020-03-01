import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        self.Kd = np.diag(np.array([4.5, 4.5, 4.5]))
        self.Kp = np.diag(np.array([8.5, 8.5, 8]))
        self.K_R = np.diag(np.array([2400, 2400, 400]))
        self.K_w = np.diag(np.array([60, 60, 50]))

        # self.Kd = np.diag(np.array([4.5, 4.5, 4.5]))
        # self.Kp = np.diag(np.array([0.1,0.1,0.1]))
        # self.K_R = np.diag(np.array([2400, 2400, 400]))
        # self.K_w = np.diag(np.array([60, 60, 50]))

        # -------------------------------------------------------------------------------------------

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys                       # State
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys  # Desired
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys  # Output
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE --------------------------------------------------------------------------------------------
        # define error

        error_pos = state.get('x') - flat_output.get('x')
        error_vel = state.get('v') - flat_output.get('x_dot')

        error_vel = np.array(error_vel).reshape(3, 1)
        error_pos = np.array(error_pos).reshape(3, 1)

        # Equation 26
        rdd_des = np.array(flat_output.get('x_ddot')).reshape(3, 1) - np.matmul(self.Kd, error_vel) - np.matmul(self.Kp,
                                                                                                                error_pos)
        # Equation 28
        F_des = (self.mass * rdd_des) + np.array([0, 0, self.mass * self.g]).reshape(3, 1)  # (3 * 1)

        # Find Rotation matrix
        R = Rotation.as_matrix(Rotation.from_quat(state.get('q')))  # Quaternions to Rotation Matrix
        # print(R.shape)
        # Equation 29, Find u1
        b3 = R[0:3, 2:3]

        # print(b3)
        u1 = np.matmul(b3.T, F_des)  # u1[0,0] to access value
        # print(np.transpose(b3))

        # ----------------------- the following is to  find u2 ---------------------------------------------------------

        # Equation 30
        b3_des = F_des / np.linalg.norm(F_des)  # 3 * 1
        a_Psi = np.array([np.cos(flat_output.get('yaw')), np.sin(flat_output.get('yaw')), 0]).reshape(3, 1)  # 3 * 1
        b2_des = np.cross(b3_des, a_Psi, axis=0) / np.linalg.norm(np.cross(b3_des, a_Psi, axis=0))
        b1_des = np.cross(b2_des, b3_des, axis=0)

        # Equation 33
        R_des = np.hstack((b1_des, b2_des, b3_des))
        # print(R_des)

        # Equation 34
        # R_temp = 0.5 * (np.matmul(np.transpose(R_des), R) - np.matmul(np.transpose(R), R_des))
        temp = R_des.T @ R - R.T @ R_des
        R_temp = 0.5 * temp
        # orientation error vector
        e_R = 0.5 * np.array([-R_temp[1, 2], R_temp[0, 2], -R_temp[0, 1]]).reshape(3, 1)
        # Equation 35
        u2 = self.inertia @ (-self.K_R @ e_R - self.K_w @ (np.array(state.get('w')).reshape(3, 1)))

        gama = self.k_drag / self.k_thrust
        Len = self.arm_length
        cof_temp = np.array(
            [1, 1, 1, 1, 0, Len, 0, -Len, -Len, 0, Len, 0, gama, -gama, gama, -gama]).reshape(4, 4)

        u = np.vstack((u1, u2))

        F_i = np.matmul(np.linalg.inv(cof_temp), u)

        for i in range(4):
            if F_i[i, 0] < 0:
                F_i[i, 0] = 0
                cmd_motor_speeds[i] = self.rotor_speed_max
            cmd_motor_speeds[i] = np.sqrt(F_i[i, 0] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max

        cmd_thrust = u1[0, 0]
        cmd_moment[0] = u2[0, 0]
        cmd_moment[1] = u2[1, 0]
        cmd_moment[2] = u2[2, 0]
        cmd_q = Rotation.as_quat(Rotation.from_matrix(R_des))
        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}

        return control_input
