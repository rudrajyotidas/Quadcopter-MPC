import numpy as np
from scipy.integrate import RK45
from utils import *

class Quadcopter:

    def __init__(self, parameters):

        '''
        Initialise Quadcopter object with parameters
        '''

        # Distance of propellers from body-X-axis
        self.x_1 = parameters[0]
        self.x_2 = parameters[1]
        self.x_3 = parameters[2]
        self.x_4 = parameters[3]

        # Distance of propellers from body-Y-axis
        self.y_1 = parameters[4]
        self.y_2 = parameters[5]
        self.y_3 = parameters[6]
        self.y_4 = parameters[7]

        # Torque to thrust correlation coeffecient for each propeller
        self.c_1 = parameters[8]
        self.c_2 = parameters[9]
        self.c_3 = parameters[10]
        self.c_4 = parameters[11]

        # Moment of Inertia
        self.J = np.diag(parameters[12:15])

        # Mass of quadcopter
        self.m = parameters[15]

        # Gravity
        self.g = 9.8

        # M_layout matrix
        self.M = np.array([[1, 1, 1, 1],
                      [self.y_1, self.y_2, self.y_3, self.y_4],
                      [-self.x_1, -self.x_2, -self.x_3, -self.x_4],
                      [self.c_1, self.c_2, self.c_3, self.c_4]
                      ])

        # Initialise full-state with a column vector of 12 zeros
        self.full_state = np.zeros(12)
        self.full_state = self.full_state[:, np.newaxis]

        # Initialise control-input as a column vector of 4 zeros
        self.u = np.zeros(4)
        self.u = self.u[:, np.newaxis]

    def dynamics(self, X, U):

        '''
        X: Column vector of full state (12 elements)
        U: Column vector of control inputs (4 elements). U accepts the 'actuator inputs' as defined in the book and not thrusts

        Returns X_dot
        '''
        p = X[0:3, :]
        p_dot = X[3:6, :]
        psi = X[6:9, :]
        psi_dot = X[9:, :]

        # Define vectors for convenience
        G = np.array([0, 0, -self.m*self.g])
        G = G[:, np.newaxis]

        U1 = np.array([0, 0, U[0, 0]])
        U1 = U1[:, np.newaxis]

        U2 = U[1:, :]

        # Calculate Matrices for convenience
        R = Rot_B_I(psi)
        T_psi = T(psi)
        T_psi_inv = T_inv(psi)
        T_psi_dot = T_dot(np.vstack((psi, psi_dot)))

        # Dynamics
        w = T_psi @ psi_dot
        w_dot = np.linalg.inv(self.J)@(U2 \
                                    - np.squeeze(np.cross(w.T, (self.J @ w).T))[:, np.newaxis])
        
        psi_ddot = T_psi_inv@(w_dot - T_psi_dot@psi_dot)

        p_ddot = (R@U1 + G)/self.m

        return np.vstack((p_dot, p_ddot, psi_dot, psi_ddot))
    
    def modelRK(self, t, X_U):

        '''
        For ease to interface with RK45, takes inputs in the form of row vector and returns a row vector
        Concatenated X and U together
        Takes time as input but not uses it because dynamics does not depend on time
        '''
        X = X_U[:12]
        U = X_U[12:]

        X = X[:, np.newaxis]
        U = U[:, np.newaxis]

        X_dot = self.dynamics(X, U)
        U_dot = np.zeros((4,1))

        return np.squeeze(np.vstack((X_dot, U_dot)))
    
    def HoverDynamics(self):

        '''
        Returns A and B about equilibrium point assuming hover
        '''

        b1 = np.zeros((3,3))
        b2 = np.eye(3)
        b3 = np.array([[0, self.g, 0],
                        [-self.g, 0, 0],
                        [0, 0, 0]])
        b4 = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1/self.m, 1/self.m, 1/self.m, 1/self.m]])
        b5 = np.linalg.inv(self.J)@(self.M[1:, :])
        
        A_hover = np.block([[b1, b2, b1, b1],
                            [b1, b1, b3, b1],
                            [b1, b1, b1, b2],
                            [b1, b1, b1, b1]])
        
        B_hover = np.vstack((np.zeros((3, 4)), b4, np.zeros((3, 4)), b5))

        return A_hover, B_hover
    
    def HoverThrusts(self):

        '''
        Returns row vector of thrusts to keep hovering
        '''

        f = np.array([self.m*self.g, 0, 0, 0])

        thrusts = np.linalg.solve(self.M, f)

        return thrusts


class QuadcopterSimulator:

    def __init__(self, qc_obj:Quadcopter, Ts=0.05):

        '''
        Initialise with a Quadcopter object and a sampling time Ts
        Ts must be greater than 0.05
        '''

        self.qc_obj = qc_obj
        self.Ts = Ts

    def step(self, X, U):

        '''
        Takes row/column vectors X(t) and U(t) as input
        Returns row vector X(t+Ts)
        '''

        X0_U0 = np.concatenate((np.squeeze(X), np.squeeze(U)))
        qc_sol = RK45(fun=self.qc_obj.modelRK, t0=0, y0=X0_U0, t_bound=self.Ts, max_step=0.01, rtol=1e-5)

        status = 'running'

        while status=='running':

            qc_sol.step()
            status = qc_sol.status

        return (qc_sol.y[:12])