import numpy as np

class Trajectory:

    def __init__(self, Ts, T):

        self.T = T
        self.Ts = Ts
        self.samples = int(T/Ts)
        self.x_traj = np.zeros(self.samples)
        self.y_traj = np.zeros(self.samples)
        self.z_traj = np.zeros(self.samples)
        self.times = np.linspace(0, T, num=self.samples)

    def CircularTraj(self, R, theta):

        omega = 90/(np.pi*self.T)
        self.z_traj = R*np.sin(omega*self.times)
        self.x_traj = np.cos(theta)*R*np.cos(omega*self.times)
        self.y_traj = np.sin(theta)*R*np.cos(omega*self.times)

    def LinearTraj(self, m):

        self.x_traj = m*self.times
        self.y_traj = m*self.times
        self.z_traj = m*self.times
        pass

    def EightTraj(self, Ts, T):

        pass

    def getReferences(self, N, t_step):

        R = np.zeros((12, N))
        start_index = int(t_step)
        R[0, :] = self.x_traj[start_index:start_index + N]
        R[1, :] = self.y_traj[start_index:start_index + N]
        R[2, :] = self.z_traj[start_index:start_index + N]

        return R