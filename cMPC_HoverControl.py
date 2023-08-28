from Quadcopter import *
from Controller import *
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

# Defining the parameters of the drone
x_baseline = 0.3
y_baseline = 0.3
c_baseline = 0.1

params = np.zeros(16)
params[:4] = np.array([-1, -1, 1, 1])*x_baseline
params[4:8] = np.array([-1, 1, -1, 1])*y_baseline
params[8:12] = np.array([-1, 1, 1, -1])*c_baseline
params[12:15] = np.array([11, 11, 22])*1e-4
params[15] = 0.2

# Create Quadcopter Object
qcop = Quadcopter(params)

# Dynamics about hover
A, B = qcop.HoverDynamics()
C = np.eye(12)
D = np.zeros((12, 4))

# Cost Matrices
Q = np.eye(12)*10
R = np.eye(4)

Ts = 0.1

# Discretize the Dynamics
Ad, Bd, *rest = cont2discrete((A,B,C,D), Ts, method='bilinear')

# Controller gain
ctrl = Controller(Ad, Bd)

xmax = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])*3
xmin = -xmax
umax = np.array([1, 1, 1, 1])
umin = np.array([-1, -1, -1, -1])
xf_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
xf_min = -xf_max

state_input_constraints = (umin, umax, xmin, xmax, xf_min, xf_max)

P = 50*np.ones(12)
Q = 10*np.ones(12)
R = np.ones(4)

N = 15

prob, x_init, u = ctrl.constrainedMPC(P, Q, R, 15, state_input_constraints)

# Random intial conditions
p0 = np.random.normal(0, 1, size=(3))
p0_dot = np.random.normal(0, 0.1, size=(3))
psi0 = np.random.normal(0, 0.5, size=(3))
psi0_dot = np.random.normal(0, 0.1, size=(3))

X0 = np.hstack((p0, p0_dot, psi0, psi0_dot))
Xeq = np.zeros(12)[:, np.newaxis]
Ueq = qcop.HoverThrusts()[:, np.newaxis]

print('Initial Condition: ', X0)

# Start simulation
timesteps = 100
X_traj = np.zeros((12, timesteps))
U_traj = np.zeros((4, timesteps))

sim = QuadcopterSimulator(qcop, Ts)
full_state = X0

for i in range(timesteps):

    x_init.value = full_state
    prob.solve(solver='OSQP', warm_start=True)
    U_ctrl = Ueq + (u.value[:, 0])[:, np.newaxis]
    full_state = sim.step(full_state, np.squeeze(qcop.M @ U_ctrl))
    full_state[6:9] = (full_state[6:9] + np.pi)%(2*np.pi) - np.pi
    X_traj[:, i] = full_state
    U_traj[:, i] = U_ctrl.squeeze()

# Plot results

fig, axs = plt.subplots(4,4, figsize=(30, 30))
axs[0][0].plot(X_traj[0, :])
axs[0][1].plot(X_traj[1, :])
axs[0][2].plot(X_traj[2, :])
axs[0][3].plot(X_traj[3, :])
axs[1][0].plot(X_traj[4, :])
axs[1][1].plot(X_traj[5, :])
axs[1][2].plot(X_traj[6, :])
axs[1][3].plot(X_traj[7, :])
axs[2][0].plot(X_traj[8, :])
axs[2][1].plot(X_traj[9, :])
axs[2][2].plot(X_traj[10, :])
axs[2][3].plot(X_traj[11, :])
axs[3][0].plot(U_traj[0, :])
axs[3][1].plot(U_traj[1, :])
axs[3][2].plot(U_traj[2, :])
axs[3][3].plot(U_traj[3, :])

plt.show()


