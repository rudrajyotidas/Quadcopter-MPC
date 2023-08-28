from Quadcopter import *
from Controller import *
from traj_utils import *
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

Ts = 0.1

# Discretize the Dynamics
Ad, Bd, *rest = cont2discrete((A,B,C,D), Ts, method='bilinear')

# Controller gain
ctrl = Controller(Ad, Bd)

xmax = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])*3
xmin = -xmax
umax = np.array([1, 1, 1, 1])*3
umin = np.array([-1, -1, -1, -1])*3
dumax = np.array([1, 1, 1, 1])*20
dumin = -dumax

state_input_constraints = (umin, umax, dumin, dumax, xmin, xmax)

P = 50*np.ones(12)
Q = 20*np.ones(12)
R = 0.1*np.ones(4)

N = 15

prob, references, x_init, u_prev, du = ctrl.offsetFreeMPCTracking(P, Q, R, N, state_input_constraints)

# Random intial conditions
p0 = np.random.normal(0, 0.1, size=(3)) + np.array([np.sin(np.pi/4), np.cos(np.pi/4), 0])*2  # IMPORTANT
p0_dot = np.random.normal(0, 0.1, size=(3))
psi0 = np.random.normal(0, 0.5, size=(3))*0
psi0_dot = np.random.normal(0, 0.1, size=(3))*0

X0 = np.hstack((p0, p0_dot, psi0, psi0_dot))
Xeq = np.zeros(12)[:, np.newaxis]
Uhover = qcop.HoverThrusts()[:, np.newaxis]

print('Initial Condition: ', X0)

# Start simulation
timesteps = 600
X_traj = np.zeros((12, timesteps))
U_traj = np.zeros((4, timesteps))

deltaU = np.zeros((4,1))
full_state_p = X0 # IMPORTANT

sim = QuadcopterSimulator(qcop, Ts)
full_state = X0

# Generate Trajectory
trajGen = Trajectory(Ts, (timesteps+N)*Ts)
trajGen.CircularTraj(2, np.pi/4)
#trajGen.LinearTraj(0.1)

# Loop
for i in range(timesteps):

    x_init.value = np.hstack((full_state-full_state_p, full_state))
    u_prev.value = np.squeeze(deltaU)
    references.value = trajGen.getReferences(N+1, i)

    prob.solve(solver='OSQP', warm_start=True)

    deltaU += (du.value[:, 0])[:, np.newaxis]
    U_ctrl = Uhover + deltaU

    full_state_p = full_state

    full_state = sim.step(full_state, np.squeeze(qcop.M @ U_ctrl))
    full_state[6:9] = (full_state[6:9] + np.pi)%(2*np.pi) - np.pi

    X_traj[:, i] = full_state
    U_traj[:, i] = U_ctrl.squeeze()

# Plot results

fig, axs = plt.subplots(3,1, figsize=(20, 20))
axs[0].plot(X_traj[0, :])
axs[0].plot(trajGen.x_traj[:timesteps])
axs[1].plot(X_traj[1, :])
axs[1].plot(trajGen.y_traj[:timesteps])
axs[2].plot(X_traj[2, :])
axs[2].plot(trajGen.z_traj[:timesteps])

# axs[0][0].plot(X_traj[0, :])
# axs[0][0].plot(trajGen.x_traj[:timesteps])
# axs[0][1].plot(X_traj[1, :])
# axs[0][1].plot(trajGen.y_traj[:timesteps])
# axs[0][2].plot(X_traj[2, :])
# axs[0][2].plot(trajGen.z_traj[:timesteps])
# axs[0][3].plot(X_traj[3, :])
# axs[1][0].plot(X_traj[4, :])
# axs[1][1].plot(X_traj[5, :])
# axs[1][2].plot(X_traj[6, :])
# axs[1][3].plot(X_traj[7, :])
# axs[2][0].plot(X_traj[8, :])
# axs[2][1].plot(X_traj[9, :])
# axs[2][2].plot(X_traj[10, :])
# axs[2][3].plot(X_traj[11, :])
# axs[3][0].plot(U_traj[0, :])
# axs[3][1].plot(U_traj[1, :])
# axs[3][2].plot(U_traj[2, :])
# axs[3][3].plot(U_traj[3, :])

plt.show()


