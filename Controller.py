import numpy as np
from scipy import linalg as la
from scipy import sparse
import cvxpy as cv

class Controller:

    def __init__(self, A, B):

        '''
        Intialise using A and B matrices post linearisation
        '''

        self.A = A
        self.B = B

    def K_LQR(self, Q, R):

        '''
        Returns LQR Gain
        u = K@x
        '''

        P = la.solve_discrete_are(self.A, self.B, Q, R)
        K = -np.linalg.inv(self.B.T@P@self.B + R)@self.B.T@P@self.A

        return K
    
    def MPC_unconstrained(self, P, Q, R, N):

        '''
        P: Terminal cost
        Q: State cost
        R: Input cost
        N: Horizon Length

        Returns N gain matrices F(i) over the horizon.
        u(k) = F(i)x(i)

        For MPC apply control input u = F(0)x
        '''

        P_k = list()
        F_k = list()

        P_k.append(P)
        # F_k.append(-np.linalg.inv(self.B.T@P@self.B + R)@self.B.T@P@self.A)

        for i in range(N):

            P = self.A.T@P@self.A + Q \
                -self.A.T@P@self.B@(np.linalg.inv(self.B.T@P@self.B + R))@self.B.T@P@self.A
            
            P_k.append(P)

            F = -(np.linalg.inv(self.B.T@P@self.B + R)@self.B.T@P@self.A)
            F_k.append(F)

        F_k.reverse()
        P_k.reverse()

        return F_k, P_k
    
    def constrainedMPC(self, P, Q, R, N, constr):

        '''
        P: Terminal cost (array with diagonal elements)
        Q: State cost (array with diagonal elements)
        R: Input cost (array with diagonal elements)
        N: Horizon Length

        constr: 6-tuple of bounding values (umin, umax, xmin, xmax, xf_min, xf_max)
        '''

        A = sparse.csc_matrix(self.A)
        B = sparse.csc_matrix(self.B)

        [nx, nu] = B.shape

        umin = constr[0]
        umax = constr[1]
        xmin = constr[2]
        xmax = constr[3]
        xf_min = constr[4]
        xf_max = constr[5]

        P = sparse.diags(P)
        Q = sparse.diags(Q)
        R = sparse.diags(R)

        # Decision variables
        u = cv.Variable((nu, N))
        x = cv.Variable((nx, N+1))
        x_init = cv.Parameter(nx)

        obj = 0
        constraints = [x[:,0] == x_init]

        # Construct objective function and constraints

        for k in range(N):

            obj += cv.quad_form(x[:, k], Q) + cv.quad_form(u[:, k], R)

            constraints += [x[:, k+1] == A@x[:, k] + B@u[:, k]]
            constraints += [xmin <= x[:,k], x[:,k] <= xmax]
            constraints += [umin <= u[:,k], u[:,k] <= umax]

        obj += cv.quad_form(x[:, N], P)
        constraints += [xf_min <= x[:, N], x[:, N] <= xf_max]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, x_init, u
    
    def augmented_MPC(self, P, Q, R, N, constr):

        '''
        P: Terminal cost (array with diagonal elements)
        Q: State cost (array with diagonal elements)
        R: Input cost (array with diagonal elements)
        N: Horizon Length

        constr: 8-tuple of bounding values (umin, umax, dumin, dumax, xmin, xmax, xf_min, xf_max)
        '''

        A_aug = np.block([[self.A, np.zeros((12, 12))],
                          [self.A, np.eye(12)]])
        
        B_aug = np.block([[self.B],
                          [self.B]])
        
        A = sparse.csc_matrix(A_aug)
        B = sparse.csc_matrix(B_aug)

        [nx_aug, nu] = B.shape
        nx = int(nx_aug/2)

        umin = constr[0]
        umax = constr[1]
        d_umin = constr[2]
        d_umax = constr[3]
        xmin = constr[4]
        xmax = constr[5]
        xf_min = constr[6]
        xf_max = constr[7]

        P = sparse.diags(P)
        Q = sparse.diags(Q)
        R = sparse.diags(R)

        # Decision variables
        du = cv.Variable((nu, N))
        x = cv.Variable((nx_aug, N+1))
        x_init = cv.Parameter(nx_aug)
        u_prev = cv.Parameter(nu)
        u = cv.Variable((nu, N+1))

        obj = 0
        constraints = [x[:,0] == x_init]
        u[:, 0] == u_prev

        for k in range(N):

            obj += cv.quad_form(x[nx:, k], Q) + cv.quad_form(du[:, k], R)

            u[:, k+1] == u[:, k] + du[:, k]

            constraints += [x[:, k+1] == A@x[:, k] + B@du[:, k]]
            constraints += [xmin <= x[nx:, k], x[nx:, k] <= xmax]
            constraints += [d_umin <= du[:,k], du[:,k] <= d_umax]
            constraints += [umin <= u[:, k+1], u[:, k+1] <= umax]

        obj += cv.quad_form(x[nx:, N], P)
        constraints += [xf_min <= x[nx:, N], x[nx:, N] <= xf_max]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, x_init, u_prev, du
    
    def MPCTracking(self, P, Q, R, N, constr):

        '''
        P: Terminal cost (array with diagonal elements)
        Q: State cost (array with diagonal elements)
        R: Input cost (array with diagonal elements)
        N: Horizon Length

        constr: 4-tuple of bounding values (umin, umax, xmin, xmax)
        '''

        A = sparse.csc_matrix(self.A)
        B = sparse.csc_matrix(self.B)

        [nx, nu] = B.shape

        umin = constr[0]
        umax = constr[1]
        xmin = constr[2]
        xmax = constr[3]
        #xf_min = constr[4]
        #xf_max = constr[5]

        P = sparse.diags(P)
        Q = sparse.diags(Q)
        R = sparse.diags(R)

        # Decision variables
        u = cv.Variable((nu, N))
        x = cv.Variable((nx, N+1))
        ref = cv.Parameter((nx, N+1))
        x_init = cv.Parameter(nx)

        obj = 0
        constraints = [x[:,0] == x_init]

        # Construct objective function and constraints

        for k in range(N):

            obj += cv.quad_form(x[:, k] - ref[:, k], Q) + cv.quad_form(u[:, k], R)

            constraints += [x[:, k+1] == A@x[:, k] + B@u[:, k]]
            constraints += [xmin <= x[:,k], x[:,k] <= xmax]
            constraints += [umin <= u[:,k], u[:,k] <= umax]

        obj += cv.quad_form(x[:, N] - ref[:, N], P)
        #constraints += [xf_min <= x[:, N], x[:, N] <= xf_max]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, ref, x_init, u
    
    def offsetFreeMPCTracking(self, P, Q, R, N, constr):

        '''
        P: Terminal cost (array with diagonal elements)
        Q: State cost (array with diagonal elements)
        R: Input cost (array with diagonal elements)
        N: Horizon Length

        constr: 6-tuple of bounding values (umin, umax, dumin, dumax, xmin, xmax)
        '''

        A_aug = np.block([[self.A, np.zeros((12, 12))],
                          [self.A, np.eye(12)]])
        
        B_aug = np.block([[self.B],
                          [self.B]])
        
        A = sparse.csc_matrix(A_aug)
        B = sparse.csc_matrix(B_aug)

        [nx_aug, nu] = B.shape
        nx = int(nx_aug/2)

        umin = constr[0]
        umax = constr[1]
        d_umin = constr[2]
        d_umax = constr[3]
        xmin = constr[4]
        xmax = constr[5]
        # xf_min = constr[6]
        # xf_max = constr[7]

        P = sparse.diags(P)
        Q = sparse.diags(Q)
        R = sparse.diags(R)

        # Decision variables
        du = cv.Variable((nu, N))
        x = cv.Variable((nx_aug, N+1))
        x_init = cv.Parameter(nx_aug)
        ref = cv.Parameter((nx, N+1))
        u_prev = cv.Parameter(nu)
        u = cv.Variable((nu, N+1))

        obj = 0
        constraints = [x[:,0] == x_init]
        u[:, 0] == u_prev

        for k in range(N):

            obj += cv.quad_form(x[nx:, k] - ref[:, k], Q) + cv.quad_form(du[:, k], R)

            u[:, k+1] == u[:, k] + du[:, k]

            constraints += [x[:, k+1] == A@x[:, k] + B@du[:, k]]
            constraints += [xmin <= x[nx:, k], x[nx:, k] <= xmax]
            constraints += [d_umin <= du[:,k], du[:,k] <= d_umax]
            constraints += [umin <= u[:, k+1], u[:, k+1] <= umax]

        obj += cv.quad_form(x[nx:, N] - ref[:, N], P)
        # constraints += [xf_min <= x[nx:, N], x[nx:, N] <= xf_max]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, ref, x_init, u_prev, du
