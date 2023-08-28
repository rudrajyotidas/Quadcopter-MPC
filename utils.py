import numpy as np

def Rot_B_I(attitude):

    '''
    Input: 
    Returns the 3X3 Rotation Matrix that transforms Body frame to the Inertial frame
    v_I = R*v_B
    '''

    gamma = attitude[0, 0]
    beta = attitude[1, 0] 
    alpha = attitude[2, 0]

    R = np.array([[np.cos(alpha)*np.cos(beta), np.cos(beta)*np.sin(alpha), -np.sin(beta)],
                  [-np.cos(gamma)*np.sin(alpha) + np.sin(gamma)*np.sin(beta)*np.cos(alpha), np.cos(gamma)*np.cos(alpha) + np.sin(gamma)*np.sin(beta)*np.sin(alpha), np.sin(gamma)*np.cos(beta)],
                  [np.sin(gamma)*np.sin(alpha) + np.cos(gamma)*np.sin(beta)*np.cos(alpha), -np.sin(gamma)*np.cos(alpha) + np.cos(gamma)*np.sin(beta)*np.sin(alpha), np.cos(gamma)*np.cos(beta)]
                  ])
    
    return R.T

def T(attitude):

    '''
    Input:
    Returns the 3X3 transformation matrix between Body Angular Velocities and Euler Angle rates
    w = T_psi*psi_dot
    '''

    gamma = attitude[0, 0]
    beta = attitude[1, 0] 
    alpha = attitude[2, 0]

    T_psi = np.array([[1, 0, -np.sin(beta)],
                      [0, np.cos(gamma), np.sin(gamma)*np.cos(beta)],
                      [0, -np.sin(gamma), np.cos(gamma)*np.cos(beta)]
                      ])
    return T_psi

def T_inv(attitude):

    '''
    Input:
    Returns the inverse of the transformation matrix
    '''

    gamma = attitude[0, 0]
    beta = attitude[1, 0] 
    alpha = attitude[2, 0]

    T_psi_inv = np.array([[1, np.sin(gamma)*np.tan(beta), np.cos(gamma)*np.tan(beta)],
                          [0, np.cos(gamma), -np.sin(gamma)],
                          [0, np.sin(gamma)/np.cos(beta), np.cos(gamma)/np.cos(beta)]
                          ])
    
    return T_psi_inv

def T_dot(attitude_and_rates):

    '''
    Input:
    Returns the derivative of the transformation matrix with respect to euler angle rates
    '''

    gamma = attitude_and_rates[0, 0]
    beta = attitude_and_rates[1, 0] 
    alpha = attitude_and_rates[2, 0]
    gamma_d = attitude_and_rates[3, 0]
    beta_d = attitude_and_rates[4, 0] 
    alpha_d = attitude_and_rates[5, 0]

    T_psi_dot = np.array([[1, 0, -beta_d*np.cos(beta)],
                          [0, -gamma_d*np.sin(gamma), gamma_d*np.cos(gamma)*np.cos(beta) - beta_d*np.sin(gamma)*np.sin(beta)],
                          [0, -gamma_d*np.cos(gamma), -gamma_d*np.sin(gamma)*np.cos(beta) - beta_d*np.cos(gamma)*np.sin(beta)]
                          ])
    
    return T_psi_dot

def thrust2U(M_layout, thrusts):

    '''
    Input: Column vector with thrusts by the propellers
    Returns control inputs for actuators as a column vector
    '''
    return M_layout@thrusts

