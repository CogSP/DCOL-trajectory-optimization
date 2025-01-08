import numpy as np
import random
import copy
from proximity.proximity import proximity_mrp
from proximity.proximity_gradient import proximity_gradient
from primitives.misc_primitive_constructor import create_rect_prism, ConeMRP
from primitives.mass_properties import mass_properties
import math

def skew(w):
    """
    Skew-symmetric matrix from a 3D vector.
    """
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def rigid_body_dynamics(J, m, x, u):
    """
    Compute the dynamics of the rigid body.
    
    Args:
        J: Inertia matrix
        m: mass
        x: State vector [position (3), velocity (3), MRP (3), angular velocity (3)].
        u: Control inputs (rotor speeds).
    
    Returns:
        np.ndarray: Derivative of the state vector.
    """
    #Extract state variables
    r = x[:3]        # Position
    v = x[3:6]       # Velocity
    p = x[6:9]       # Attitude (MRP)
    omega = x[9:12]  # Angular velocity

    #Extract control variables
    f=u[:3]
    tau=u[3:6]

    # Dynamics calculation
    v_dot = f / m
    norm_p = np.linalg.norm(p)
    skew_p = skew(p)
    p_dot = ((1 + norm_p**2) / 4) * (
        np.eye(3) + 2 * (np.dot(skew_p, skew_p) + skew_p) / (1 + norm_p**2)
    ).dot(omega)
    omega_dot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

    return np.concatenate([v, v_dot, p_dot, omega_dot])

def dynamics(p, x, u, k):
    """
    Wrapper function.

    Args:
        p: Dictionary or named tuple with keys 'J' and 'm'.
        x: State vector.
        u: Control input.
        k: Current timestep index.

    Returns:
        np.ndarray: Derivative of the state vector.
    """
    return rigid_body_dynamics(p['J'], p['m'], x, u)

def discrete_dynamics(p, x, u, k):
    """
    Simulates system dynamics using RK4 integration.

    Args:
        p (dict): Simulation parameters.
        x (ndarray): Current state.
        u (ndarray): Control input.
        k (int): Current timestep index.

    Returns:
        ndarray: Updated state after integration.
    """
    dt = p['dt']
    k1 = dt * dynamics(p, x, u, k)
    k2 = dt * dynamics(p, x + 0.5 * k1, u, k)
    k3 = dt * dynamics(p, x + 0.5 * k2, u, k)
    k4 = dt * dynamics(p, x + k3, u, k)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def inequality_constraints_u(p, u):
    """
    Inequality constraints for the control inputs.

    Args:
        p (dict): Dictionary containing the environment parameters, including control dimensions.
        u (numpy.ndarray): Control vector.

    Returns:
        numpy.ndarray: Stacked inequality constraints for all control inputs.
    """
    u_max = p['u_max']
    u_min = p['u_min']
    return np.concatenate([u - u_max, -u + u_min])

def inequality_constraints_u_grad(p, u):
    """
    Compute the Jacobian matrix for control inequality constraints.

    Args:
        p (dict): Dictionary containing the environment parameters, including control dimensions.
        u (numpy.ndarray): Control vector.

    Returns:
        numpy.ndarray: Stacked Jacobian matrix for all control constraints.
    """
    nu = p["nu"]
    identity = np.eye(nu)
    return np.vstack([identity, -identity])

def inequality_constraints_x(p, x):
    """
    Calculates state-based inequality constraints.

    Args:
        p (dict): Simulation parameters.
        x (ndarray): Current state.

    Returns:
        ndarray: Constraint violations for the current state.
    """
    p['P_vic'].r = np.array(x[0:3])
    p['P_vic'].p = np.array(x[6:9])
    
    contacts = []
    for obs in p['P_obs']:
        alpha, _ = proximity_mrp(p['P_vic'], obs, verbose=False)
        contacts.append(1 - alpha)
        
    return np.array(contacts)

def inequality_constraints_x_grad(p, x):
    """
    Compute the Jacobian matrix for state inequality constraints.

    Args:
        p (dict): Dictionary containing the environment parameters, including P_vic and P_obs.
        x (numpy.ndarray): State vector.

    Returns:
        numpy.ndarray: Stacked Jacobian matrix for all constraints.
    """
    # Update P_vic's position and orientation based on the state vector
    p["P_vic"].r = np.array(x[:3])  # Position
    p["P_vic"].p = np.array(x[6:9])  # Orientation (Modified Rodrigues Parameters, MRP)

    # Calculate the Jacobians from the DCOL library
    Js = [proximity_gradient(p["P_vic"], obs)[1] for obs in p["P_obs"]]
    # print("\n Js:\n ", Js)


    # Extract relevant portions of the Jacobians for each constraint
    contact_J = [
        np.hstack([
            -Js[i][:3].reshape(1, -1),  
            np.zeros((1, 3)),           
            -Js[i][3:6].reshape(1, -1), 
            np.zeros((1, 3))            
        ])
        for i in range(len(p["P_obs"]))
    ]


    # Stack all Jacobians vertically
    return np.vstack(contact_J)

def linear_interp(dt, x0, xg, N):
    """
    Linearly interpolates between two points x0 and xg over N points.
    
        Refer to the same function in the Quadrotor system for Args and Returns

    """

    delta_p = (xg[0:3] - x0[0:3])
    positions = np.array([((i-1)*(delta_p/(N-1)) + x0[0:3]) for i in range(1, N+1)])
    
    delta_p = (xg[6:9] - x0[6:9])
    attitudes = np.array([((i-1)*(delta_p/(N-1)) + x0[6:9]) for i in range(1, N+1)])

    # Assertions to verify the interpolation
    assert np.array_equal(positions[0], x0[0:3]), "Initial position does not match x0[0:3]"
    assert np.array_equal(positions[-1], xg[0:3]), "Final position does not match xg[0:3]"
    assert np.array_equal(attitudes[0], x0[6:9]), "Initial attitude does not match x0[6:9]"
    assert np.array_equal(attitudes[-1], xg[6:9]), "Final attitude does not match xg[6:9]"

    #possible error-->/N-1
    velocities = [delta_p/((N-1)*dt) for _ in range(N)]

    interpolated_states = [
        np.concatenate([positions[i], velocities[i], attitudes[i], np.zeros(3)]) for i in range(N)
    ]

    return interpolated_states

def mrp_from_q(q):
    """
    Converts a quaternion in Modified Rodrigues Parameters (MRP).
    """
    return np.array(q[1:4]) / (1 + q[0])

def initialize_coneThroughWall():
    """
    Initializes the parameters for the cone system.
    """
    print("Initializing parameters for the coneThroughWall...")
    # main parameters
    nx = 12
    nu = 6
    N = 60
    dt = 0.1

    max_linesearch_iters = 20
    atol = 1e-1
    max_iters = 3000
    reg_min = 1e-6
    reg = reg_min
    reg_max = 1e2
    rho=1e0 
    phi=10.0
    convio_tol = 1e-4

    X_hist = []
    U_hist = []
    hx_hist = []
    hu_hist = []
    
    # initial and final state, the state is [r;v;p;ω]
    x0 = np.array([-4, -7, 9, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0])
    xg = np.array([-4.5, 7, 3, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0])
    
    # reference trajectory
    Xref = linear_interp(dt,x0,xg,N)
    Uref = [np.zeros(nu) for _ in range(N-1)]

    # cost matrices
    Q = np.diag(np.ones(nx))
    Qf = np.diag(np.ones(nx))
    R = np.diag([1.,1.,1.,100.,100.,100.])

    P_vic=ConeMRP(height=2.0, beta=math.radians(22))
    mass,inertia = mass_properties(P_vic)

    # obstacles
    P_obs = [
        create_rect_prism(10.0, 10.0, 1.0),
        create_rect_prism(10.0, 10.0, 1.0),
        create_rect_prism(4.1, 4.1, 1.1),
        create_rect_prism(4.1, 4.1, 1.1),
    ]

    # obstacles position and orientation
    P_obs[0].r = np.array([-6, 0, 5.0])
    P_obs[0].p = mrp_from_q([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    P_obs[1].r = np.array([6, 0, 5.0])
    P_obs[1].p = mrp_from_q([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    P_obs[2].r = np.array([0, 0, 2.05])
    P_obs[2].p = mrp_from_q([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    P_obs[3].r = np.array([0, 0, 7.96])
    P_obs[3].p = mrp_from_q([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])

    # Limiti di controllo
    u_min = -20 * np.ones(nu)
    u_max =  20 * np.ones(nu)

    # Limiti di stato
    x_min = -20 * np.ones(nx)
    x_max =  20 * np.ones(nx)

    # Numero di vincoli
    ncx = len(P_obs)
    ncu = 2 * nu

    p = {
        "nx": nx,
        "nu": nu,
        "ncx": ncx,
        "ncu": ncu,
        "N": N,
        "Q": Q,
        'R': R,
        "Qf": Qf,
        "u_min": u_min,
        "u_max": u_max,
        "x_min": x_min,
        "x_max": x_max,
        "Xref": Xref,
        "Uref": Uref,
        "dt": dt,
        "m": mass,
        "J" : inertia,
        "P_obs": P_obs,
        "P_vic": P_vic,
        'max_linesearch_iters': max_linesearch_iters,
        'atol': atol,
        'max_iters': max_iters,
        'reg_min': reg_min,
        'reg': reg,
        'reg_max': reg_max,
        'rho': rho,
        'phi': phi,
        'convio_tol': convio_tol,
        'system': 'coneThroughWall',
        'X_hist': X_hist,
        'U_hist': U_hist,
        'hx_hist': hx_hist,
        'hu_hist': hu_hist,
    }

    X = [copy.deepcopy(x0) for _ in range(N)]


    # IMPORTANT NOTE: Differently from the other systems, here we decided to use a random generator with seed 2,
    # that will not generate the same numbers as the Julia implementation. However, the results are the same.
    np.random.seed(2)
    U = [0.01 * np.random.randn(nu) for _ in range(N-1)]

    Xn = copy.deepcopy(X)
    Un = copy.deepcopy(U)

    p['X_hist'].append(X)
    p['U_hist'].append(U)

    return p, Xn, Un