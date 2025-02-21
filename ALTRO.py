import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.plots import plot_cost, plot_trajectories, plot_constraint_violations, plot_regularization
import copy
from scipy.linalg import solve_triangular
import importlib
from scipy.linalg import cho_factor, cho_solve


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def eval_mask(mu, h):
    """
    Evaluate the mask for the augmented Lagrangian.

    Args:
        mu: Dual variable.
        h: Constraint violation.

    Returns:
        np.array: Mask for the augmented Lagrangian.
    """

    mask = np.diag(np.zeros(len(h)))
    for i in range(len(h)):
        mask[i,i] = (mu[i] > 0 or h[i] > 0)
    return mask


def calc_max_k(k):
    """
    Calculate the maximum norm of the control gains.

    Args:
        k: List of control gains.

    Returns:
        float: Maximum norm of the control gains.
    """

    km = 0
    for i in range(len(k)):
        km = max(km, np.linalg.norm(k[i]))
    return km


def update_reg(params, alpha):
    """
    Update the regularization parameter based on the forward pass step size.

    Args:
        params: Dictionary containing parameters such as reg, reg_min, reg_max.
        alpha: Step size from the forward pass.

    Returns:
        float: Updated regularization parameter.
    """

    # If the step size is zero, increase the regularization parameter
    if alpha == 0.0:
        if params['reg'] == params['reg_max']:
            raise ValueError("Regularization parameter reached maximum value.")
        return min(params['reg_max'], params['reg'] * 10)
    
    # If the step size is one, decrease the regularization parameter
    if alpha == 1.0:
        return max(params['reg_min'], params['reg'] / 10)

    # If the step size is between zero and one, keep the regularization parameter the same
    return params['reg']


def compute_jacobian(func, input_vector, delta=1e-6):
    """
    Computes the Jacobian of `func` at `input_vector` using finite differences.

    Args:
        func: Function to compute the Jacobian for. Should accept a vector and return a vector.
        input_vector: Vector at which to compute the Jacobian.
        delta: Small perturbation for finite differences.

    Returns:
        Jacobian matrix as a NumPy array.
    """
    n = len(input_vector)
    output = func(input_vector)
    m = len(output)
    jacobian = np.zeros((m, n))
    
    for i in range(n):
        perturbed_vector = input_vector.copy()
        perturbed_vector[i] += delta
        perturbed_output = func(perturbed_vector)
        jacobian[:, i] = (perturbed_output - output) / delta

    return jacobian


def compute_total_cost(params, X, U, mu, mux, lambd, system_module):
    """
    Compute the total cost for the current state and control trajectories.

    Args:
        params: Dictionary containing parameters such as Q, R, and reference trajectories.
        X: Current state trajectory.
        U: Current control trajectory.
        mu, mux, lambd: Augmented Lagrangian parameters.
        system_module: Module containing the system dynamics and constraints.

    Returns:
        float: Total cost.
    """

    N = params["N"]
    cost = 0
    for t in range(N - 1):

        # Running cost for each time step
        cost += running_cost(params, X[t], U[t], t)

        # AL term for the control constraints
        hu = system_module.inequality_constraints_u(params, U[t])
        mask = eval_mask(mu[t], hu)
        cost += np.dot(mu[t], hu) + 0.5 * params['rho'] * hu.T @ mask @ hu
       
        # AL term for the state constraints
        hx = system_module.inequality_constraints_x(params, X[t])
        mask = eval_mask(mux[t], hx)
        cost += np.dot(mux[t], hx) + 0.5 * params['rho'] * hx.T @ mask @ hx
       
    # AL term for the state constraints at the final time step
    cost += terminal_cost(params, X[-1], None, N - 1)
    hx = system_module.inequality_constraints_x(params, X[-1])
    mask = eval_mask(mux[-1], hx)
    cost += np.dot(mux[-1], hx) + 0.5 * params['rho'] * hx.T @ mask @ hx
    
    # AL term for the goal constraints
    hx = X[-1] - params['Xref'][-1]
    cost += np.dot(lambd, hx) + 0.5 * params['rho'] * hx.T @ hx
    
    return cost


def terminal_cost(params, x, u, t):
    """
    Compute the terminal cost for the final time step.

    Args:
        params: Dictionary containing parameters such as Q, R, and reference trajectories.
        x: Current state vector.
        u: Control -> not used for the terminal cost
        t: Time step index.
    Returns:
        float: Terminal cost.
    """

    dx = x - params["Xref"][-1]
    return 0.5 * dx.T @ params["Qf"] @ dx


def running_cost(params, x, u, t):
    """
    Compute the stage cost for a single time step.

    Args:
        params: Dictionary containing parameters such as Q, R, and reference trajectories.
        x: Current state vector.
        u: Current control vector.
        t: Time step index.
    Returns:
        float: Stage cost.
    """

    dx = x - params["Xref"][t]
    du = u - params["Uref"][t]
    return 0.5 * dx.T @ params["Q"] @ dx + 0.5 * du.T @ params["R"] @ du


def forward_pass(params, X, U, mu, mux, lambd, gains, iter, system_module):
    """
    Perform the forward pass.

    Args:
        params: Dictionary of parameters, including N and reg.
        X: Current state trajectory.
        U: Current control trajectory.
        mu, mux, lambd: Augmented Lagrangian parameters.
        gains: List of feedback gains and feedforward terms.
        iter: Current iteration number.
        system_module: Module containing the system dynamics and constraints.

    Returns:
        Updated state and control trajectories (X, U), step size (alpha), and new cost.
    """

    N = params['N']
    max_linesearch_iters = params['max_linesearch_iters']

    Xn = [X[t].copy() for t in range(N)]
    Un = [U[t].copy() for t in range(N - 1)]
    alpha = 1.0
    forward_pass_success = False
    old_cost = None
    new_cost = None

    
    # Perform line search to find the optimal step size
    for i in range (max_linesearch_iters):

        # Compute old cost
        old_cost = compute_total_cost(params, X, U, mu, mux, lambd, system_module)
       
        # Update control trajectories using the feedback gains and feedforward terms from the backward pass
        # Then update state trajectories using the updated control trajectories
        for t in range(N - 1):
            Un[t] = U[t] - gains[t][0] @ (Xn[t] - X[t]) - alpha * gains[t][1] 
            Xn[t + 1] = system_module.discrete_dynamics(params, Xn[t], Un[t], t)

        # Compute new cost
        new_cost = compute_total_cost(params, Xn, Un, mu, mux, lambd, system_module) 
       
        # If the cost has decreased, the forward pass was successful
        if new_cost < old_cost:
            X = copy.deepcopy(Xn)
            U = copy.deepcopy(Un)
            return X, U, alpha, new_cost
        
        # If the cost has not decreased, reduce the step size and try again
        else:
            alpha = alpha * 0.5

    # If the cost has not decreased after the line search, increase the regularization parameter
    logging.warning("Forward pass failed to reduce cost after line search, increasing reg")
    alpha = 0
    return X, U, alpha, old_cost


def backward_pass(params, X, U, mu, mux, lambd, system_module):

    """
    Perform the backward pass.

    Args:
        params: Dictionary of parameters, including N and reg.
        X: Current state trajectory.
        U: Current control trajectory.
        mu, mux, lambd: Augmented Lagrangian parameters.
        system_module: Module containing the system dynamics and constraints.

    Returns:
        List of feedback gains and feedforward terms (gains) and change in cost (delta_J).
    """
    
    N = params['N']
    delta_J = 0

    gains = []
    V_x = np.zeros((N, params['nx']))
    V_xx = np.zeros((N, params['nx'], params['nx']))

    for t in range(N - 1, -1, -1): # Iterate backwards in time from N - 1 to 0 

        # Compute terminal cost
        if t == N - 1:

            l_x = copy.deepcopy(params['Qf']) @ (X[t] - params['Xref'][t])
            l_xx = copy.deepcopy(params['Qf'])
            V_x[t] = l_x
            V_xx[t] = l_xx

            # add AL term for the state constraints to the terminal cost
            hx = system_module.inequality_constraints_x(params, X[-1])
            mask = eval_mask(mux[-1], hx)
            grad_hx = system_module.inequality_constraints_x_grad(params, X[-1])
            V_x[t] += grad_hx.T @ (mux[-1] + params['rho'] * (mask @ hx))
            V_xx[t] += params['rho'] * grad_hx.T @ mask @ grad_hx

            # add AL term for the goal constraints to the terminal cost
            hx = X[-1] - params['Xref'][-1]
            grad_hx = np.eye(params['nx'])
            V_x[t] += grad_hx.T @ (lambd + params['rho'] * hx)

            V_xx[t] += params['rho'] * grad_hx.T @ grad_hx

        # Compute running cost
        else:

            # Compute Jacobians A and B for dynamics
            A_t = compute_jacobian(lambda x_: system_module.discrete_dynamics(params, x_, U[t], t), X[t])
            B_t = compute_jacobian(lambda u_: system_module.discrete_dynamics(params, X[t], u_, t), U[t])
            
            l_x = copy.deepcopy(params['Q']) @ (X[t] - params['Xref'][t])
            l_u = copy.deepcopy(params['R']) @ (U[t] - params['Uref'][t])
            l_xx = copy.deepcopy(params['Q'])
            l_uu = copy.deepcopy(params['R'])

            # add AL term for the control constraints
            hu = system_module.inequality_constraints_u(params, U[t])
            mask = eval_mask(mu[t], hu)
            grad_hu = system_module.inequality_constraints_u_grad(params, U[t])
            l_u += grad_hu.T @ (mu[t] + params['rho'] * (mask @ hu))
            l_uu += params['rho'] * grad_hu.T @ mask @ grad_hu

       
            # add AL term for the state constraints
            hx = system_module.inequality_constraints_x(params, X[t])
            mask = eval_mask(mux[t], hx)
            grad_hx = system_module.inequality_constraints_x_grad(params, X[t])
            l_x += grad_hx.T @ (mux[t] + params['rho'] * (mask @ hx))
            l_xx += params['rho'] * grad_hx.T @ mask @ grad_hx

            Qx = l_x + A_t.T @ V_x[t+1]
            Qu = l_u + B_t.T @ V_x[t+1]
            
            Qxx = l_xx + A_t.T @ (V_xx[t+1] + params['reg'] * np.eye(params['nx'])) @ A_t
            Quu = l_uu + B_t.T @ (V_xx[t+1] + params['reg'] * np.eye(params['nx'])) @ B_t
            Qux = B_t.T @ (V_xx[t+1] + params['reg'] * np.eye(params['nx'])) @ A_t
            

            Quu_cho = cho_factor(Quu)

            k_t = cho_solve(Quu_cho, Qu)
            K_t = cho_solve(Quu_cho, Qux)
    

            # Cost-to-go Recurrence (PSD stabilizing version)
            V_xx[t] = l_xx + K_t.T @ l_uu @ K_t + (A_t - B_t @ K_t).T @ V_xx[t+1] @ (A_t - B_t @ K_t)
            V_x[t] = l_x - K_t.T @ l_u + K_t.T @ l_uu @ k_t + (A_t - B_t @ K_t).T @ (V_x[t+1] - V_xx[t+1] @ B_t @ k_t)
            
            delta_J += Qu.T @ k_t

            gains.insert(0, (K_t, k_t))
        
    return gains, delta_J


def import_system_modules(system_name):
    """
    Dynamically imports the appropriate modules based on the system name using importlib.
    
    Args:
        system (str): Name of the system (e.g. 'piano_mover').
    
    Returns:
        module: The imported module with the required functions.
    """
    if system_name == "piano_mover":
        module_name = "systems.piano_mover"
    elif system_name == "quadrotor":
        module_name = "systems.cluttered_hallway_quadrotor"
    elif system_name == "coneThroughWall":
        module_name = "systems.cone_through_wall"
    else:
        raise ValueError(f"System '{system_name}' is not recognized. Must be 'piano_mover', 'quadrotor' or 'coneThroughWall'.")

    # Dynamically import the module
    module = importlib.import_module(module_name)
    return module


def ALTRO(params, X, U):
    """
    Perform the iLQR optimization with augmented Lagrangian.

    Args:
        params: Dictionary containing parameters such as N, nx, nu, ncx, ncu, and max_iters.
        X: Initial state trajectory.
        U: Initial control trajectory.

    Returns:
        Updated state and control trajectories (X, U).
    """

    N = params['N']  
    nx = params['nx']
    nu = params['nu']
    ncx = params['ncx']
    ncu = params['ncu']
    max_iters = params['max_iters']

    # Import the appropriate module
    system_module = import_system_modules(params["system"])


    assert len(X) == N, "Length of X does not match params.N"
    assert len(U) == N - 1, "Length of U does not match params.N-1"
    assert len(X[0]) == nx, "Length of X[0] does not match params.nx"
    assert len(U[0]) == nu, "Length of U[0] does not match params.nu"
    assert len(system_module.inequality_constraints_u(params, U[0])) == ncu, "Length of ineq_con_u(params, U[0]) does not match params.ncu"
    assert len(system_module.inequality_constraints_x(params, X[0])) == ncx, "Length of ineq_con_x(params, X[0]) does not match params.ncx"

    plot_trajectories(params['X_hist'], params['U_hist'], params, -1)

    # Forward pass with initial guess (initial roullout)
    for i in range(N - 1):
        X[i + 1] = system_module.discrete_dynamics(params, X[i], U[i], i)

    params['X_hist'].append(X)
    params['U_hist'].append(U)
    plot_trajectories(params['X_hist'], params['U_hist'], params, 0)

    cost_list = []
    reg_list = []

    # Initialize dual variables
    mu = [np.zeros(ncu) for _ in range(N - 1)]
    mux = [np.zeros(ncx) for _ in range(N)]
    lambd = np.zeros(nx)    

    # Main loop
    for itr in range(max_iters):

        gains, delta_J = backward_pass(params, X, U, mu, mux, lambd, system_module)
        X, U, alpha, J = forward_pass(params, X, U, mu, mux, lambd, gains, itr, system_module)
        params['X_hist'].append(X)
        params['U_hist'].append(U)
        cost_list.append(J)


        if itr % 10 == 0 or itr == max_iters - 1:
            plot_trajectories(params['X_hist'], params['U_hist'], params, itr)

        params['reg'] = update_reg(params, alpha)
        reg_list.append(params['reg'])
        
        
        k_list = []
        for gain in gains:
            k_list.append(gain[1])

        kmax = calc_max_k(k_list)

        if itr % 50 == 0:
            print("iter     J           ΔJ        |d|         α        reg         ρ")
            print("---------------------------------------------------------------------")
        print(f"{itr+1:3d}   {J:10.3e}  {delta_J:9.2e}  {kmax:9.2e}  {alpha:6.4f}   {params['reg']:9.2e}   {params['rho']:9.2e}")


        # If the forward pass was successful and the control gain is below the tolerance
        if alpha > 0 and kmax < params['atol']: 
   
            convio = 0 # constraint violation

            # Update control constraints dual variables and compute control constraint violation
            for t in range(N - 1):
                hu = system_module.inequality_constraints_u(params, U[t])
                mask = eval_mask(mu[t], hu)
                mu[t] = np.maximum(0, mu[t] + params['rho'] * (mask @ hu))
                convio = max(convio, np.linalg.norm(hu + abs(hu), ord=np.inf))
            

            # Update state constraints dual variables and compute state constraint violation
            for t in range(N):
                hx = system_module.inequality_constraints_x(params, X[t])
                mask = eval_mask(mux[t], hx)
                mux[t] = np.maximum(0, mux[t] + params['rho'] * (mask @ hx))
                convio = max(convio, np.linalg.norm(hx + abs(hx), ord=np.inf))


            # Update goal constraints dual variables and compute goal constraint violation
            hx = X[-1] - params['Xref'][-1]
            lambd += params['rho'] * hx
            convio = max(convio, np.linalg.norm(hx, ord=np.inf))

            # If the constraint violation is below the tolerance, the optimization has converged
            if convio < params['convio_tol']:
                logging.info(f"Convergence reached in {itr} iterations.")
                plot_trajectories(params['X_hist'], params['U_hist'], params, itr)
                plot_cost(params, cost_list)
                plot_regularization(params, reg_list)
                return X, U
            
            # If the constraint violation is not below the tolerance, increase the penalty parameter
            # to enforce stricter constraint satisfaction in the next iteration
            logging.info(f"convio: {convio}, increasing penalty parameter")
            phi = params['phi']
            params['rho'] *= phi 

        
    logging.info("iLQR optimization complete without convergence")
    plot_trajectories(params['X_hist'], params['U_hist'], params, itr)
    plot_cost(params, cost_list)
    plot_regularization(params, reg_list)
    return X, U
