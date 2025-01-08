import logging
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_cost(params, new_cost_list):
    """
    Plot and save the cost during iLQR optimization.

    Args:
        params: Dictionary containing parameters of the problem
        new_cost_list: List of costs recorded during the optimization process.
    """
    dirs = ["result_images", f"result_images/{params['system']}/costs"] 
    
    # Check and create each directory if it doesn't exist 
    for directory in dirs: 
        if not os.path.exists(directory): 
            os.makedirs(directory)

    plt.figure(figsize=(8, 6))
    plt.plot(new_cost_list, label='Cost', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Cost During iLQR')
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/costs/cost.png")
    plt.close()


def plot_trajectories(X_hist, U_hist, params, iter):
    """
    Plot and save the state and control trajectories over iterations.

    Args:
        X_hist: List of state trajectories at each iteration.
        U_hist: List of control trajectories at each iteration.
        params: Dictionary of problem parameters.
        iter: Current iteration number.
    """
    N = params['N']
    nx = params['nx']
    nu = params['nu']

    dirs = ["result_images", f"result_images/{params['system']}/state_trajectories_history", f"result_images/{params['system']}/control_trajectories_history"] 
    
    # Check and create each directory if it doesn't exist 
    for directory in dirs: 
        if not os.path.exists(directory): 
            os.makedirs(directory)

    # State trajectories
    plt.figure(figsize=(12, 6))
    for i in range(nx):
        plt.plot(
            [X_hist[-1][t][i] for t in range(N)],
            label=f"State {i + 1}",
        )
    plt.xlabel("Time Step")
    plt.ylabel("State Values")
    plt.title(f"State Trajectories Over Iterations (Iteration {iter})")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/state_trajectories_history/state_trajectories_iter_{iter}.png")
    plt.close()

    # Control trajectories
    plt.figure(figsize=(12, 6))
    for i in range(nu):
        plt.plot(
            [U_hist[-1][t][i] for t in range(N - 1)],
            label=f"Control {i + 1}",
        )
    plt.xlabel("Time Step")
    plt.ylabel("Control Values")
    plt.title(f"Control Trajectories Over Iterations (Iteration {iter})")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/control_trajectories_history/control_trajectories_iter_{iter}.png")
    plt.close()


# NOTE: this is not used in the current implementation
def plot_constraint_violations(params, hx_hist, hu_hist, iter):
    """
    Plot and save the constraint violations for states and controls over iterations.

    Args:
        params: Dictionary of problem parameters.
        hx_hist: List of state constraint violations at each iteration.
        hu_hist: List of control constraint violations at each iteration.
        iter: Current iteration number.
    """

    # Ensure the directory exists 
    output_dir = f"result_images/params['system']" 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    
    plt.figure(figsize=(12, 6))
    for i in range(len(hx_hist[0])):
        plt.plot(
            [hx[i] for hx in hx_hist],
            label=f"State Constraint {i + 1}",
        )
        
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation")
    plt.title(f"State Constraint Violations Over Iterations (Iteration {iter})")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/state_constraints_iter_{iter}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for i in range(len(hu_hist[0])):
        plt.plot(
            [hu[i] for hu in hu_hist],
            label=f"Control Constraint {i + 1}",
        )
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation")
    plt.title(f"Control Constraint Violations Over Iterations (Iteration {iter})")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/control_constraints_iter_{iter}.png")
    plt.close()
    

def plot_regularization(params, reg_values):
    """
    Plot and save the regularization term over iterations.

    Args:
        params: Dictionary of problem parameters.
        reg_values: List of regularization values recorded during the optimization process.
    """

    dirs = ["result_images", f"result_images/{params['system']}/reg"] 
    
    # Check and create each directory if it doesn't exist 
    for directory in dirs: 
        if not os.path.exists(directory): 
            os.makedirs(directory)
        
    plt.figure(figsize=(8, 6))
    plt.plot(reg_values, marker='o', linestyle='-')
    plt.title("Regularization Term Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Regularization Term (reg)")
    plt.grid(True)
    plt.savefig(f"result_images/{params['system']}/reg/regularization.png")
    plt.close()