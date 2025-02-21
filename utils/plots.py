import logging
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_cost(params, new_cost_list):
    """
    Plot and save the cost during iLQR optimization.

    Args:
        params: Dictionary containing parameters of the problem.
        new_cost_list: List of costs recorded during the optimization process.
    """
    dirs = ["result_images", f"result_images/{params['system']}/costs"] 
    # Create directories if they don't exist 
    for directory in dirs: 
        if not os.path.exists(directory): 
            os.makedirs(directory)

    plt.figure(figsize=(8, 6))
    plt.plot(new_cost_list, label='Cost', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost During iLQR')
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/costs/cost.png")
    plt.close()


def plot_trajectories(X_hist, U_hist, params, iter):
    """
    Plot and save the state and control trajectories over iterations.
    For the state trajectories, four separate graphs are produced:
      - Position
      - Linear Velocity
      - Orientation
      - Angular Velocity

    For the control trajectories:
      - For piano_mover, the controls are split into two graphs:
          one for linear acceleration (first two elements) and one for angular acceleration (third element).
      - For coneThroughWall, the controls are split into two graphs:
          one for forces (first three elements) and one for torques (last three elements).
      - Otherwise, controls are plotted together.

    Args:
        X_hist: List of state trajectories at each iteration.
        U_hist: List of control trajectories at each iteration.
        params: Dictionary of problem parameters.
        iter: Current iteration number.
    """
    N = params['N']
    nx = params['nx']
    nu = params['nu']
    dt = params['dt']

    # Create time vectors
    time_vector   = np.linspace(0, (N - 1) * dt, N)   # For state trajectories
    time_vector_u = np.linspace(0, (N - 2) * dt, N - 1)  # For control trajectories

    # Create directories for saving plots
    dirs = [
        "result_images", 
        f"result_images/{params['system']}/state_trajectories_history", 
        f"result_images/{params['system']}/control_trajectories_history"
    ]
    for directory in dirs: 
        if not os.path.exists(directory): 
            os.makedirs(directory)

    system = params['system']
    # Determine the indices and labels for state components
    if system == 'piano_mover':
        # State is [x, y, vx, vy, theta, omega]
        pos_indices    = [0, 1]
        pos_labels     = [r"$x$", r"$y$"]

        vel_indices    = [2, 3]
        vel_labels     = [r"$v_x$", r"$v_y$"]

        orient_indices = [4]
        orient_labels  = [r"$\theta$"]

        angvel_indices = [5]
        angvel_labels  = [r"$\omega$"]
    elif system in ['quadrotor', 'coneThroughWall']:
        # State is [x, y, z, vx, vy, vz, phi, theta, psi, omega_x, omega_y, omega_z]
        pos_indices    = [0, 1, 2]
        pos_labels     = [r"$x$", r"$y$", r"$z$"]

        vel_indices    = [3, 4, 5]
        vel_labels     = [r"$v_x$", r"$v_y$", r"$v_z$"]

        orient_indices = [6, 7, 8]
        orient_labels  = [r"$\phi$", r"$\theta$", r"$\psi$"]

        angvel_indices = [9, 10, 11]
        angvel_labels  = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
    else:
        # Default (similar to piano_mover)
        pos_indices    = [0, 1]
        pos_labels     = [r"$x$", r"$y$"]

        vel_indices    = [2, 3]
        vel_labels     = [r"$v_x$", r"$v_y$"]

        orient_indices = [4]
        orient_labels  = [r"$\theta$"]

        angvel_indices = [5]
        angvel_labels  = [r"$\omega$"]

    # Retrieve the latest state trajectory from history
    X_latest = X_hist[-1]

    # --- Plot Position ---
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(pos_indices):
        plt.plot(time_vector, [X_latest[t][idx] for t in range(N)], label=pos_labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")  # Updated with units: m
    plt.title("Position Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{system}/state_trajectories_history/position_iter_{iter}.png")
    plt.close()

    # --- Plot Linear Velocity ---
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(vel_indices):
        plt.plot(time_vector, [X_latest[t][idx] for t in range(N)], label=vel_labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Linear Velocity [m/s]")  # Updated with units: m/s
    plt.title("Linear Velocity Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{system}/state_trajectories_history/velocity_iter_{iter}.png")
    plt.close()

    # --- Plot Orientation ---
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(orient_indices):
        plt.plot(time_vector, [X_latest[t][idx] for t in range(N)], label=orient_labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Orientation [rad]")  # Updated with units: rad
    plt.title("Orientation Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{system}/state_trajectories_history/orientation_iter_{iter}.png")
    plt.close()

    # --- Plot Angular Velocity ---
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(angvel_indices):
        plt.plot(time_vector, [X_latest[t][idx] for t in range(N)], label=angvel_labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")  # Updated with units: rad/s
    plt.title("Angular Velocity Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{system}/state_trajectories_history/angular_velocity_iter_{iter}.png")
    plt.close()

    if system == 'piano_mover':
        # Plot linear acceleration (first two components)
        control_labels = [r"$a_{v_x}$", r"$a_{v_y}$", r"$a_{\omega}$"]
        plt.figure(figsize=(12, 6))
        for i in range(2):
            plt.plot(time_vector_u, [U_hist[-1][t][i] for t in range(N - 1)], label=control_labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Linear Acceleration [m/s²]")  # Updated with units: m/s²
        plt.title("Linear Acceleration Trajectories")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_images/{system}/control_trajectories_history/linear_acceleration_iter_{iter}.png")
        plt.close()
        
        # Plot angular acceleration (third component)
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector_u, [U_hist[-1][t][2] for t in range(N - 1)], label=control_labels[2])
        plt.xlabel("Time [s]")
        plt.ylabel("Angular Acceleration [deg/s²]")  # Updated with units: deg/s²
        plt.title("Angular Acceleration Trajectories")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_images/{system}/control_trajectories_history/angular_acceleration_iter_{iter}.png")
        plt.close()
    elif system == "coneThroughWall":
        # Plot forces (first three components)
        control_labels = [r"$f_1$", r"$f_2$", r"$f_3$", r"$\tau_1$", r"$\tau_2$", r"$\tau_3$"]
        plt.figure(figsize=(12, 6))
        for i in range(3):
            plt.plot(time_vector_u, [U_hist[-1][t][i] for t in range(N - 1)], label=control_labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Forces [N]")  # Updated with units: N
        plt.title("Force Trajectories")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_images/{system}/control_trajectories_history/forces_iter_{iter}.png")
        plt.close()
        
        # Plot torques (last three components)
        plt.figure(figsize=(12, 6))
        for i in range(3, 6):
            plt.plot(time_vector_u, [U_hist[-1][t][i] for t in range(N - 1)], label=control_labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Torques [N·m]")  # Updated with units: N·m
        plt.title("Torque Trajectories")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_images/{system}/control_trajectories_history/torques_iter_{iter}.png")
        plt.close()
    else:
        # For quadrotor (and any other system), plot all controls together.
        if system == 'quadrotor':
            control_labels = [r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$"]
        else:
            control_labels = [f"Control {i+1}" for i in range(nu)]
        plt.figure(figsize=(12, 6))
        for i in range(nu):
            plt.plot(time_vector_u, [U_hist[-1][t][i] for t in range(N - 1)], label=control_labels[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Rotor Angular Velocity [rad/s]")
        plt.title("Control Trajectories")
        plt.legend()
        plt.grid()
        plt.savefig(f"result_images/{system}/control_trajectories_history/control_trajectories_iter_{iter}.png")
        plt.close()


def plot_constraint_violations(params, hx_hist, hu_hist, iter):
    """
    Plot and save the constraint violations for states and controls over iterations.

    Args:
        params: Dictionary of problem parameters.
        hx_hist: List of state constraint violations at each iteration.
        hu_hist: List of control constraint violations at each iteration.
        iter: Current iteration number.
    """
    output_dir = f"result_images/{params['system']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 6))
    for i in range(len(hx_hist[0])):
        plt.plot([hx[i] for hx in hx_hist], label=f"State Constraint {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Constraint Violation")
    plt.title(f"State Constraint Violations Over Iterations (Iteration {iter})")
    plt.legend()
    plt.grid()
    plt.savefig(f"result_images/{params['system']}/state_constraints_iter_{iter}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for i in range(len(hu_hist[0])):
        plt.plot([hu[i] for hu in hu_hist], label=f"Control Constraint {i + 1}")
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
