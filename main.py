import argparse
import numpy as np
from ALTRO import ALTRO
from systems.piano_mover import initialize_piano_mover
from systems.cluttered_hallway_quadrotor import initialize_quadrotor
from systems.cone_through_wall import initialize_coneThroughWall
from utils.visualize_scene_piano_mover import visualize_scene_piano_mover
from utils.visualize_scene_quadrotor_and_cone import visualize_scene_quadrotor_and_cone


"""

System Simulation
=============================

Usage:
------
Run the script with the desired system name using the --system argument:

    python main.py --system <system_name>

"""

def main():
    """
    Main script for setting up the simulation, running ALTRO, and visualizing the results.
    """

    parser = argparse.ArgumentParser(description="Simulate different systems.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Specify the system to simulate (e.g., 'piano_mover')."
    )
    args = parser.parse_args()

    #np.set_printoptions(precision=3, suppress=True) # Set print options for better readability

    if args.system == "piano_mover":
        params, X, U = initialize_piano_mover()
    elif args.system == "quadrotor":
        params, X, U = initialize_quadrotor()
    elif args.system == "coneThroughWall":
        params, X, U = initialize_coneThroughWall()
    elif args.system == "unicycle":
        params, X, U = initialize_unicycle()
    else:
        raise ValueError (f"System '{args.system}' not recognized.")

    print("Starting ALTRO optimization...")
    Xn, Un = ALTRO(params, X, U)
    print("ALTRO optimization complete.")
    
    if args.system == "piano_mover":
        visualize_scene_piano_mover(params, Xn)
    elif args.system == "quadrotor":
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="side_az_90")
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="top_down")
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="custom")
    elif args.system == "coneThroughWall":
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="side_az_90")
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="top_down")
        visualize_scene_quadrotor_and_cone(params, Xn, view_mode="custom")
    elif args.system == "unicycle":
        visualize_scene_unicycle(params, Xn)


if __name__ == "__main__":
    main()