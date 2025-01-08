import os
import numpy as np
from primitives.misc_primitive_constructor import create_rect_prism
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from primitives.misc_primitive_constructor import SphereMRP, PolytopeMRP, ConeMRP, PolygonMRP, CylinderMRP, CapsuleMRP


def visualize_scene_piano_mover(params, X, view_mode="none"):
    """
    Visualizes the trajectory of the robot in the environment and saves multiple images.

    Args:
        params: Parameters of the system, including obstacles and robot dimensions.
        X: State trajectory (list of states).
        view_mode: The viewing angle of the 3D plot. Options: "none", "top_down", "side", "side_az_90
    """
    def draw_obj(ax, obj, color="blue", alpha=0.5, edgecolor="black"):
        """
        Draw a geometric object in the 3D plot based on its type.

        Args:
            ax: Matplotlib axis object.
            obj: The object to draw.
            color: Color of the object.
            alpha: Transparency of the object.
            edgecolor: Edge color of the object.
        """
        if isinstance(obj, PolytopeMRP):
            length = obj.length
            width = obj.width
            height = obj.height
            center = obj.r
            orientation = obj.p

            vertices = np.array([
                [-length / 2, -width / 2, -height / 2],
                [length / 2, -width / 2, -height / 2],
                [length / 2, width / 2, -height / 2],
                [-length / 2, width / 2, -height / 2],
                [-length / 2, -width / 2, height / 2],
                [length / 2, -width / 2, height / 2],
                [length / 2, width / 2, height / 2],
                [-length / 2, width / 2, height / 2],
            ])

            rotation = R.from_euler('xyz', orientation, degrees=False)
            rotated_vertices = rotation.apply(vertices)
            translated_vertices = rotated_vertices + center

            faces = [
                [translated_vertices[0], translated_vertices[1], translated_vertices[2], translated_vertices[3]],
                [translated_vertices[4], translated_vertices[5], translated_vertices[6], translated_vertices[7]],
                [translated_vertices[0], translated_vertices[1], translated_vertices[5], translated_vertices[4]],
                [translated_vertices[2], translated_vertices[3], translated_vertices[7], translated_vertices[6]],
                [translated_vertices[0], translated_vertices[3], translated_vertices[7], translated_vertices[4]],
                [translated_vertices[1], translated_vertices[2], translated_vertices[6], translated_vertices[5]],
            ]

            ax.add_collection3d(
                Poly3DCollection(
                    faces, facecolors=color, alpha=alpha, edgecolors=edgecolor
                )
            )

    intervals = [1, 5, 10, 15, 20, 25, 30, 35, 40]  # Intervals for drawing the robot
    for interval in intervals:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        dirs = ["result_images", f"result_images/{params['system']}", f"result_images/{params['system']}/scene"]
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Plot the trajectory line with a dashed black style
        trajectory = np.array([[state[0], state[1], 0] for state in X])
        ax.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            label="Trajectory", color="black", linestyle="--", linewidth=2
        )

        # Draw the robot at the specified interval
        for i in range(0, len(X), interval):
            params['P_vic'].r = np.array([X[i][0], X[i][1], 0])
            params['P_vic'].p = np.array([0, 0, X[i][4]])  # X[4] is the MRP angle
            draw_obj(ax, params['P_vic'], color="red", alpha=0.7)

        # Draw obstacles
        for obs in params['P_obs']:
            draw_obj(ax, obs, color="gray", alpha=0.5)

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-2, 2)

        ax.view_init(elev=90, azim=0)  # Top-down view, the P.O.V. can be modified changing these values

        # Aspect ratio equal
        ax.set_box_aspect([1, 1, 0.5])  # X, Y, Z aspect ratio

        # Hide Z-axis grid and labels for a cleaner look
        ax.grid(False)
        ax.zaxis.set_label_position('none')
        ax.set_zticks([])

        ax.legend(loc='upper left')
        plt.tight_layout()

        # Save the image
        plt.savefig(f"result_images/{params['system']}/scene/scene_interval_{interval}.png")
        plt.close(fig)
