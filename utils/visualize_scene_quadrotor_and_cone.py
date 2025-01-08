import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import os
import h5py
from primitives.misc_primitive_constructor import (
    SphereMRP,
    PolytopeMRP,
    ConeMRP,
    PolygonMRP,
    CylinderMRP,
    CapsuleMRP,
    create_rect_prism,
    create_n_sided
)


def compute_polytope_vertices(A, b):
    """
    Compute the vertices of a polytope defined by A * x <= b.

    Args:
        A (ndarray): Constraint matrix of size (m, n), where m is the number of constraints and n is the dimension.
        b (ndarray): Constraint vector of size (m,).

    Returns:
        vertices (ndarray): Array of vertices (rows represent vertices, columns represent coordinates).
    """
    from itertools import combinations

    m, n = A.shape
    vertices = []

    # Iterate through all combinations of n constraints (choose n from m constraints)
    for indices in combinations(range(m), n):
        # Subset of constraints
        A_subset = A[list(indices), :]
        b_subset = b[list(indices)]

        try:
            # Solve A_subset * x = b_subset
            vertex = np.linalg.solve(A_subset, b_subset)

            # Check if the solution satisfies all the constraints A * x <= b
            if np.all(np.dot(A, vertex) <= b + 1e-6):  # Add small tolerance for numerical stability
                vertices.append(vertex)
        except np.linalg.LinAlgError:
            # Singular matrix (e.g., constraints don't define a unique intersection)
            continue

    # Remove duplicate vertices
    vertices = np.unique(vertices, axis=0)
    return vertices

def visualize_scene_quadrotor_and_cone(params, X, view_mode="side_az_90"):
    """
    Visualizes 3D objects and trajectories.

    Args:
        X: 3D trajectory as a list or array of shape (N, 3).
        objects: List of primitives
        view_mode: Camera view mode ("side_az_90", etc.).
    """

    def draw_obj(ax, obj, alpha=0.5, robot=False):
        """Draws 3D geometric objects (spheres, cylinders, etc.)."""
        
        # Compute rotation matrix from MRP (if orientation exists)
        # NOTE: in the current implementation, the orientation is always defined
        if hasattr(obj, "p"):
            rot = R.from_mrp(obj.p).as_matrix()  # Rotation matrix
        else:
            rot = np.eye(3)  # No rotation if p is not defined (actually, in our code p is always defined)

        if isinstance(obj, SphereMRP):
            u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
            x = obj.R * np.cos(u) * np.sin(v)
            y = obj.R * np.sin(u) * np.sin(v)
            z = obj.R * np.cos(v)
            if robot:
                # Make the robot sphere more prominent
                ax.plot_surface(
                    x + obj.r[0],
                    y + obj.r[1],
                    z + obj.r[2],
                    color="red",
                    alpha=1.0,
                    edgecolor="none",
                    linewidth=0.5,
                )
            else:
                ax.plot_surface(
                    x + obj.r[0],
                    y + obj.r[1],
                    z + obj.r[2],
                    color="yellow",
                    alpha=alpha,
                    edgecolor="none",
                )
        if isinstance(obj, CylinderMRP):
            z = np.linspace(-obj.L / 2, obj.L / 2, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = obj.R * np.cos(theta_grid)
            y_grid = obj.R * np.sin(theta_grid)
            vertices = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

            # Rotate and translate vertices
            rotated_vertices = vertices @ rot.T
            x_grid, y_grid, z_grid = (
                rotated_vertices[:, 0].reshape(x_grid.shape),
                rotated_vertices[:, 1].reshape(y_grid.shape),
                rotated_vertices[:, 2].reshape(z_grid.shape),
            )
            ax.plot_surface(
                x_grid + obj.r[0],
                y_grid + obj.r[1],
                z_grid + obj.r[2],
                color="blue",
                alpha=alpha,
                edgecolor="none",
            )
        
            # Add the top and bottom caps
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_x = obj.R * np.cos(theta)
            circle_y = obj.R * np.sin(theta)
            bottom_circle = np.array([circle_x, circle_y, np.full_like(circle_x, -obj.L / 2)]).T
            top_circle = np.array([circle_x, circle_y, np.full_like(circle_x, obj.L / 2)]).T

            # Rotate and translate the caps
            bottom_circle_rotated = bottom_circle @ rot.T + np.array(obj.r)
            top_circle_rotated = top_circle @ rot.T + np.array(obj.r)

            # Draw the caps
            bottom_face = Poly3DCollection([bottom_circle_rotated], color="blue", alpha=alpha, edgecolor="none")
            top_face = Poly3DCollection([top_circle_rotated], color="blue", alpha=alpha, edgecolor="none")
            ax.add_collection3d(bottom_face)
            ax.add_collection3d(top_face)

        if isinstance(obj, ConeMRP):
            height = obj.H
            radius = height * np.tan(obj.beta)
            theta = np.linspace(0, 2 * np.pi, 50)
            z = np.linspace(0, height, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = (radius * (1 - z_grid / height)) * np.cos(theta_grid)
            y_grid = (radius * (1 - z_grid / height)) * np.sin(theta_grid)
            vertices = np.array([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

            # Rotate and translate vertices
            rotated_vertices = vertices @ rot.T
            x_grid, y_grid, z_grid = (
                rotated_vertices[:, 0].reshape(x_grid.shape),
                rotated_vertices[:, 1].reshape(y_grid.shape),
                rotated_vertices[:, 2].reshape(z_grid.shape),
            )
            
            
            ax.plot_surface(
                x_grid + obj.r[0],
                y_grid + obj.r[1],
                z_grid + obj.r[2],
                color="Red",
                alpha=alpha,
                edgecolor="none",
            )

        if isinstance(obj, PolytopeMRP):

            if obj.length != 0 and obj.width != 0 and obj.height != 0:
                length = obj.length
                width  = obj.width
                height = obj.height
                center = obj.r
                orientation = obj.p

                vertices = np.array([
                    [-length / 2, -width / 2, -height / 2],
                    [ length / 2, -width / 2, -height / 2],
                    [ length / 2,  width / 2, -height / 2],
                    [-length / 2,  width / 2, -height / 2],
                    [-length / 2, -width / 2,  height / 2],
                    [ length / 2, -width / 2,  height / 2],
                    [ length / 2,  width / 2,  height / 2],
                    [-length / 2,  width / 2,  height / 2],
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

                # Add the cuboid to the 3D plot with transparency
                ax.add_collection3d(
                    Poly3DCollection(
                        faces,
                        facecolors=(0.5, 0.5, 0.5, 0.1),  # Light gray with high transparency (alpha=0.1)
                        edgecolors="black",  # Black edges for visibility
                        linewidths=0.5,      # Thin edge lines
                    )
                )

            else:

                # Compute the vertices
                vertices = compute_polytope_vertices(obj.A, obj.b)

                # Convert MRP to rotation matrix
                rot = R.from_mrp(obj.p).as_matrix()

                # Apply rotation and translation to vertices
                rotated_vertices = vertices @ rot.T
                translated_vertices = rotated_vertices + np.array(obj.r)

                # Create the convex hull
                hull = ConvexHull(translated_vertices)

                # Draw the faces of the polytope
                for simplex in hull.simplices:
                    face = Poly3DCollection([translated_vertices[simplex]], color="orange", alpha=alpha, edgecolor="black")
                    ax.add_collection3d(face)
                                        

        elif isinstance(obj, CapsuleMRP):
            
            height = obj.L
            radius = obj.R
            center = np.array(obj.r)
            orientation = obj.p

            # Cylinder part
            z = np.linspace(-height / 2, height / 2, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid)
            y_grid = radius * np.sin(theta_grid)

            # Combine into vertices
            cylinder_vertices = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

            # Hemisphere caps
            u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi / 2:25j]
            x_cap = radius * np.cos(u) * np.sin(v)
            y_cap = radius * np.sin(u) * np.sin(v)
            z_cap_top = radius * np.cos(v) + height / 2
            z_cap_bottom = -radius * np.cos(v) - height / 2

            # Combine caps into vertices
            top_cap_vertices = np.vstack([x_cap.ravel(), y_cap.ravel(), z_cap_top.ravel()]).T
            bottom_cap_vertices = np.vstack([x_cap.ravel(), y_cap.ravel(), z_cap_bottom.ravel()]).T

            # Apply rotation and translation
            rotation = R.from_mrp(orientation).as_matrix()
            rotated_cylinder = cylinder_vertices @ rotation.T
            rotated_top_cap = top_cap_vertices @ rotation.T
            rotated_bottom_cap = bottom_cap_vertices @ rotation.T

            # Translate vertices
            cylinder_vertices_translated = rotated_cylinder + center
            top_cap_vertices_translated = rotated_top_cap + center
            bottom_cap_vertices_translated = rotated_bottom_cap + center

            # Reshape for plotting
            x_cyl, y_cyl, z_cyl = (
                cylinder_vertices_translated[:, 0].reshape(x_grid.shape),
                cylinder_vertices_translated[:, 1].reshape(y_grid.shape),
                cylinder_vertices_translated[:, 2].reshape(z_grid.shape),
            )
            x_top, y_top, z_top = (
                top_cap_vertices_translated[:, 0].reshape(x_cap.shape),
                top_cap_vertices_translated[:, 1].reshape(y_cap.shape),
                top_cap_vertices_translated[:, 2].reshape(z_cap_top.shape),
            )
            x_bottom, y_bottom, z_bottom = (
                bottom_cap_vertices_translated[:, 0].reshape(x_cap.shape),
                bottom_cap_vertices_translated[:, 1].reshape(y_cap.shape),
                bottom_cap_vertices_translated[:, 2].reshape(z_cap_bottom.shape),
            )

            # Plot cylinder and hemisphere caps
            ax.plot_surface(x_cyl, y_cyl, z_cyl, color="green", alpha=alpha, edgecolor="none")
            ax.plot_surface(x_top, y_top, z_top, color="green", alpha=alpha, edgecolor="none")
            ax.plot_surface(x_bottom, y_bottom, z_bottom, color="green", alpha=alpha, edgecolor="none")


        elif isinstance(obj, PolygonMRP):
            # Compute 2D vertices from constraints A * x <= b
            vertices_2d = compute_polytope_vertices(obj.A, obj.b)  # 2D vertices on the polygon plane

            # Create top and bottom vertices by adding and subtracting the cushion radius
            vertices_top = np.hstack([vertices_2d, np.full((vertices_2d.shape[0], 1), obj.R)])
            vertices_bottom = np.hstack([vertices_2d, np.full((vertices_2d.shape[0], 1), -obj.R)])

            # Combine top and bottom vertices
            vertices_3d = np.vstack([vertices_top, vertices_bottom])

            # Apply rotation and translation
            rotation = R.from_mrp(obj.p).as_matrix()
            rotated_vertices = vertices_3d @ rotation.T
            translated_vertices = rotated_vertices + obj.r

            # Split top and bottom vertices after rotation and translation
            num_vertices = vertices_2d.shape[0]
            translated_top = translated_vertices[:num_vertices]
            translated_bottom = translated_vertices[num_vertices:]

            # Create faces
            faces = []

            # Top face
            faces.append(translated_top)

            # Bottom face (reverse order for proper orientation)
            faces.append(translated_bottom[::-1])

            # Side walls
            for i in range(num_vertices):
                next_i = (i + 1) % num_vertices
                side_face = np.array([
                    translated_top[i],
                    translated_top[next_i],
                    translated_bottom[next_i],
                    translated_bottom[i],
                ])
                faces.append(side_face)

            # Add all faces to the 3D plot
            face_collection = Poly3DCollection(faces, color="purple", alpha=alpha, edgecolor="black")
            ax.add_collection3d(face_collection)


    # Ensure the directory exists 
    output_dir = f"result_images/{params['system']}/scene" 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111, projection="3d")

    objects = params["P_obs"]

    # Draw the objects
    for i in range(len(objects)):
        draw_obj(ax, objects[i])


    # Draw the trajectory as a dashed black line
    X = np.array(X)
    ax.plot(
        X[:, 0], X[:, 1], X[:, 2],
        color="black", linestyle="--", linewidth=2, label="Trajectory"
    )

    # Add robots along the trajectory
    for i in range(0, len(X), max(1, len(X) // 10)):  # Add robots at intervals
        params['P_vic'].r = X[i]  # Update sphere position to current state
        draw_obj(ax, params['P_vic'], alpha=1, robot=True)

    # Add corridor bounding planes
    x_plane = np.linspace(-10, 10, 10)
    y_plane = np.linspace(-10, 10, 10)
    x_grid, y_grid = np.meshgrid(x_plane, y_plane)
    z_top = np.full_like(x_grid, 10)  # Top plane at z=10
    z_bottom = np.full_like(x_grid, -10)  # Bottom plane at z=-10

    ax.plot_surface(x_grid, y_grid, z_top, color="gray", alpha=0.3)
    ax.plot_surface(x_grid, y_grid, z_bottom, color="gray", alpha=0.3)

    # Adjust view
    if view_mode == "side_az_90":
        ax.view_init(elev=0, azim=90)
        # Remove Y-axis for side view
        ax.set_zticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.line.set_visible(False)
        ax.yaxis.label.set_visible(False)
    elif view_mode == "top_down":
        ax.view_init(elev=90, azim=-90)
        # Remove Z-axis for top-down view
        ax.set_zticks([])
        ax.zaxis.set_ticklabels([])
        ax.zaxis.line.set_visible(False)
        ax.zaxis.label.set_visible(False)
    else:
        ax.view_init(elev=15, azim=-60)


    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    if view_mode not in ["side_az_90", "top_down"]:
        ax.set_zlabel("Z-axis")
        plt.tight_layout()

    # Legend and plot
    ax.legend(loc="upper left")
    plt.savefig(f"result_images/{params['system']}/scene/scene_{view_mode}.png")