import numpy as np


class PolygonMRP:
    """
    Represents a polygon using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, A, b, radius):
        self.r = np.array([0, 0, 0.0])
        self.p = np.array([0, 0, 0.0])
        self.A = A  
        self.b = b
        self.R = radius
        self.r_offset = np.array([0, 0, 0.0])
        self.Q_offset = np.eye(3)

class ConeMRP:
    """
    Represents a cone using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, height, beta):
        self.r = np.array([0, 0, 0.0])
        self.p = np.array([0, 0, 0.0])
        self.H = height  # Height of the cone
        self.beta = beta  # Angle of the cone
        self.r_offset = np.array([0, 0, 0.0])  # Offset for the position
        self.Q_offset = np.eye(3)  # Identity matrix for Q offset

class CapsuleMRP:
    """
    Represents a capsule using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, radius, height):
        self.r = np.array([0, 0, 0.0])  # Position vector
        self.p = np.array([0, 0, 0.0])
        self.R = radius  # Radius of the capsule
        self.L = height  # Length of the capsule, excluding the hemispheres
        self.r_offset = np.array([0, 0, 0.0])  # Offset for the position
        self.Q_offset = np.eye(3)  # Identity matrix for Q offset

class CylinderMRP:
    """
    Represents a cylinder using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, radius, height):
        self.r = np.array([0, 0, 0.0])  # Position vector
        self.p = np.array([0, 0, 0.0])
        self.R = radius  # Radius of the cylinder
        self.L = height  # Length of the cylinder
        self.r_offset = np.array([0, 0, 0.0])  # Offset for the position
        self.Q_offset = np.eye(3)  # Identity matrix for Q offset

class SphereMRP:
    """
    Represents a sphere using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, radius):
        self.r = np.array([0, 0, 0.0])  # Position vector
        self.p = np.array([0, 0, 0.0])  # Orientation vector (MRP)
        self.R = radius # Radius of the sphere
        self.r_offset = np.array([0, 0, 0.0])  # Offset for the position
        self.Q_offset = np.eye(3)  # Identity matrix for Q offset


class PolytopeMRP:
    """
    Represents a polytope using Modified Rodrigues Parameters (MRP).
    """
    def __init__(self, A, b, length=0, width=0, height=0):
        """
        Initialize the PolytopeMRP with given constraints.

        Args:
            A (ndarray): Matrix representing the polytope normals.
            b (ndarray): Vector representing the polytope offsets.
            length (float): Length of the rectangular prism.
            width (float): Width of the rectangular prism.
            height (float): Height of the rectangular prism.
        """
        self.r = np.array([0, 0, 0.0])  # Position vector
        self.p = np.array([0, 0, 0.0])  # Orientation vector (MRP)
        self.A = A  # Normal matrix
        self.b = b  # Offset vector
        self.length = length
        self.width = width
        self.height = height
        self.r_offset = np.array([0, 0, 0.0])  # Offset for the position
        self.Q_offset = np.eye(3)  # Identity matrix for Q offset


def create_rect_prism(length=20.0, width=20.0, height=2.0, attitude="MRP"):
    """
    Create a rectangular prism with given dimensions and attitude representation.

    Args:
        length (float): Length of the prism.
        width (float): Width of the prism.
        height (float): Height of the prism.
        attitude (str): Attitude representation ("MRP" or "quat").

    Returns:
        tuple: (Polytope object, mass, inertia matrix)
    """
    # Define normal vectors and center offsets for each face
    normals = [
        np.array([1, 0, 0.0]),
        np.array([0, 1, 0.0]),
        np.array([0, 0, 1.0]),
        np.array([-1, 0, 0.0]),
        np.array([0, -1, 0.0]),
        np.array([0, 0, -1.0]),
    ]
    centers = [
        np.array([length / 2, 0, 0.0]),
        np.array([0, width / 2, 0.0]),
        np.array([0, 0, height / 2]),
        np.array([-length / 2, 0, 0.0]),
        np.array([0, -width / 2, 0.0]),
        np.array([0, 0, -height / 2]),
    ]

    # Construct the A matrix and b vector
    A = np.zeros((6, 3))
    b = np.zeros(6)

    for i in range(6):
        A[i, :] = normals[i]
        b[i] = np.dot(normals[i], centers[i])

    # Compute mass and inertia
    mass = length * width * height
    inertia = (mass / 12) * np.diag([width**2 + height**2, length**2 + height**2, length**2 + width**2])

    # Return the polytope and associated properties
    if attitude == "MRP":
        return PolytopeMRP(A, b, length=length, width=width, height=height)
    elif attitude == "quat":
        # Assuming a hypothetical Polytope class for quaternion-based representation
        # NOTE: this is a placeholder and should be replaced with the actual class in a future implementation
        return Polytope(A, b)
    else:
        raise ValueError("Attitude must be 'MRP' or 'quat'")


def create_n_sided(N, d):
    """
    Creates a 2D polygon with N sides and distance `d` for each side from the origin.

    Args:
        N (int): Number of sides.
        d (float): Distance of each side from the origin.

    Returns:
        dict: A dictionary with A matrix and b vector representing the polygon.
    """
    # Compute normal vectors for polygon sides
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    normals = np.array([[np.cos(theta), np.sin(theta)] for theta in angles])

    # Compute A matrix and b vector
    A = normals
    b = np.full(N, d)

    return {"A": A, "b": b}
