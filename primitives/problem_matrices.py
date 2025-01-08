import numpy as np
from primitives.misc_primitive_constructor import SphereMRP, PolytopeMRP, ConeMRP, PolygonMRP, CylinderMRP, CapsuleMRP

def capsule_problem_matrices(R, L, r, n_Q_b):
    """
    Compute problem matrices for a capsule given its radius, height, position, and orientation.

    Args:
        R: Radius of the capsule.
        L: Height of the capsule.
        r: Position vector (numpy array of shape (3,)).
        n_Q_b: Rotation matrix (numpy array of shape (3, 3)).

    Returns:
        G_ort: Placeholder for orthogonal constraints (empty numpy array).
        h_ort: Placeholder for orthogonal constraint bounds (empty numpy array).
        G_soc: Second-order cone constraint matrix (numpy array of shape (2, 4)).
        h_soc: Second-order cone constraint bounds (numpy array of shape (2,)).
    """
    bx = n_Q_b @ np.array([1, 0, 0])
    G_soc_top = np.array([[0, 0, 0.0, -R, 0]])
    G_soc_bot = np.hstack([
        -np.eye(3),
        np.zeros((3, 1)),
        bx.reshape(-1, 1)
    ])
    G_soc = np.vstack([G_soc_top, G_soc_bot])
    h_soc = np.concatenate([[0], -r])

    # G_ort = np.array([
    #     [0, 0, 0, -L / 2, 1],
    #     [0, 0, 0, -L / 2, -1.0],
    #     [-bx[0], -bx[1], -bx[2], -L / 2, 0],
    #     [bx[0], bx[1], bx[2], -L / 2, 0]
    # ])
    # h_ort = np.array([0, 0, -np.dot(bx, r), np.dot(bx, r)])
    G_ort = np.array([
        [0, 0, 0, -L/2, 1],
        [0, 0, 0, -L/2, -1.0]
    ])

    h_ort = np.array([0, 0.0])
    
    return G_ort, h_ort, G_soc, h_soc


def cylinder_problem_matrices(R, L, r, n_Q_b):
    """
    Compute problem matrices for a cylinder given its radius, height, position, and orientation.

    Args:
        R: Radius of the cylinder.
        L: Height of the cylinder.
        r: Position vector (numpy array of shape (3,)).
        n_Q_b: Rotation matrix (numpy array of shape (3, 3)).

    Returns:
        G_ort: Placeholder for orthogonal constraints (empty numpy array).
        h_ort: Placeholder for orthogonal constraint bounds (empty numpy array).
        G_soc: Second-order cone constraint matrix (numpy array of shape (2, 4)).
        h_soc: Second-order cone constraint bounds (numpy array of shape (2,)).
    """

    bx = n_Q_b @ np.array([1, 0, 0])

    G_soc_top = np.array([[0, 0, 0, -R, 0]])

    G_soc_bot = np.hstack([
        -np.eye(3),
        np.zeros((3, 1)),
        bx.reshape(-1, 1)
    ])

    G_soc = np.vstack([G_soc_top, G_soc_bot])

    h_soc = np.concatenate([[0], -r])

    G_ort = np.array([
        [0, 0, 0, -L / 2, 1],
        [0, 0, 0, -L / 2, -1],
        [-bx[0], -bx[1], -bx[2], -L / 2, 0],
        [bx[0], bx[1], bx[2], -L / 2, 0]
    ])

    h_ort = np.array([0, 0, -np.dot(bx, r), np.dot(bx, r)])
    
    return G_ort, h_ort, G_soc, h_soc
     

def polygon_problem_matrices(A, b, R, r, n_Q_b):
    """
    Compute problem matrices for a polygon given its constraints and transformation.

    Args:
        A: Constraint matrix (numpy array of shape (nh, 3)).
        b: Offset vector (numpy array of shape (nh,)).
        R: Position vector (numpy array of shape (3,)).
        r: Position vector (numpy array of shape (3,)).
        n_Q_b: Rotation matrix (numpy array of shape (3, 3)).

    Returns:
        G_ort: Transformed orthogonal constraint matrix (numpy array of shape (nh, 4)).
        h_ort: Orthogonal constraint bounds (numpy array of shape (nh,)).
        G_soc: Placeholder for second-order cone constraints (empty numpy array).
        h_soc: Placeholder for second-order cone bounds (empty numpy array).
    """

    Q_tilde = n_Q_b[:, :2]
    G_ort = np.hstack([np.zeros((A.shape[0], 3)), -b.reshape(-1, 1), A])
    h_ort = np.zeros(A.shape[0])

    G_soc_top = np.array([[0, 0, 0, -R, 0, 0]])
    G_soc_bot = np.hstack([
        -np.eye(3),
        np.zeros((3, 1)),
        Q_tilde
    ])
    G_soc = np.vstack([G_soc_top, G_soc_bot])
    h_soc = np.concatenate([[0], -r])
    return G_ort, h_ort, G_soc, h_soc
    



def cone_problem_matrices(H, beta, r, n_Q_b):
    """
    Computes problem matrices for a cone.

    Args:
        H (float): Height of the cone.
        β (float): Half-angle at the apex of the cone.
        r (np.array): Position vector of the cone (3D).
        n_Q_b (np.array): Orientation matrix (3x3) representing the rotation from body to global frame.

    Returns:
        tuple: G_ort, h_ort, G_soc, h_soc matrices.
    """
    tan_beta = np.tan(beta)
    E = np.diag([tan_beta, 1, 1.0])
    bx = n_Q_b @ np.array([1, 0, 0])
    EQt = E @ n_Q_b.T
    h_soc = -EQt @ r
    G_soc = np.block([
        [-EQt, -np.array([[tan_beta * 3 * H / 4, 0, 0]]).T]
    ])
    G_ort = np.array([bx[0], bx[1], bx[2], -H / 4])
    h_ort = np.array([np.dot(bx, r)])
    return G_ort, h_ort, G_soc, h_soc


def sphere_problem_matrices(R, r):
    """
    Compute problem matrices for a sphere given its radius and position.

    Args:
        R: Radius of the sphere.
        r: Position vector (numpy array of shape (3,)).

    Returns:
        G_ort: Placeholder for orthogonal constraints (empty numpy array).
        h_ort: Placeholder for orthogonal constraint bounds (empty numpy array).
        G_soc: Second-order cone constraint matrix (numpy array of shape (1, 4)).
        h_soc: Second-order cone constraint bounds (numpy array of shape (1,)).
    """
    # Placeholder for orthogonal constraints
    G_ort = np.empty((0, 4))  # Empty 2D array with 4 columns
    h_ort = np.empty((0,))    # Empty 1D array

    # Second-order cone constraints
    h_soc = np.concatenate([[0], -r])
    G_soc = np.array([
        [0, 0, 0, -R],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0]
    ])

    return G_ort, h_ort, G_soc, h_soc


def polytope_problem_matrices(A, b, r, n_Q_b):
    """
    Compute problem matrices for a polytope given its constraints and transformation.

    Args:
        A: Constraint matrix (numpy array of shape (nh, 3)).
        b: Offset vector (numpy array of shape (nh,)).
        r: Position vector (numpy array of shape (3,)).
        n_Q_b: Rotation matrix (numpy array of shape (3, 3)).

    Returns:
        G_ort: Transformed orthogonal constraint matrix (numpy array of shape (nh, 4)).
        h_ort: Orthogonal constraint bounds (numpy array of shape (nh,)).
        G_soc: Placeholder for second-order cone constraints (empty numpy array).
        h_soc: Placeholder for second-order cone bounds (empty numpy array).
    """
    # Transform the constraint matrix

    AQt = A @ n_Q_b.T

    # Orthogonal constraints
    G_ort = np.hstack((AQt, -b.reshape(-1, 1)))
    h_ort = AQt @ r

    # Placeholders for second-order cone constraints
    G_soc = np.empty((0, 4))  # Empty 2D array with 4 columns
    h_soc = np.empty((0,))    # Empty 1D array

    return G_ort, h_ort, G_soc, h_soc



def dcm_from_mrp(p):
    """
    Compute the Direction Cosine Matrix (DCM) from Modified Rodrigues Parameters (MRP).

    Args:
        p (ndarray): A 3-element array representing the MRP.

    Returns:
        ndarray: A 3x3 Direction Cosine Matrix (DCM).
    """
    # Extract components of the MRP vector
    p1, p2, p3 = p

    # Compute the denominator term (used for normalization)
    den = (p1**2 + p2**2 + p3**2 + 1)**2

    # Compute intermediate variable `a` for simplifying matrix elements
    a = (4 * p1**2 + 4 * p2**2 + 4 * p3**2 - 4)

    # Construct the DCM matrix
    dcm = np.array([
        [
            -((8 * p2**2 + 8 * p3**2) / den - 1) * den, 
            (8 * p1 * p2 + p3 * a), 
            (8 * p1 * p3 - p2 * a)
        ],
        [
            (8 * p1 * p2 - p3 * a), 
            -((8 * p1**2 + 8 * p3**2) / den - 1) * den, 
            (8 * p2 * p3 + p1 * a)
        ],
        [
            (8 * p1 * p3 + p2 * a), 
            (8 * p2 * p3 - p1 * a), 
            -((8 * p1**2 + 8 * p2**2) / den - 1) * den
        ]
    ]) / den

    return dcm



def problem_matrices(shape, r, p):
    """
    Compute problem matrices for the shape using MRP-based orientation.

    Args:
        shape: Object with attributes:
                  - A: Constraint matrix.
                  - b: Constraint vector.
                  - r_offset: Position offset vector.
                  - Q_offset: Orientation offset matrix.
        r: Position vector (numpy array of shape (3,)).
        p: MRP (numpy array of shape (3,)).

    Returns:
        Problem matrices computed using the polytope and MRP.
    """

    if isinstance(shape, PolytopeMRP):
        polytope = shape
        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        # Transform polytope matrices
        return polytope_problem_matrices(
            polytope.A, 
            polytope.b, 
            r + n_Q_b @ polytope.r_offset, 
            n_Q_b @ polytope.Q_offset
        )

    if isinstance(shape, SphereMRP):
        sphere = shape
        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        # Adjust position by adding offset transformed by orientation
        adjusted_r = r + n_Q_b @ sphere.r_offset
        
        # Transform sphere matrices
        return sphere_problem_matrices(
            sphere.R, 
            adjusted_r
        )

    if isinstance(shape, ConeMRP):
        cone = shape
        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        # Adjust position by adding offset transformed by orientation
        adjusted_r = r + n_Q_b @ cone.r_offset
        adjusted_Q_offset = n_Q_b @ cone.Q_offset

        # Transform cone matrices
        return cone_problem_matrices(
            cone.H, 
            cone.beta, 
            adjusted_r, 
            adjusted_Q_offset
        )

    if isinstance(shape, PolygonMRP):
        polygon = shape
        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        # Transform polygon matrices
        return polygon_problem_matrices(
            polygon.A, 
            polygon.b, 
            polygon.R,
            r + n_Q_b @ polygon.r_offset, 
            n_Q_b @ polygon.Q_offset
        )

    if isinstance(shape, CylinderMRP):
        cylinder = shape
        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        # Adjust position by adding offset transformed by orientation
        adjusted_r = r + n_Q_b @ cylinder.r_offset
        adjusted_Q_offset = n_Q_b @ cylinder.Q_offset
        
        # print("adjusted_Q_offset: ", adjusted_Q_offset)
        
        # Transform cylinder matrices
        return cylinder_problem_matrices(
            cylinder.R, 
            cylinder.L, 
            adjusted_r,
            adjusted_Q_offset
        )

    if isinstance(shape, CapsuleMRP):
        
        capsule = shape

        # Compute direction cosine matrix from MRP
        n_Q_b = dcm_from_mrp(p)

        adjusted_r = r + n_Q_b @ capsule.r_offset
        adjusted_Q_offset = n_Q_b @ capsule.Q_offset

        return capsule_problem_matrices(
            capsule.R, 
            capsule.L, 
            adjusted_r,
            adjusted_Q_offset
        )