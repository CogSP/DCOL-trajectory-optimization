from primitives.problem_matrices import problem_matrices
from primitives.combine_problem_matrices import combine_problem_matrices
from proximity.pdip import solve_lp_pdip
import numpy as np
from scipy.optimize import approx_fprime


def lag_con_part(capsule, cone, x, s, z, r1, p1, r2, p2, idx_ort, idx_soc1, idx_soc2):
    """
    Computes the lagrangian constraint term for the SOCP problem.

    Args:
        capsule: The first primitive object.
        cone: The second primitive object.
        x: State vector (numpy array).
        s: Slack variable vector (numpy array).
        z: Dual variable vector (numpy array).
        r1: Position vector for capsule (numpy array of size 3).
        p1: Orientation vector for capsule (numpy array of size 3).
        r2: Position vector for cone (numpy array of size 3).
        p2: Orientation vector for cone (numpy array of size 3).
        idx_ort: Indices for orthogonal constraints.
        idx_soc1: Indices for SOC1 constraints.
        idx_soc2: Indices for SOC2 constraints.

    Returns:
        The computed value of z.T @ (G @ x - h).
    """

    # Compute problem matrices for capsule and cone
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule, r1, p1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone, r2, p2)
    
    # Ensure G_ort matrices are 2D
    if len(G_ort1.shape) == 1:
        G_ort1 = G_ort1.reshape(1, -1)
    if len(G_ort2.shape) == 1:
        G_ort2 = G_ort2.reshape(1, -1)

    # Combine problem matrices
    c, G, h, _, _, _ = combine_problem_matrices(
        G_ort1, h_ort1, G_soc1, h_soc1,
        G_ort2, h_ort2, G_soc2, h_soc2
    )
    
    # Compute and return the result
    return np.dot(z.T, np.dot(G, x) - h)


def obj_val_grad(capsule, cone, x, s, z, idx_ort, idx_soc1, idx_soc2):
    """
    Compute the gradient of the objective value with respect to the positions and orientations.

    Args:
        capsule: The first primitive object with attributes `r` (position) and `p` (orientation).
        cone: The second primitive object with attributes `r` (position) and `p` (orientation).
        x: State vector (numpy array).
        s: Slack variable vector (numpy array).
        z: Dual variable vector (numpy array).
        idx_ort: Indices for orthogonal constraints.
        idx_soc1: Indices for SOC1 constraints.
        idx_soc2: Indices for SOC2 constraints.

    Returns:
        gradient: The gradient of the objective value with respect to the state.
    """

    # Define indices for subcomponents
    idx_x = np.arange(len(x))  # Indices for x
    idx_z = np.arange(len(x), len(x) + len(z))  # Indices for z
    idx_r1 = np.arange(3)  # Indices for capsule.r
    idx_p1 = np.arange(3, 6)  # Indices for capsule.p
    idx_r2 = np.arange(6, 9)  # Indices for cone.r
    idx_p2 = np.arange(9, 12)  # Indices for cone.p

    # Combine capsule and cone state vectors
    theta = np.concatenate([capsule.r, capsule.p, cone.r, cone.p])

    # Define the function for gradient computation
    def lag_con_part_wrapper(theta):
        r1, p1, r2, p2 = theta[idx_r1], theta[idx_p1], theta[idx_r2], theta[idx_p2]
        return lag_con_part(capsule, cone, x, s, z, r1, p1, r2, p2, idx_ort, idx_soc1, idx_soc2)

    # Compute the gradient using finite differences
    epsilon = np.sqrt(np.finfo(float).eps)  # Machine epsilon for finite differences
    gradient = approx_fprime(theta, lag_con_part_wrapper, epsilon)

    return gradient


def proximity_gradient(prim1, prim2, pdip_tol=1e-6, verbose=False):
    """
    Compute the proximity gradient between two primitives.

    Args:
        prim1: The first primitive object (must support methods for MRP-specific operations).
        prim2: The second primitive object (must support methods for MRP-specific operations).
        pdip_tol: Tolerance for the primal-dual interior point solver.
        verbose: Whether to enable verbose logging for debugging.

    Returns:
        alpha: Scalar proximity value.
        d_alpha_d_state: Gradient of the proximity value with respect to the state.
    """

    # MRP-specific problem matrices
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1, prim1.r, prim1.p)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2, prim2.r, prim2.p)

    # Ensure G_ort matrices are 2D
    if len(G_ort1.shape) == 1:
        G_ort1 = G_ort1.reshape(1, -1)
    if len(G_ort2.shape) == 1:
        G_ort2 = G_ort2.reshape(1, -1)

    

    # Combine problem matrices and solve the SOCP
    c, G, h, idx_ort, idx_soc1, idx_soc2 = combine_problem_matrices(
        G_ort1, h_ort1, G_soc1, h_soc1,
        G_ort2, h_ort2, G_soc2, h_soc2
    )

    # print in a pretty way the results of combine_problem_matrices

    x, s, z = solve_lp_pdip(c, G, h, idx_ort, idx_soc1, idx_soc2, pdip_tol=pdip_tol)

    # print("x: ", x)
    # print("s: ", s)
    # print("z: ", z)

    # Extract proximity value
    alpha = x[3]

    # Compute the gradient of the objective value
    d_alpha_d_state = obj_val_grad(prim1, prim2, x, s, z, idx_ort, idx_soc1, idx_soc2)

    return alpha, d_alpha_d_state
