import numpy as np
from scipy.linalg import cholesky, solve_triangular
from proximity.NT.NT_scaling import calc_NT_scalings, multiply_nt_scaling_vector, multiply_nt_scaling_matrix, solve_nt_scaling_vector, solve_nt_scaling_matrix
from scipy.linalg import cho_factor, cho_solve


def ort_linesearch(x: np.ndarray, dx: np.ndarray) -> float:
    """
    Computes the maximum alpha in [0, 1] such that (x + alpha * dx > 0).

    Args:
        x (np.ndarray): Current state vector for the orthant.
        dx (np.ndarray): Direction vector for the orthant.

    Returns:
        float: Maximum allowable alpha.
    """
    alpha = 1.0
    for i in range(len(x)):
        if dx[i] < 0:
            alpha = min(alpha, -x[i] / dx[i])
    return alpha


def soc_linesearch(y: np.ndarray, delta: np.ndarray) -> float:
    """
    Computes the line search step size for second-order cone (SOC).

    Args:
        y (np.ndarray): Current state vector for the SOC.
        delta (np.ndarray): Direction vector for the SOC.

    Returns:
        float: Maximum allowable alpha.
    """
    v_idx = slice(1, len(y))
    yv = y[v_idx]
    delta_v = delta[v_idx]
    nu = max(y[0]**2 - np.dot(yv, yv), 1e-25)  # Numerical safeguard
    zeta = y[0] * delta[0] - np.dot(yv, delta_v)

    rho = np.concatenate([
        [zeta / nu],
        delta_v / np.sqrt(nu) -
        ((zeta / np.sqrt(nu) + delta[0]) /
         (y[0] / np.sqrt(nu) + 1)) * (yv / nu)
    ])

    if np.linalg.norm(rho[v_idx]) > rho[0]:
        return min(1.0, 1 / (np.linalg.norm(rho[v_idx]) - rho[0]))
    else:
        return 1.0


def linesearch(x: np.ndarray, dx: np.ndarray,
               idx_ort: np.ndarray, idx_soc1: np.ndarray, idx_soc2: np.ndarray) -> float:
    """
    Computes the maximum alpha in [0, 1] for the line search step across orthant and SOCs.

    Args:
        x (np.ndarray): Current state vector.
        dx (np.ndarray): Direction vector.
        idx_ort (np.ndarray): Indices for orthant constraints.
        idx_soc1 (np.ndarray): Indices for the first SOC.
        idx_soc2 (np.ndarray): Indices for the second SOC.

    Returns:
        float: Maximum allowable alpha.
    """
    x_ort = x[idx_ort]
    dx_ort = dx[idx_ort]
    x_soc1 = x[idx_soc1]
    dx_soc1 = dx[idx_soc1]
    x_soc2 = x[idx_soc2]
    dx_soc2 = dx[idx_soc2]

    alpha = 1.0
    if len(idx_ort) > 0:
        alpha = min(alpha, ort_linesearch(x_ort, dx_ort))
    if len(idx_soc1) > 0:
        alpha = min(alpha, soc_linesearch(x_soc1, dx_soc1))
    if len(idx_soc2) > 0:
        alpha = min(alpha, soc_linesearch(x_soc2, dx_soc2))

    return alpha


def inverse_soc_cone_product(u, w):
    """
    Computes the inverse cone product for two vectors within a second-order cone (SOC).
    
    Given two vectors u and w in an SOC, this function computes a new vector v such that
    the cone product of u and v yields w in the SOC. It effectively inverts the SOC
    multiplication operation under certain conditions.

    Args:
        u (np.ndarray): A vector in the second-order cone.
        w (np.ndarray): Another vector in the same second-order cone.

    Returns:
        np.ndarray: The result of the inverse SOC cone product, satisfying the relation
                    soc_cone_product(u, v) ≈ w.
    """
    n = len(u)
    if n > 0:
        u0 = u[0]
        u1 = u[1:]
        w0 = w[0]
        w1 = w[1:]
        # ρ = u0^2 - dot(u1,u1)
        rho = u0**2 - np.dot(u1, u1)
        # ν = dot(u1,w1)
        nu = np.dot(u1, w1)

        # Build the result vector
        scalar_part = u0*w0 - nu
        vector_part = (nu/u0 - w0)*u1 + (rho/u0)*w1

        return (1.0/rho) * np.concatenate(([scalar_part], vector_part))
    else:
        # Return empty if n == 0
        return np.array([], dtype=float)


def inverse_cone_product(lam, v, idx_ort, idx_soc1, idx_soc2):
    """
    Computes the inverse cone product for a composite cone structure consisting of an orthant
    and two second-order cones (SOCs). It applies the inverse operations componentwise to
    each part of the cone.

    Args:
        lam (np.ndarray): Vector in the composite cone.
        v (np.ndarray): Vector in the same composite cone to be 'divided' by lam.
        idx_ort (np.ndarray): Indices corresponding to the orthant part.
        idx_soc1 (np.ndarray): Indices corresponding to the first SOC part.
        idx_soc2 (np.ndarray): Indices corresponding to the second SOC part.

    Returns:
        np.ndarray: The vector resulting from the inverse cone product lam^{-1} ∘ v,
                    computed elementwise for the orthant part and using inverse_soc_cone_product
                    for the SOC parts.
    """

    # Extract sub-vectors for orthant part
    lam_ort = lam[idx_ort]
    v_ort   = v[idx_ort]
    
    # Extract sub-vectors for SOC parts
    lam_soc1 = lam[idx_soc1]
    v_soc1   = v[idx_soc1]
    lam_soc2 = lam[idx_soc2]
    v_soc2   = v[idx_soc2]

    # Elementwise division for the orthant part
    top = v_ort / lam_ort

    # For the SOC parts
    bot1 = inverse_soc_cone_product(lam_soc1, v_soc1)
    bot2 = inverse_soc_cone_product(lam_soc2, v_soc2)

    # Concatenate results into one array
    return np.concatenate([top, bot1, bot2])


def soc_cone_product(u, v):
    """
    Computes the cone product of two vectors u and v within a second-order cone (SOC).
    
    The cone product is defined specifically for SOCs, and this implementation calculates
    a vector that results from this product operation.

    Args:
        u (np.ndarray): A vector in the second-order cone.
        v (np.ndarray): Another vector in the same second-order cone.

    Returns:
        np.ndarray: The result of the SOC cone product between u and v.
                    For nonempty vectors, returns a new vector whose first element is the
                    dot product of u and v, and whose remaining elements are u0*v1 + v0*u1.
    """

    n = len(u)
    if n > 0:

        u0 = u[0]
        u1 = u[1:]
        v0 = v[0]
        v1 = v[1:]

        # scalar part
        scalar_part = np.dot(u, v)

        # vector part
        vector_part = u0 * v1 + v0 * u1

        # Concatenate scalar_part as a 1D array with vector_part
        return np.concatenate(([scalar_part], vector_part))
    else:
        # Return an empty array if there are no elements
        return np.array([], dtype=float)


def gen_e(idx_ort, idx_soc1, idx_soc2):
    """
    Generates a unit direction vector 'e' corresponding to the structure of the composite cone.
    
    For the orthant part, the entries are ones. For each SOC part, the first component
    is one followed by zeros, corresponding to the "central" direction of the cone.

    Args:
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC.
        idx_soc2 (np.ndarray): Indices for the second SOC.

    Returns:
        np.ndarray: A vector of appropriate size whose entries are ones for the orthant 
                    components and ones followed by zeros for each SOC part.
    """
    n_ort  = len(idx_ort)
    n_soc1 = len(idx_soc1)
    n_soc2 = len(idx_soc2)

    # Start e as an array of ones of length n_ort
    # (or an empty array if n_ort == 0)
    e = np.ones(n_ort) if n_ort > 0 else np.array([], dtype=float)

    # If n_soc1 > 0, append a vector: [1.0, 0.0, 0.0, ..., 0.0]
    if n_soc1 > 0:
        e = np.concatenate([e, np.array([1.0] + [0.0] * (n_soc1 - 1))])

    # If n_soc2 > 0, append [1.0, 0.0, ..., 0.0]
    if n_soc2 > 0:
        e = np.concatenate([e, np.array([1.0] + [0.0] * (n_soc2 - 1))])

    return e

def bring2cone(r, idx_ort, idx_soc1, idx_soc2):
    """
    Adjusts a given vector 'r' to ensure it lies within the composite cone defined
    by orthant and second-order cone (SOC) constraints.

    The function checks each part of the vector corresponding to the orthant and SOCs.
    If any component violates the cone's constraints (i.e., non-positive for orthant,
    or not satisfying SOC conditions), it computes an adjustment factor 'alpha' and
    corrects 'r' by moving it along the direction generated by 'gen_e'.

    Args:
        r (np.ndarray): The vector to be adjusted.
        idx_ort (np.ndarray): Indices for orthant constraints.
        idx_soc1 (np.ndarray): Indices for the first SOC.
        idx_soc2 (np.ndarray): Indices for the second SOC.

    Returns:
        np.ndarray: A vector adjusted to lie within the composite cone.
    """

    alpha = -1

    # Extract relevant slices/sub-vectors
    r_ort  = r[idx_ort]
    r_soc1 = r[idx_soc1]
    r_soc2 = r[idx_soc2]

    # Check if any orthant component is <= 0
    if np.any(r_ort <= 0):
        alpha = -np.min(r_ort)

    # For the first SOC:
    if len(idx_soc1) > 0:
        # r_soc1[0] is the "first" element in that cone,
        # r_soc1[1:] are the remaining "vector" components
        res = r_soc1[0] - np.linalg.norm(r_soc1[1:])
        if res <= 0:
            alpha = max(alpha, -res)

    # For the second SOC:
    if len(idx_soc2) > 0:
        res = r_soc2[0] - np.linalg.norm(r_soc2[1:])
        if res <= 0:
            alpha = max(alpha, -res)

    # If alpha < 0, r is already feasible
    if alpha < 0:
        return r
    else:
        # gen_e(...) should generate the unit direction e for these cone parts
        return r + (1 + alpha)*gen_e(idx_ort, idx_soc1, idx_soc2)



def initialize(c, G, h, idx_ort, idx_soc1, idx_soc2):
    """
    Initializes the variables for the Primal-Dual Interior Point Method for LP problems
    with orthant and second-order cone constraints.

    The function computes an initial feasible point (x_hat, s_hat, z_hat) such that:
        - G x_hat - h + s_hat = 0 with s_hat in the cone.
        - G.T z_hat + c = 0 (dual feasibility).
    It uses Cholesky decompositions and solves triangular systems to find these initial points.
    The function also adjusts s_hat and z_hat to ensure they lie within the respective cones.

    Args:
        c (np.ndarray): Coefficient vector of the objective function.
        G (np.ndarray): Constraint matrix.
        h (np.ndarray): Right-hand side vector.
        idx_ort (np.ndarray): Indices for orthant constraints.
        idx_soc1 (np.ndarray): Indices for the first SOC.
        idx_soc2 (np.ndarray): Indices for the second SOC.

    Returns:
        tuple:
            - np.ndarray: Initial primal variable x_hat.
            - np.ndarray: Adjusted slack variable s_hat within the cone.
            - np.ndarray: Adjusted dual slack variable z_hat within the cone.
    """

    F = np.linalg.cholesky(G.T @ G)
    y = solve_triangular(F, G.T @ h, lower=True)
    x_hat = solve_triangular(F.T, y, lower=False)


    s_tilde = G @ x_hat - h
    s_hat = bring2cone(s_tilde, idx_ort, idx_soc1, idx_soc2)


    y_x = solve_triangular(F, -c)
    x = solve_triangular(F.T, y_x)

    z_tilde = G @ x
    z_hat = bring2cone(z_tilde, idx_ort, idx_soc1, idx_soc2)

    return x_hat, s_hat, z_hat


def cone_product(s, z, idx_ort, idx_soc1, idx_soc2):
    """
    Computes the componentwise product (cone product) of two vectors s and z over a composite cone.
    
    For the orthant part, it performs elementwise multiplication.
    For each SOC part, it applies the soc_cone_product function.

    Args:
        s (np.ndarray): First vector belonging to the composite cone.
        z (np.ndarray): Second vector belonging to the composite cone.
        idx_ort (np.ndarray): Indices for orthant components.
        idx_soc1 (np.ndarray): Indices for the first SOC components.
        idx_soc2 (np.ndarray): Indices for the second SOC components.

    Returns:
        np.ndarray: The resulting vector from the cone product of s and z.
    """

    # Extract sub-vectors based on the given indices
    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    s_soc1 = s[idx_soc1]
    z_soc1 = z[idx_soc1]
    s_soc2 = s[idx_soc2]
    z_soc2 = z[idx_soc2]

    # Elementwise product for the orthant part
    product_ort = s_ort * z_ort  # NumPy does elementwise multiplication with *

    # "Cone product" for the SOC parts 
    # (assumes you have a Python version of soc_cone_product)
    product_soc1 = soc_cone_product(s_soc1, z_soc1)
    product_soc2 = soc_cone_product(s_soc2, z_soc2)

    # Concatenate results into one array
    return np.concatenate([product_ort, product_soc1, product_soc2])


def solve_lp_pdip(c, G, h, idx_ort, idx_soc1, idx_soc2, max_iter=20, pdip_tol=1e-5):
    """
    Solve the LP problem using Primal-Dual Interior Point Method.
    min c'x
    s.t. Gx + s = h
         s >= 0

    Args:
        c (np.ndarray): Coefficients of the objective function.
        G (np.ndarray): Coefficients of the constraints.
        h (np.ndarray): Right-hand side of the constraints.
        idx_ort (np.ndarray): Indices for orthant constraints.
        idx_soc1 (np.ndarray): Indices for the first SOC.
        idx_soc2 (np.ndarray): Indices for the second SOC.
        max_iter (int): Maximum number of iterations.
        pdip_tol (float): Tolerance for the primal-dual interior point method.

    Returns:
        np.ndarray: Solution vector x.
        np.ndarray: Solution vector s.
        np.ndarray: Solution vector z.
    """
    
    x, s, z = initialize(c, G, h, idx_ort, idx_soc1, idx_soc2)


    e = gen_e(idx_ort, idx_soc1, idx_soc2)

    cone_degree = len(idx_ort)

    if len(idx_soc1) > 0:
        cone_degree += 1
    if len(idx_soc2) > 0:
        cone_degree += 1

    for i in range(50):
        
        W = calc_NT_scalings(s, z, idx_ort, idx_soc1, idx_soc2)

        lambd = multiply_nt_scaling_vector(W, z, idx_ort, idx_soc1, idx_soc2)
        
        lambd_lambd = cone_product(lambd, lambd, idx_ort, idx_soc1, idx_soc2)

        rx = G.T @ z + c
        rz = s + G @ x - h
        mu = np.dot(s, z) / cone_degree

        if mu < pdip_tol:
            # print("pdip_tol: ", pdip_tol)
            return x, s, z

        bx = -rx
        lambd_ds = inverse_cone_product(lambd, -lambd_lambd, idx_ort, idx_soc1, idx_soc2)

        b_z_tilde = solve_nt_scaling_vector(W, - rz - multiply_nt_scaling_vector(W, lambd_ds, idx_ort, idx_soc1, idx_soc2), idx_ort, idx_soc1, idx_soc2)

        G_tilde = solve_nt_scaling_matrix(W, G, idx_ort, idx_soc1, idx_soc2)


        # Compute F using Cholesky decomposition of G̃.T @ G̃
        #F = cholesky(G_tilde.T @ G_tilde)
        F = cholesky(G_tilde.T @ G_tilde)

        delta_x = cho_solve((F, False), bx + G_tilde.T @ b_z_tilde)

        delta_z = solve_nt_scaling_vector(W, G_tilde @ delta_x - b_z_tilde, idx_ort, idx_soc1, idx_soc2) 

        delta_s = multiply_nt_scaling_vector(W, lambd_ds - multiply_nt_scaling_vector(W, delta_z, idx_ort, idx_soc1, idx_soc2), idx_ort, idx_soc1, idx_soc2)

        # print("delta_s: ", delta_s)
    

        # linesearch on affine step
        alpha = min(linesearch(s, delta_s, idx_ort, idx_soc1, idx_soc2), linesearch(z, delta_z, idx_ort, idx_soc1, idx_soc2))
        rho = np.dot(s + alpha*delta_s, z + alpha*delta_z) / np.dot(s, z)
        sigma = max(0, min(1, rho))**3

        ds = - lambd_lambd - cone_product(solve_nt_scaling_vector(W, delta_s, idx_ort, idx_soc1, idx_soc2), multiply_nt_scaling_vector(W, delta_z, idx_ort, idx_soc1, idx_soc2), idx_ort, idx_soc1, idx_soc2) + sigma * mu * e
        lambd_ds = inverse_cone_product(lambd, ds, idx_ort, idx_soc1, idx_soc2)
        b_z_tilde = solve_nt_scaling_vector(W, -rz - multiply_nt_scaling_vector(W, lambd_ds, idx_ort, idx_soc1, idx_soc2), idx_ort, idx_soc1, idx_soc2)
        
        # y_delta_x = solve_triangular(F, bx + G_tilde.T @ b_z_tilde, lower=False)
        # delta_x = solve_triangular(F.T, y_delta_x, lower=True)

        delta_x = cho_solve((F, False), bx + G_tilde.T @ b_z_tilde)

        delta_z = solve_nt_scaling_vector(W, G_tilde @ delta_x - b_z_tilde, idx_ort, idx_soc1, idx_soc2)
        delta_s = multiply_nt_scaling_vector(W, lambd_ds - multiply_nt_scaling_vector(W, delta_z, idx_ort, idx_soc1, idx_soc2), idx_ort, idx_soc1, idx_soc2)

        alpha = min(1, 0.99*min(linesearch(s, delta_s, idx_ort, idx_soc1, idx_soc2), linesearch(z, delta_z, idx_ort, idx_soc1, idx_soc2)))

        x += alpha * delta_x
        s += alpha * delta_s
        z += alpha * delta_z

        n_ort = len(idx_ort)

    raise Exception("Maximum number of iterations reached, PDIP failed")
