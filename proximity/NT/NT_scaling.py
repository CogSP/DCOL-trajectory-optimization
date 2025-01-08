import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Optional, Tuple
from scipy.linalg import solve_triangular, cholesky
from scipy.linalg import cho_factor, cho_solve

def soc_quad_J(x):
    """
    Computes xs^2 - dot(xv, xv) for the vector x,
    where xs = x[0] and xv = x[1:].
    
    This is the quadratic form associated with a second-order cone.
    """
    xs = x[0]
    xv = x[1:]
    return xs**2 - np.dot(xv, xv)

def normalize_soc(x):
    """
    Normalizes a vector x with respect to the second-order cone quadratic form.
    """

    return x / np.sqrt(soc_quad_J(x))


class NTScaling2:
    """
    Data structure to store Nesterov-Todd (NT) scaling matrices and factors
    for a composite cone consisting of an orthant and two second-order cones (SOCs).
    
    Attributes:
        ort (NDArray[np.float64]): 1D array containing scaling factors for the orthant part.
        soc1 (NDArray[np.float64]): 2D array representing the NT scaling matrix for the first SOC.
        soc1_fact (Tuple[NDArray[np.float64], bool]): Cholesky factorization of soc1.
        soc2 (NDArray[np.float64]): 2D array representing the NT scaling matrix for the second SOC.
        soc2_fact (Tuple[NDArray[np.float64], bool]): Cholesky factorization of soc2.
    """

    def __init__(
        self,
        ort:  NDArray[np.float64],
        soc1: NDArray[np.float64],
        soc1_fact: Tuple[NDArray[np.float64], bool],
        soc2: NDArray[np.float64],
        soc2_fact: Tuple[NDArray[np.float64], bool]
    ):
        self.ort       = ort       # 1D array of length n_ort
        self.soc1      = soc1      # 2D array of shape (n_soc1, n_soc1)
        self.soc1_fact = soc1_fact # Cholesky factor for soc1
        self.soc2      = soc2 
        self.soc2_fact = soc2_fact

class Scaling2:
    """
    Data structure to store scaling matrices for a composite cone consisting
    of an orthant and two second-order cones (SOCs) without NT-specific factors.
    
    Attributes:
        ort (NDArray[np.float64]): 1D array containing scaling factors for the orthant part.
        soc1 (NDArray[np.float64]): 2D array representing the scaling matrix for the first SOC.
        soc2 (NDArray[np.float64]): 2D array representing the scaling matrix for the second SOC.
    """

    def __init__(
        self,
        ort:  NDArray[np.float64],
        soc1: NDArray[np.float64],
        soc2: NDArray[np.float64]
    ):
        self.ort  = ort   # 1D array of length n_ort
        self.soc1 = soc1  # 2D array of shape (n_soc1, n_soc1)
        self.soc2 = soc2  # 2D array of shape (n_soc2, n_soc2)


def solve_nt_scaling_vector(W1: NTScaling2, g: np.ndarray,
                            idx_ort: np.ndarray,
                            idx_soc1: np.ndarray,
                            idx_soc2: np.ndarray) -> np.ndarray:
    """
    Emulates:  W1 \ g   where W1 is an NTScaling2 object and g is a vector.
    
    - idx_ort, idx_soc1, idx_soc2 define which entries belong to the orthant
      or the two SOC parts.
    - For the orthant part, divides by W1.ort elementwise.
    - For the SOC parts, solves using the stored Cholesky factors.
    
    Args:
        W1 (NTScaling2): NT scaling object containing scaling matrices and factors.
        g (np.ndarray): The vector to be scaled.
        idx_ort (np.ndarray): Indices for the orthant part of the vector.
        idx_soc1 (np.ndarray): Indices for the first SOC part of the vector.
        idx_soc2 (np.ndarray): Indices for the second SOC part of the vector.
    
    Returns:
        np.ndarray: The result of applying the inverse NT scaling to g.
    """

    # Orthant part
    g_ort = g[idx_ort]
    sol_ort = g_ort / W1.ort

    # Possibly extend the solution for the first SOC
    sol_list = [sol_ort]

    
    if len(idx_soc1) > 0:
        g_soc1 = g[idx_soc1]

        sol_soc1 = cho_solve(W1.soc1_fact, g_soc1)
        # print("sol_soc1:\n", sol_soc1)
        # y_W1_soc1_fact = solve_triangular(W1.soc1_fact, g_soc1, lower=False)
        # sol_soc1 = solve_triangular(W1.soc1_fact.T, y_W1_soc1_fact, lower=True)
        sol_list.append(sol_soc1)

    # Possibly extend for the second SOC
    if len(idx_soc2) > 0:
        g_soc2 = g[idx_soc2]

        sol_soc2 = cho_solve(W1.soc2_fact, g_soc2)
        # print("sol_soc2:\n", sol_soc2)
        
        # y_W1_soc2_fact = solve_triangular(W1.soc2_fact, g_soc2, lower=False)
        # sol_soc2 = solve_triangular(W1.soc2_fact.T, y_W1_soc2_fact, lower=True)
        sol_list.append(sol_soc2)

    return np.concatenate(sol_list)


def solve_nt_scaling_scaling2(W1: NTScaling2, W2: Scaling2) -> Scaling2:
    """
    Emulates:  W1 \ W2   where W2 is a Scaling2 object (matrix-like).
    
    For the orthant portion: performs elementwise division W2.ort ./ W1.ort.
    For the SOC portions: uses the stored Cholesky factors in W1 to solve systems
    and transform the corresponding blocks of W2.
    
    Args:
        W1 (NTScaling2): NT scaling object containing scaling matrices and factors.
        W2 (Scaling2): Scaling matrix object to be transformed.
    
    Returns:
        Scaling2: A new Scaling2 object resulting from applying the inverse NT scaling.
    """
    # Orthant
    ort_new = W2.ort / W1.ort

    # soc1
    if W1.soc1.size > 0:  # i.e. if n_soc1 > 0
        y_W1_soc1_fact = solve_triangular(W1.soc1_fact, W2.soc1, lower=True)
        soc1_new = solve_triangular(W1.soc1_fact.T, y_W1_soc1_fact, lower=False)
    else:
        soc1_new = np.zeros((0, 0))

    # soc2
    if W1.soc2.size > 0:
        y_W1_soc2_fact = solve_triangular(W1.soc2_fact, W2.soc2, lower=True)
        soc2_new = solve_triangular(W1.soc2_fact.T, y_W1_soc2_fact, lower=False)
    else:
        soc2_new = np.zeros((0, 0))

    return Scaling2(ort_new, soc1_new, soc2_new)


def solve_nt_scaling_matrix(W1: NTScaling2,
                            G: np.ndarray,
                            idx_ort: np.ndarray,
                            idx_soc1: np.ndarray,
                            idx_soc2: np.ndarray
) -> np.ndarray:
    """
    Emulates: W1 \ G  where G is a matrix.
    
    The function applies 'solve_nt_scaling_vector' to each column of G
    and horizontally concatenates the results to form the transformed matrix.
    
    Args:
        W1 (NTScaling2): NT scaling object containing scaling matrices and factors.
        G (np.ndarray): Matrix to be scaled.
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC part.
        idx_soc2 (np.ndarray): Indices for the second SOC part.
    
    Returns:
        np.ndarray: The matrix resulting from applying the inverse NT scaling to G.
    """
    c = G.shape[1]
    # Solve column by column
    cols = []
    for i in range(c):
        g_col = G[:, i]
        col_solved = solve_nt_scaling_vector(W1, g_col, idx_ort, idx_soc1, idx_soc2)
        cols.append(col_solved.reshape(-1, 1))

    # print("=====================================")
    # print("=====================================")
    # print("=====================================")


    # print("cols:\n", cols)


    return np.hstack(cols)


def multiply_nt_scaling_vector(W1: NTScaling2,
                               g: np.ndarray,
                               idx_ort: np.ndarray,
                               idx_soc1: np.ndarray,
                               idx_soc2: np.ndarray) -> np.ndarray:
    """
    Emulates: W1 * g  where W1 is an NTScaling2 object and g is a vector.
    
    - For the orthant part: multiplies elementwise by W1.ort.
    - For the SOC parts: multiplies using the stored SOC scaling matrices.
    
    Args:
        W1 (NTScaling2): NT scaling object.
        g (np.ndarray): Vector to be scaled.
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC part.
        idx_soc2 (np.ndarray): Indices for the second SOC part.
    
    Returns:
        np.ndarray: The result of multiplying W1 by g.
    """
    g_ort = g[idx_ort]
    sol_ort = g_ort * W1.ort

    sol_list = [sol_ort]
    if len(idx_soc1) > 0:
        g_soc1 = g[idx_soc1]
        sol_soc1 = W1.soc1 @ g_soc1
        sol_list.append(sol_soc1)

    if len(idx_soc2) > 0:
        g_soc2 = g[idx_soc2]
        sol_soc2 = W1.soc2 @ g_soc2
        sol_list.append(sol_soc2)

    return np.concatenate(sol_list)


def multiply_nt_scaling_matrix(W1: NTScaling2, 
                               G: np.ndarray,
                               idx_ort: np.ndarray,
                               idx_soc1: np.ndarray,
                               idx_soc2: np.ndarray) -> np.ndarray:
    """
    Emulates: W1 * G  for a matrix G.
    
    Applies 'multiply_nt_scaling_vector' to each column of G
    and horizontally concatenates the results.
    
    Args:
        W1 (NTScaling2): NT scaling object.
        G (np.ndarray): Matrix to be multiplied.
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC part.
        idx_soc2 (np.ndarray): Indices for the second SOC part.
    
    Returns:
        np.ndarray: The result of multiplying W1 by G.
    """
    c = G.shape[1]
    cols = []
    for i in range(c):
        g_col = G[:, i]
        col_mult = multiply_nt_scaling_vector(W1, g_col, idx_ort, idx_soc1, idx_soc2)
        cols.append(col_mult.reshape(-1, 1))
    return np.hstack(cols)


def multiply_scaling2_vector(W: Scaling2,
                             g: np.ndarray,
                             idx_ort: np.ndarray,
                             idx_soc1: np.ndarray,
                             idx_soc2: np.ndarray) -> np.ndarray:
    """
    Emulates: W * g  where W is a Scaling2 object and g is a vector.
    
    Multiplies the orthant part elementwise and applies matrix
    multiplication for the SOC parts.
    
    Args:
        W (Scaling2): Scaling matrix object.
        g (np.ndarray): Vector to be scaled.
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC part.
        idx_soc2 (np.ndarray): Indices for the second SOC part.
    
    Returns:
        np.ndarray: The result of multiplying W by g.
    """
    g_ort = g[idx_ort]
    sol_ort = g_ort * W.ort

    sol_list = [sol_ort]
    if len(idx_soc1) > 0:
        g_soc1 = g[idx_soc1]
        sol_soc1 = W.soc1 @ g_soc1
        sol_list.append(sol_soc1)

    if len(idx_soc2) > 0:
        g_soc2 = g[idx_soc2]
        sol_soc2 = W.soc2 @ g_soc2
        sol_list.append(sol_soc2)

    return np.concatenate(sol_list)

def multiply_scaling2_matrix(W: Scaling2,
                             G: np.ndarray,
                             idx_ort: np.ndarray,
                             idx_soc1: np.ndarray,
                             idx_soc2: np.ndarray) -> np.ndarray:
    """
    Emulates: W * G  for a matrix G where W is a Scaling2 object.
    
    Applies 'multiply_scaling2_vector' to each column of G
    and horizontally concatenates the results.
    
    Args:
        W (Scaling2): Scaling matrix object.
        G (np.ndarray): Matrix to be multiplied.
        idx_ort (np.ndarray): Indices for the orthant part.
        idx_soc1 (np.ndarray): Indices for the first SOC part.
        idx_soc2 (np.ndarray): Indices for the second SOC part.
    
    Returns:
        np.ndarray: The result of multiplying W by G.
    """
    c = G.shape[1]
    cols = []
    for i in range(c):
        g_col = G[:, i]
        col_mult = multiply_scaling2_vector(W, g_col, idx_ort, idx_soc1, idx_soc2)
        cols.append(col_mult.reshape(-1, 1))
    return np.hstack(cols)


def soc_NT_scaling(s_soc: np.ndarray, z_soc: np.ndarray) -> np.ndarray:
    """
    Computes the Nesterov-Todd scaling matrix for a single second-order cone (SOC)
    given slack variables s_soc and z_soc belonging to that cone.
    
    The function normalizes the input vectors, computes scaling factors,
    and constructs the NT scaling matrix based on the SOC structure.
    
    Args:
        s_soc (np.ndarray): Slack vector in the SOC.
        z_soc (np.ndarray): Dual slack vector in the SOC.
    
    Returns:
        np.ndarray: The Nesterov-Todd scaling matrix for the given SOC.
    """

    n_soc = s_soc.size
    if n_soc == 0:
        # Return an empty 0x0 matrix
        return np.zeros((0, 0))


    z_bar = normalize_soc(z_soc)
    s_bar = normalize_soc(s_soc)
    gamma = np.sqrt((1.0 + np.dot(z_bar, s_bar)) / 2.0)

    # w_bar = 1/(2*gamma)*(s_bar + [z_bar[0], -z_bar[1:]])
    v_idx = slice(1, n_soc)    
    z_bar_tail = z_bar[v_idx]
    s_bar_tail = s_bar[v_idx]

    w_bar_top = (s_bar[0] + z_bar[0]) / (2 * gamma)
    w_bar_tail = (s_bar_tail - z_bar_tail) / (2 * gamma)
    w_bar = np.concatenate(([w_bar_top], w_bar_tail))

    # b = 1.0 / (w_bar[0] + 1)
    b = 1.0 / (w_bar[0] + 1.0)


    # Build the top part
    w_bar_reshaped = w_bar.reshape(-1, 1)  # column vector
    # top row is w_bar^T (1 x n_soc)
    W_top = w_bar[np.newaxis, :]  # shape (1, n_soc)

    # Build the bottom part
    n_tail = n_soc - 1
    I_n_tail = np.eye(n_tail)
    w_tail_tailT = w_bar_tail.reshape(-1, 1) @ w_bar_tail.reshape(1, -1)
    bottom_right = I_n_tail + b * w_tail_tailT

    # We want to stack horizontally: [w_bar_tail, bottom_right] 
    # but 'w_bar_tail' is (n_tail,) -> need (n_tail, 1)
    w_bar_tail_col = w_bar_tail.reshape(-1, 1)
    bottom_block = np.hstack([w_bar_tail_col, bottom_right])

    # Now stack vertically with W_top
    W_bar = np.vstack([
        W_top,               # shape (1, n_soc)
        bottom_block         # shape (n_tail, n_soc)
    ])

    eta = ((soc_quad_J(s_soc) / soc_quad_J(z_soc)) ** 0.25
           if soc_quad_J(z_soc) != 0 else 1.0)

    W_soc = eta * W_bar
    return W_soc

def calc_NT_scalings(s: np.ndarray,
                     z: np.ndarray,
                     idx_ort: np.ndarray,
                     idx_soc1: np.ndarray,
                     idx_soc2: np.ndarray) -> NTScaling2:
    """
    Computes the Nesterov-Todd scaling for a composite cone consisting of 
    an orthant and two second-order cones (SOCs).
    
    Given slack vectors s and z partitioned according to the indices for 
    the orthant and each SOC, this function:
      - Computes the scaling factors for the orthant part.
      - Computes the NT scaling matrices for each SOC using soc_NT_scaling.
      - Performs Cholesky factorizations of these SOC scaling matrices.
    
    Args:
        s (np.ndarray): Slack variables partitioned into orthant and SOC parts.
        z (np.ndarray): Dual slack variables partitioned similarly to s.
        idx_ort (np.ndarray): Indices corresponding to the orthant portion.
        idx_soc1 (np.ndarray): Indices corresponding to the first SOC portion.
        idx_soc2 (np.ndarray): Indices corresponding to the second SOC portion.
    
    Returns:
        NTScaling2: An object containing orthant scaling factors, SOC scaling matrices, 
                    and their Cholesky factorizations.
    """

    # Orth
    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    W_ort = np.sqrt(s_ort / z_ort)  # elementwise

    # SOC 1
    s_soc1 = s[idx_soc1]
    z_soc1 = z[idx_soc1]
    W_soc1 = soc_NT_scaling(s_soc1, z_soc1)

    # SOC 2
    s_soc2 = s[idx_soc2]
    z_soc2 = z[idx_soc2]
    W_soc2 = soc_NT_scaling(s_soc2, z_soc2)


    # Build Cholesky factors
    if W_soc1.size > 0:
        #soc1_fact = cholesky(W_soc1)
        soc1_fact = cho_factor(W_soc1)
    else:
        soc1_fact = (np.zeros((0, 0)), True)

    if W_soc2.size > 0:
        #soc2_fact = cholesky(W_soc2)
        soc2_fact = cho_factor(W_soc2)
    else:
        soc2_fact = (np.zeros((0, 0)), True)

    return NTScaling2(W_ort, W_soc1, soc1_fact, W_soc2, soc2_fact)
