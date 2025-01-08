import logging
from primitives.problem_matrices import problem_matrices
from primitives.combine_problem_matrices import combine_problem_matrices
from proximity.pdip import solve_lp_pdip

def proximity_mrp(prim1, prim2, pdip_tol=1e-6, verbose=False):
    """
    Compute the proximity distance and contact point between two primitives using Modified Rodrigues Parameters (MRP).

    Args:
        prim1: First geometric primitive with attributes `r` (position) and `p` (orientation as MRP).
        prim2: Second geometric primitive with attributes `r` (position) and `p` (orientation as MRP).
        pdip_tol: Tolerance for the SOCP solver.
        verbose: Whether to print solver information.

    Returns:
        distance: Proximity distance between the two primitives.
        contact_point: 3D vector representing the contact point.
    """


    # Compute problem matrices for the primitives
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1, prim1.r, prim1.p)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2, prim2.r, prim2.p)

    # Ensure G_ort matrices are 2D
    if len(G_ort1.shape) == 1:
        G_ort1 = G_ort1.reshape(1, -1)
    if len(G_ort2.shape) == 1:
        G_ort2 = G_ort2.reshape(1, -1)

    # print(f"Problem matrices for prim1 computed: G_ort1:\t\n {G_ort1}\n, h_ort1:\t\n {h_ort1}\n, G_soc1:\t\n {G_soc1}\n, h_soc1:\t\n {h_soc1}")
    
    # print(f"Problem matrices for prim2 computed: G_ort2:\t\n {G_ort2}\n, h_ort2:\t\n {h_ort2}\n, G_soc2:\t\n {G_soc2}\n, h_soc2:\t\n {h_soc2}")


    # Combine problem matrices into a single SOCP problem
    c, G, h, idx_ort, idx_soc1, idx_soc2 = combine_problem_matrices(
        G_ort1, h_ort1, G_soc1, h_soc1,
        G_ort2, h_ort2, G_soc2, h_soc2
    )

    # Solve the Second-Order Cone Programming (SOCP) problem
    x, s, z = solve_lp_pdip(c, G, h, idx_ort, idx_soc1, idx_soc2, pdip_tol=pdip_tol)

    # print("x: ", x)
    # print("s: ", s)
    # print("z: ", z)

    # Extract and return the proximity distance and contact point
    distance = x[3]
    contact_point = x[:3]

    return distance, contact_point