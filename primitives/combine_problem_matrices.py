import numpy as np

def combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1, G_ort2, h_ort2, G_soc2, h_soc2):
    """
    Combine problem matrices for two primitives.

    Args:
        G_ort1, h_ort1: Orthogonal constraints for the first primitive.
        G_soc1, h_soc1: Second-order cone constraints for the first primitive.
        G_ort2, h_ort2: Orthogonal constraints for the second primitive.
        G_soc2, h_soc2: Second-order cone constraints for the second primitive.

    Returns:
        c: Cost vector.
        G: Combined constraint matrix.
        h: Combined offset vector.
        idx_ort: Indices for orthogonal constraints.
        idx_soc1: Indices for second-order cone constraints of the first primitive.
        idx_soc2: Indices for second-order cone constraints of the second primitive.
    """
    # Compute total orthogonal constraints
    n_ort1, v1 = G_ort1.shape
    n_ort2, v2 = G_ort2.shape
    n_soc1, _ = G_soc1.shape
    n_soc2, _ = G_soc2.shape

    n_ort = n_ort1 + n_ort2
    c = np.array([0, 0, 0, 1.0] + [0] * max(v1, v2 - 4))
    
    idx_ort = np.arange(0, n_ort)  
    idx_soc1 = np.arange(n_ort, n_ort + n_soc1)
    idx_soc2 = np.arange(n_ort + n_soc1, n_ort + n_soc1 + n_soc2)

    if v1 == 4 and v2 == 4:
        # Case 1: Direct stacking
        G = np.vstack([G_ort1, G_ort2, G_soc1, G_soc2])
        
        # stack vector horizontally
        h = np.hstack([h_ort1, h_ort2, h_soc1, h_soc2]) 
        return c[:v1], G, h, idx_ort, idx_soc1, idx_soc2

    elif v1 > 4 and v2 == 4:
        # Case 2: Add zeros to the second
        G_ort_bot = np.hstack([G_ort2, np.zeros((n_ort2, v1 - v2))])
        G_soc_bot = np.hstack([G_soc2, np.zeros((n_soc2, v1 - v2))])
        G = np.vstack([G_ort1, G_ort_bot, G_soc1, G_soc_bot])
        h = np.hstack([h_ort1, h_ort2, h_soc1, h_soc2])
        return c[:v1], G, h, idx_ort, idx_soc1, idx_soc2

    elif v1 == 4 and v2 > 4:
        # Case 3: Add zeros to the first
        G_ort_top = np.hstack([G_ort1, np.zeros((n_ort1, v2 - v1))])
        G_soc_top = np.hstack([G_soc1, np.zeros((n_soc1, v2 - v1))])
        G = np.vstack([G_ort_top, G_ort2, G_soc_top, G_soc2])
        h = np.hstack([h_ort1, h_ort2, h_soc1, h_soc2])
        return c[:v2], G, h, idx_ort, idx_soc1, idx_soc2

    elif v1 > 4 and v2 > 4:
        # Case 4: Extra columns alignment
        v1_extra = v1 - 4
        v2_extra = v2 - 4
        G_ort_bot = np.hstack([G_ort2[:, :4], np.zeros((n_ort2, v1_extra)), G_ort2[:, 4:]])
        G_soc_bot = np.hstack([G_soc2[:, :4], np.zeros((n_soc2, v1_extra)), G_soc2[:, 4:]])
        G = np.vstack([G_ort1, G_ort_bot, G_soc1, G_soc_bot])
        h = np.hstack([h_ort1, h_ort2, h_soc1, h_soc2])
        nc = v1 + v2 - 4
        return c[:nc], G, h, idx_ort, idx_soc1, idx_soc2


    raise ValueError("Failed to combine problem matrices.")
