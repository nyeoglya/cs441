import numpy as np

def match_l2_crosscheck(descA, descB):
    """
    L2 nearest-neighbor matching with cross-check (mutual NN).

    Input
    -----
    descA : list[np.ndarray]  # descriptors from image A, each (D,)
    descB : list[np.ndarray]  # descriptors from image B, each (D,)

    Output
    ------
    matches : list[(i, j, dist)] where i in A, j in B, and 'dist' is L2 distance

    ------------
    - Matching descriptors by distance; use mutual (cross) check for robustness.
      (General descriptor matching; consistent with local feature use-cases.)
    """
    if len(descA) == 0 or len(descB) == 0:
        return []

    #############################
    ######### Implement here ####
    # Hint:
    # - Stack A and B to arrays.
    # - Build pairwise squared-distance matrix.
    # - Find NN indices for A->B and B->A.
    # - Keep pairs that are mutual AND return their distances.
    diff_mat = np.array(descB)[:, None, :] - np.array(descA) # broadcasting
    dist_mat = np.sqrt(np.sum(diff_mat**2, axis=2)).transpose() # lenA x lenB matrix
    nn = np.argmin(dist_mat, axis=1)
    
    matches = []
    for i, j in enumerate(nn):
        dist = dist_mat[i, j]
        matches.append((i, j, dist))

    return matches
    #############################


# (Optional) Lowe ratio — often used with SIFT; you may keep it provided.
# Not required by slides, so we don't ask students to implement it.
def match_lowe_ratio(descA, descB, ratio=0.75, cross_check=True):
    if len(descA) == 0 or len(descB) == 0: return []
    A = np.stack(descA); B = np.stack(descB)
    AA = (A*A).sum(1)[:,None]; BB = (B*B).sum(1)[None,:]
    D = AA + BB - 2*A.dot(B.T)
    idx = np.argsort(D, axis=1)[:, :2]
    d1 = D[np.arange(len(A)), idx[:,0]]
    d2 = D[np.arange(len(A)), idx[:,1]]
    keep = d1 < (ratio**2) * d2
    if cross_check:
        nnB = np.argmin(D, axis=0)
        return [(i, j1, float(np.sqrt(max(d1[i],0))))
                for i, (j1, _) in enumerate(idx) if keep[i] and nnB[j1] == i]
    else:
        return [(i, j1, float(np.sqrt(max(d1[i],0)))) for i, (j1, _) in enumerate(idx) if keep[i]]
