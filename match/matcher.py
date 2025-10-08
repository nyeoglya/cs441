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
    A = np.stack(descA)
    B = np.stack(descB)

    # D(i,j) = ||A[i] - B[j]||^2
    dist_sq = np.sum(A**2, axis=1)[:, np.newaxis] + np.sum(B**2, axis=1) - 2 * A.dot(B.T)
    dist_sq = np.maximum(dist_sq, 0)

    # nearest neighbors 찾기
    nn_AB = np.argmin(dist_sq, axis=1) # NN in B for each in A
    nn_BA = np.argmin(dist_sq, axis=0) # NN in A for each in B

    matches = []
    for i, j in enumerate(nn_AB):
        # crosscheck. i가 j의 NN인지 체크
        if nn_BA[j] == i:
            dist = np.sqrt(dist_sq[i, j])
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
