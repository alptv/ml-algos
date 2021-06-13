import numpy as np


def normalize(signs):
    cp_signs = np.copy(signs)
    col_count = len(signs[0])
    row_count = len(signs)
    for j in range(col_count):
        mx = signs[0][j]
        mn = signs[0][j]
        for i in range(row_count):
            mx = max(signs[i][j], mx)
            mn = min(signs[i][j], mn)
        for i in range(row_count):
            cp_signs[i][j] -= mn
            cp_signs[i][j] /= (mx - mn)
    return cp_signs


def rand(clust_targets, real_targets):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(clust_targets)):
        for j in range(i + 1, len(clust_targets)):
            same_clust = clust_targets[i] == clust_targets[j]
            same_targ = real_targets[i] == real_targets[j]
            if same_clust and same_targ:
                TP += 1
            if same_clust and (not same_targ):
                TN += 1
            if (not same_clust) and same_targ:
                FP += 1
            if (not same_clust) and (not same_targ):
                FN += 1
    return (TP + FN) / (TP + TN + FP + FN)


def silhouette(clust_targets, signs, metric, targets_count):
    sil = 0
    for i in range(len(signs)):
        dists = np.full(targets_count, 0.)
        sizes = np.full(targets_count, 0)
        for j in range(len(signs)):
            dists[clust_targets[j]] += metric(signs[i], signs[j])
            sizes[clust_targets[j]] += 1
        for j in range(targets_count):
            dists[j] /= sizes[j]
        a = dists[clust_targets[i]]
        dists = np.delete(dists, clust_targets[i])
        b = min(dists)
        sil += (b - a) / max(a, b)
    return sil / len(signs)

def pca(inp_signs, m):
    avg_signs = np.mean(inp_signs, axis=0)
    signs = inp_signs - avg_signs
    U, S, Vt = np.linalg.svd(signs)
    U = U[:, :m]
    U *= S[:m]
    return U
