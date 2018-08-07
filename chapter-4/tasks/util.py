import numpy as np


def build_connectivity_matrix(p_mtrx):
    n = p_mtrx.shape[1]
    hb = p_mtrx.T.dot(p_mtrx)
    pr = (2 * p_mtrx.T - 1).dot(p_mtrx)
    po = p_mtrx.T.dot(2 * p_mtrx - 1)
    hp = (2 * p_mtrx.T - 1).dot(2 * p_mtrx - 1)

    mask = np.ones(n) - np.eye(n)
    hb = hb * mask
    pr = pr * mask
    po = po * mask
    hp = hp * mask
    return hb, pr, po, hp


def sync_update(w, p, n):
    r = [p]
    for i in range(n):
        p = w.dot(p)
        p = np.clip(p, 0, 1)
        r.append(p)
    return np.array(r)


def async_update(w, p, n):
    r = [p]
    for i in range(1, n + 1):
        rnd_idx = np.random.randint(0, w.shape[0])
        p = p.copy()
        p[rnd_idx] = w[rnd_idx].dot(p)
        p = np.clip(p, 0, 1)
        if i % int(n / 10) == 0:
            r.append(p)
    return np.array(r)
