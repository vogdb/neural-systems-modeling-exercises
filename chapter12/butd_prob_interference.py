import numpy as np

pl = np.array([.5, .5])
pf = np.array([.5, .5])

# conditional C given L and F
# columns are pairs of L and F, rows are values of C
pc_glf = np.array([
    #  L=1,F=1; L=2,F=1; L=1,F=2; L=2,F=2
    [0.5, 0.1, 0.1, 0.3],
    [0.1, 0.5, 0.3, 0.1],
    [0.1, 0.3, 0.5, 0.1],
    [0.3, 0.1, 0.1, 0.5],
])

# conditional I given C
# columns are C, rows are I
pi_gc = np.array([
    [0.5, 0.1, 0.1, 0.3],
    [0.1, 0.5, 0.3, 0.1],
    [0.1, 0.3, 0.5, 0.1],
    [0.3, 0.1, 0.1, 0.5],
])


def get_pc_gf(F):
    if F == 1:
        # return for L=1,F=1 and L=2,F=1
        return pc_glf[:, [0, 1]]
    if F == 2:
        # return for L=1,F=2 and L=2,F=2
        return pc_glf[:, [2, 3]]


# un-normalized F=1 given I=1
I = 1
F = 1  # for it we take first two columns of pc_glf
pi1_gc = pi_gc[I - 1]  # -1 as indexing starts from 0
pf1_gi1 = np.sum(np.outer(pl, pi1_gc) * get_pc_gf(F).T)
F = 2
pf2_gi1 = np.sum(np.outer(pl, pi1_gc) * get_pc_gf(F).T)

# normalize
pf1_gi1_bayes = pf1_gi1 / (pf1_gi1 + pf2_gi1)
print(pf1_gi1_bayes)
