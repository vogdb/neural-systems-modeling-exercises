import numpy as np


def H(prob_list):
    h = 0
    for prob in prob_list:
        if prob != 0:
            h += prob * np.log2(prob)
    return -h


def infocomp(pX, condi):
    nY, nX = condi.shape

    hX = H(pX)
    # print(hX)

    # joint condi
    joint = np.zeros(condi.shape)
    for j in range(nX):
        joint[:, j] = pX[j] * condi[:, j]

    # hY is marginal of joint condi
    pY = np.sum(joint, axis=1)
    hY = H(pY)

    # mutual information I
    I = 0
    for j in range(nX):
        for i in range(nY):
            if joint[i, j] != 0:
                I += joint[i, j] * np.log2(joint[i, j] / (pX[j] * pY[i]))
    return hX, hY, I
