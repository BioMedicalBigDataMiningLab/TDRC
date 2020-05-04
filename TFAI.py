import numpy as np
import tensorly as tl
import scipy as sp


def TFAI_CP_within_mod(X, S_m, S_d, r=3, alpha=0.25, beta=1.0, lam=0.1, tol=1e-7, max_iter=500, seed=0):
    m = X.shape[0]
    d = X.shape[1]
    t = X.shape[2]

    # initialization
    np.random.seed(seed)
    C = np.mat(np.random.rand(m, r))
    P = np.mat(np.random.rand(d, r))
    D = np.mat(np.random.rand(t, r))

    X_1 = np.mat(tl.unfold(X, 0))
    X_2 = np.mat(tl.unfold(X, 1))
    X_3 = np.mat(tl.unfold(X, 2))
    D_C = np.diagflat(S_m.sum(1))
    D_P = np.diagflat(S_d.sum(1))
    L_C = D_C - S_m
    L_P = D_P - S_d

    for i in range(max_iter):
        G = np.mat(tl.tenalg.khatri_rao([P, D]))
        output_X_old = tl.fold(np.array(C * G.T), 0, X.shape)

        C = np.mat(sp.linalg.solve_sylvester(np.array(alpha*L_C+lam*np.mat(np.eye(m))), np.array(G.T*G), X_1*G))
        U = np.mat(tl.tenalg.khatri_rao([C, D]))
        P = np.mat(sp.linalg.solve_sylvester(np.array(beta*L_P+lam*np.mat(np.eye(d))), np.array(U.T*U), X_2*U))
        B = np.mat(tl.tenalg.khatri_rao([C, P]))
        D = X_3 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

        output_X = tl.fold(np.array(D * B.T), 2, X.shape)
        err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
        if err < tol:
            print(i)
            break
    predict_X = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape)
    return predict_X
