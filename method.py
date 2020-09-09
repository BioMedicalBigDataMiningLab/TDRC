import numpy as np
import tensorly as tl
import scipy as sp


class Model(object):
    def __init__(self, name='TDRC'):
        super().__init__()
        self.name = name

    def TDRC(self, X, S_d, S_m, r=4, alpha=0.125, beta=0.25, lam=0.001, tol=1e-6, max_iter=500):

        def CG(X_initial, A, B, D, mu, tol, max_iter):

            X_s = X_initial
            R = D - A * X_s * B - mu * X_s
            P = np.array(R, copy=True)

            for i in range(max_iter):
                R_norm = np.trace(R * R.T)
                Q = A * P * B + mu * P
                alpha = R_norm / np.trace(Q * P.T)
                X_s = X_s + alpha * P
                R = R - alpha * Q
                err = np.linalg.norm(R)
                if err < tol:
                    print("CG convergence: iter = %d" % i)
                    break

                beta = np.trace(R * R.T) / R_norm
                P = R + beta * P

            return X_s

        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]

        # initialization
        rho_1 = 1
        rho_2 = 1
        np.random.seed(0)
        C = np.mat(np.random.rand(m, r))
        P = np.mat(np.random.rand(d, r))
        D = np.mat(np.random.rand(t, r))

        Y_1 = 0
        Y_2 = 0

        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        for i in range(max_iter):
            G = np.mat(tl.tenalg.khatri_rao([P, D]))
            output_X_old = tl.fold(np.array(C * G.T), 0, X.shape)

            O_1 = C.T * C
            O_2 = P.T * P

            M_2 = CG(0, alpha * O_1, O_1, alpha * C.T * S_m * C, lam, 0.01, 200)

            M_3 = CG(0, beta * O_2, O_2, beta * P.T * S_d * P, lam, 0.01, 200)

            K = np.mat(np.eye(r))

            F = C * M_2
            J = (alpha * S_m.T * F + rho_1 * C + Y_1) * np.linalg.inv(alpha * F.T * F + rho_1 * np.eye(r))
            Q = M_2 * J.T
            C = (X_1 * G + alpha * S_m * Q.T + rho_1 * J - Y_1) * np.linalg.inv(
                G.T * G + alpha * Q * Q.T + rho_1 * np.eye(r))

            R = P * M_3
            W = (beta * S_d.T * R + rho_2 * P + Y_2) * np.linalg.inv(beta * R.T * R + rho_2 * np.eye(r))
            Y_1 = Y_1 + rho_1 * (C - J)
            rho_1 = rho_1 * 1.1

            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            Z = M_3 * W.T
            P = (X_2 * U + beta * S_d * Z.T + rho_2 * W - Y_2) * np.linalg.inv(
                U.T * U + beta * Z * Z.T + rho_2 * np.eye(r))
            Y_2 = Y_2 + rho_2 * (P - W)
            rho_2 = rho_2 * 1.1

            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_3 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

            output_X = tl.fold(np.array(D * B.T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            # print(err)
            if err < tol:
                # print(i)
                break

        predict_X = np.array(tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape))

        return predict_X

    def TFAI_CP_within_mod(self, X, S_m, S_d, r=3, alpha=0.25, beta=1.0, lam=0.1, tol=1e-7, max_iter=500, seed=0):
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

            C = np.mat(
                sp.linalg.solve_sylvester(np.array(alpha * L_C + lam * np.mat(np.eye(m))), np.array(G.T * G), X_1 * G))
            U = np.mat(tl.tenalg.khatri_rao([C, D]))
            P = np.mat(
                sp.linalg.solve_sylvester(np.array(beta * L_P + lam * np.mat(np.eye(d))), np.array(U.T * U), X_2 * U))
            B = np.mat(tl.tenalg.khatri_rao([C, P]))
            D = X_3 * B * np.linalg.inv(B.T * B + lam * np.eye(r))

            output_X = tl.fold(np.array(D * B.T), 2, X.shape)
            err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)
            if err < tol:
                print(i)
                break
        predict_X = tl.fold(np.array(C * np.mat(tl.tenalg.khatri_rao([P, D])).T), 0, X.shape)
        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)
