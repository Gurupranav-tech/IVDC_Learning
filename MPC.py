import numpy as np


class MPC:
    def __init__(self, A, B, C, prediction_horizon: int, control_horizon: int, W3, W4, initial_state, desired_trajectory):
        self.A = A
        self.B = B
        self.C = C
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.input_weight_matrix = W3
        self.output_weight_matrix = W4
        self.desired_trajectory = desired_trajectory

        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]

        self.time_step = 0

        self.states = [initial_state]

        self.computed_inputs = []
        self.possible_outputs = []

        self.O, self.M, self.gain_matrix = self.form_lifted_matrices()


    def form_lifted_matrices(self):
        O = np.zeros(shape=(self.prediction_horizon * self.r, self.n))

        powA = self.A
        for i in range(self.prediction_horizon):
            if i == 0:
                powA = self.A
            else:
                powA = np.matmul(powA, self.A)
            O[i*self.r:(i+1)*self.r, :] = np.matmul(self.C, powA)

        M = np.zeros(shape=(self.prediction_horizon*self.r, self.control_horizon*self.m))
        for i in range(self.prediction_horizon):
            if i < self.control_horizon:
                for j in range(i+1):
                    if j == 0:
                        powA = np.eye(self.n, self.n)
                    else:
                        powA = np.matmul(powA, self.A)
                    M[i*self.r:(i+1)*self.r, (i-j)*self.m:(i-j+1)*self.m] = np.matmul(self.C, np.matmul(powA, self.A))
            else:
                for j in range(self.control_horizon):
                    if j == 0:
                        sum_last = np.zeros(shape=(self.n, self.n))
                        for s in range(i-self.control_horizon+2):
                            if s == 0:
                                powA = np.eye(self.n, self.n)
                            else:
                                powA = np.matmul(powA, self.A)
                            sum_last = sum_last + powA
                        M[i*self.r: (i+1)*self.r, (self.control_horizon-1)*self.m: self.control_horizon*self.m] \
                            = np.matmul(self.C, np.matmul(sum_last, self.B))
                    else:
                        powA = np.matmul(powA, self.A)
                        M[i*self.r: (i+1)*self.r, (self.control_horizon-1-j)*self.m:(self.control_horizon-j)*self.m] \
                            = np.matmul(self.C, np.matmul(powA, self.B))

        tmp1 = np.matmul(M.T, np.matmul(self.output_weight_matrix, M))
        tmp2 = np.linalg.inv(tmp1 + self.input_weight_matrix)
        gain_matrix = np.matmul(tmp2, np.matmul(M.T, self.output_weight_matrix))

        return O, M, gain_matrix

    def compute_future(self, state, control_input):
        xkp1 = np.zeros(shape=(self.n, 1))
        yk = np.zeros(shape=(self.r, 1))
        xkp1 = np.matmul(self.A, state) + np.matmul(self.B, control_input)
        yk = np.matmul(self.C, state)

        return xkp1, yk