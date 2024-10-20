from scipy.optimize import minimize


class MPC:
    def __init__(self, A, B, horizon, desired_trajectory, initial_state):
        self.A = A
        self.B = B
        self.horizon = horizon
        self.desired_trajectory = desired_trajectory
        self.time_step = 0

        self.states = [initial_state]

    def propagate(self, state, control_input):
        return self.A * state + self.B * control_input

    def cost(self, control_inputs, initial_state, desired_trajectory):
        cost = 0
        state = initial_state
        size = self.horizon if len(desired_trajectory) >= self.horizon else len(desired_trajectory)
        for i in range(size):
            control = control_inputs[i]
            state = self.propagate(state, control)
            cost += abs(state - desired_trajectory[i]) + 0.1 * control
        return cost

    def solution(self):
        if self.time_step >= len(self.desired_trajectory) - self.horizon:
            desired_trajectory = self.desired_trajectory[self.time_step:self.time_step+1]
        else:
            desired_trajectory = self.desired_trajectory[self.time_step:self.time_step + self.horizon]
        # desired_trajectory = self.desired_trajectory[self.time_step:self.time_step + self.horizon]
        control_inputs = [0 for i in range(self.horizon)]
        state = self.states[self.time_step]

        result = minimize(
            self.cost,
            control_inputs,
            args=(state, desired_trajectory),
        )
        self.time_step += 1
        if self.time_step >= len(self.desired_trajectory) - self.horizon:
            for i, control in enumerate(result.x):
                state = self.states[self.time_step-1+i]
                self.states.append(self.propagate(state, control))
        else:
            control = result.x[0]
            state = self.propagate(state, control)
            self.states.append(state)

    def complete(self):
        for i in range(len(self.desired_trajectory) - self.horizon):
            self.solution()

        return self.states