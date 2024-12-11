import numpy as np

class uv_filter:
    def __init__(self, u, v, vel, cov, motion_cov, obs_cov):
        '''
        u, v: intial pixel coordinates of the object
        vel: initial velocity of the object
        cov: initial covariance matrix
        motion_cov: covariance matrix for motion model
        obs_cov: covariance matrix for observation model
        '''
        self.state = np.array([u, v, *vel])
        assert cov.shape == (4, 4)
        self.cov = cov
        self.W = motion_cov
        self.V = obs_cov

    def predict(self, dt):
        # motion model for aggregate state (x, x_dot)
        F = np.block([[np.eye(2), dt*np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + self.W
    
    def update(self, z):
        # We have no observation to velocity
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        K = self.cov @ H.T @ np.linalg.inv(H @ self.cov @ H.T + self.V)
        self.state = self.state + K @ (z - H @ self.state)
        self.cov = (np.eye(4) - K @ H) @ self.cov

    def get_state(self):
        state = np.round(self.state[:2]).astype(int)
        return state