import numpy as np

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, dim = x.shape
        mu_0 = np.mean(x[y == 0], axis=0).flatten()
        mu_1 = np.mean(x[y == 1], axis=0).flatten()
        phi = np.mean(y).astype(float)

        sigma = np.zeros((dim, dim))
        for i in range(n):
            delta = (x[i] - (mu_0 if y[i] <= 0.5 else mu_1)).reshape(-1, 1)
            sigma += delta @ delta.T
        sigma /= n

        sigma_inv = np.linalg.inv(sigma)
        theta = (sigma_inv @ (mu_1 - mu_0).reshape(-1, 1))
        theta_0 = np.log(phi/(1 - phi)) - 0.5 * (mu_1 + mu_0).reshape(1, -1) @ sigma_inv @ (mu_1 - mu_0).reshape(-1, 1)

        if theta_0.shape != (1, 1):
            theta_0 = theta_0.reshape((1, 1))

        self.theta = np.vstack((theta_0, theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return (1 / (1 + np.exp(-x.dot(self.theta))) >= 0.5).astype(int).flatten()
        # *** END CODE HERE ***
