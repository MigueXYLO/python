import numpy as np

class Perceptron:
    """
    A simple implementation of the Perceptron algorithm for binary classification.
    """

    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, xx: np.ndarray, yy: np.ndarray, learning_rate: float = 0.1, seed: int = 1) -> None:
        """
        Trains the Perceptron model on the given input data.

        Args:
            xx (np.ndarray): The input features of shape (num_samples, num_features).
            yy (np.ndarray): The target labels of shape (num_samples,).
            learning_rate (float, optional): The learning rate for updating the weights. Defaults to 0.1.
            seed (int, optional): The seed value for random number generation. Defaults to 1.
        """
        # Create random weights
        num_features = xx.shape[1]
        rng = np.random.default_rng(seed)
        self.w = rng.random(num_features)
        self.b = rng.random()
        num_samples = xx.shape[1]
        
        # Training algorithm
        change = True
        while change:
            change = False
            for i, x in enumerate(xx):
                o = self.predict(x)
                print(str(x) + ' ' + str(o))
                print("-------------------")
                if o != yy[i]:
                    change = True
                    update = learning_rate * (yy[i] - o)
                    self.w += update * x
                    self.b += update
            print("--------------------")        

    def predict(self, x: np.ndarray) -> int:
        """
        Predicts the class label for a single input sample.

        Args:
            x (np.ndarray): The input sample of shape (num_features,).

        Returns:
            int: The predicted class label (-1 or 1).
        """
        weighted_sum = np.dot(x, self.w) + self.b
        return 1 if weighted_sum > 0 else -1
