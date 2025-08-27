"""Controller module"""
import numpy as np


class Controller:
    """Simple feedforward neural network controller"""

    def __init__(self, input_size: int = 6, hidden_size: int = 8, output_size: int = 3) -> None:
        """Create a controller with random weights

        Args:
            input_size (int, optional): input layer size. Defaults to 6.
            hidden_size (int, optional): hidden layer size. Defaults to 15.
            output_size (int, optional): output layer size. Defaults to 3.
        """
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> tuple[float, float, float]:
        """Feedforward pass

        Args:
            x (np.ndarray[np.float64]): state input

        Returns:
            tuple[float, float, float]: thrust command [0,1],
                                        control surface command [-1,1],
                                        wheel brake command [0,1]
        """
        hidden = np.tanh(x @ self.w1 + self.b1)
        out = np.tanh(hidden @ self.w2 + self.b2)
        return (out[0] + 1) / 2, out[1], (out[2] + 1) / 2  # thrust [0,1], control surface [-1,1], wheel brake [0,1]
    
    def mutate(self, rate: float = 0.1):
        """Mutate the controller weights

        Args:
            rate (float, optional): mutation rate. Defaults to 0.1.
        """
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            arr += rate * np.random.randn(*arr.shape)

    def save(self, filename: str) -> None:
        """Save the controller weights to a file

        Args:
            filename (str): name of the file
        """
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, filename: str) -> "Controller":
        """Load controller weights from a file

        Args:
            filename (str): name of the file

        Returns:
            Controller: loaded controller
        """
        data = np.load(filename)
        controller = cls(input_size=data['w1'].shape[0],
                         hidden_size=data['w1'].shape[1],
                         output_size=data['w2'].shape[1])
        controller.w1 = data['w1']
        controller.b1 = data['b1']
        controller.w2 = data['w2']
        controller.b2 = data['b2']
        return controller
