import numpy as np


class Controller:

    def __init__(self, input_size=6, hidden_size=15, output_size=2):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        out = np.tanh(h @ self.w2 + self.b2)
        return (out[0] + 1) / 2, out[1]
    
    def mutate(self, rate=0.1):
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            arr += rate * np.random.randn(*arr.shape)

    def save(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
