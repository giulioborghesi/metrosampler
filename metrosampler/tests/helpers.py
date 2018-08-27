import numpy as np


class MockedConstraints:
    """Mocked constraints used during testing."""

    def __init__(self):
        self.ndim = 2
        self.example = np.array([.2, .5])

    def get_ndim(self):
        return 2

    def get_example(self):
        return self.example

    def apply(self, x):
        return True