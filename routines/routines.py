import copy
import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Calculates the sigmoid function of z.

    Parameters:
        z (array_like): A scalar or numpy array of any size.
    
    Returns:
    g (array_like): The sigmoid of z.
    """
    z = np.clip(z, -500, 500) # Prevent overflow
    g = 1 / (1 + np.exp(-z))

    return g

