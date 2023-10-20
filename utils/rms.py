import numpy as np


def rms(sigIn):
    return np.sqrt(np.mean(np.square(sigIn)))