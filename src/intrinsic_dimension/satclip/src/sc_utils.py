import numpy as np

def string_to_array(s):
    s = s.replace('[', '').replace(']', '').replace('\n', '')
    return np.array(s.split(), dtype=float)