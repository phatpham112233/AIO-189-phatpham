import numpy as np

def compute_cosine(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


x = np.array([1, 2, 3, 4])
y = np.array([1, 0, 3, 0])
print(compute_cosine(x, y))  
