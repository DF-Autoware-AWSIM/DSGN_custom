import numpy as np
from scipy.spatial.transform import Rotation as R

# Quaternion (x, y, z, w)
q = [-0.495, 0.513, -0.505, -0.487]

# Translation vector
t = [0.048, 1.913, -0.919]

# Convert quaternion to rotation matrix
rot = R.from_quat(q)
R_mat = rot.as_matrix()  # 3x3

# Build the 4x4 transformation matrix
T = np.eye(4)
T[:3, :3] = R_mat
T[:3, 3] = t

print("Transformation Matrix:\n", T)
