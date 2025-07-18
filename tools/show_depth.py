import os
import numpy as np
import matplotlib.pyplot as plt

# Set your directory containing the .npy depth/disparity files
depth_dir = '/home/arka/DSGN/data/kitti/training/depth'  # Update this path
#depth_dir = '/home/arka/DSGN/data/awsim/training/depth'

# List all .npy files in the directory
depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]

for file in depth_files:
    # Load the depth/disparity map
    depth_map = np.load(os.path.join(depth_dir, file))
    
    # Handle invalid values (optional, depending on how your data is formatted)
    # Mask out negative or zero values if needed
    masked = np.ma.masked_where(depth_map <= 0, depth_map)

    plt.figure(figsize=(8, 5))
    plt.title(f'Depth/Disparity Map: {file}')
    cmap = 'plasma'  # Or 'magma', 'viridis', etc.
    plt.imshow(masked, cmap=cmap)
    plt.colorbar(label='Depth/Disparity')
    plt.axis('off')
    plt.show()