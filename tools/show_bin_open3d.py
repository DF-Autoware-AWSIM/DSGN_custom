import numpy as np
import open3d as o3d

# Load the .bin file
#bin_path = "/home/arka/DSGN/data/kitti/training/velodyne/000000.bin"
#bin_path = "/home/arka/ros2_ws/src/lidar_to_bin/saved_bins/000149.bin"
#bin_path = "/home/arka/ros2_ws/src/awsim_to_kitti/awsim_dataset/velodyne/000173.bin"
#bin_path = "/home/arka/DSGN/data/awsim/training/velodyne/000007.bin"
bin_path = "/home/arka/DSGN/data/awsim_debug/training/depth/000000_trans_3d.bin"
points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # use only x, y, z

# Visualize
o3d.visualization.draw_geometries([pcd])


