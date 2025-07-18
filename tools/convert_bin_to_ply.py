import open3d as o3d
import numpy as np

points = np.fromfile("/home/arka/DSGN/data/awsim/training/velodyne/000205.bin", dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
o3d.io.write_point_cloud("/home/arka/DSGN/data/awsim/training/velodyne/000205.ply", pcd)
