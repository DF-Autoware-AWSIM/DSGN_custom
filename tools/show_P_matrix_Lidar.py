import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- Step 1: Read the .bin point cloud file ---
def read_bin_file(bin_path):
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud  # shape: (N, 4): x, y, z, reflectance

# --- Step 2: Apply transformation using RPY and translation ---
def transform_point_cloud(points, rpy_deg, translation):
    # Convert RPY degrees to radians
    rpy_rad = np.radians(rpy_deg)

    # Create rotation matrix
    rotation = R.from_euler('xyz', rpy_rad).as_matrix()
    print(f"Rotation Matrix:\n{rotation}")
    # Apply rotation and translation
    points_xyz = points[:, :3]
    transformed_xyz = (rotation @ points_xyz.T).T + translation

    # Combine with reflectance again
    transformed_points = np.hstack((transformed_xyz, points[:, 3:4]))
    # Keep only points with positive z values
    #transformed_points = transformed_points[transformed_points[:, 2] > 0]

    return transformed_points

# --- Step 3: Project point cloud to image using projection matrix ---
def project_lidar_to_image(points, P):
    # Only keep points in front of the LiDAR
    mask = points[:, 0] > 0
    points = points[mask]

    # Convert to homogeneous coordinates (N, 4)
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # (N, 4)

    # Project using P (3x4)
    points_2d = P @ points_hom.T  # shape: (3, N)

    # Normalize to get pixel coordinates
    points_2d[:2] /= points_2d[2]

    return points_2d[:2].T  # shape: (N, 2)

# --- Step 4: Visualize ---
import matplotlib.cm as cm

def show_projection(points_2d, image_size=(1920, 1080)):
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    num_points = len(points_2d)
    colormap = cm.get_cmap('jet')  # 'jet' goes from blue to red

    for i, pt in enumerate(points_2d):
        u, v = int(pt[0]), int(pt[1])
        if 0 <= u < image_size[0] and 0 <= v < image_size[1]:
            color = colormap(i / num_points)  # Normalize index to [0, 1]
            bgr_color = tuple(int(255 * c) for c in color[:3][::-1])  # Convert to BGR
            cv2.circle(image, (u, v), 1, bgr_color, -1)

    cv2.imshow("LiDAR Projection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main ---
if __name__ == "__main__":
    bin_path = "/home/arka/DSGN/data/awsim/training/velodyne/000007.bin"

    # --- Projection matrix (example KITTI calibration) ---
    P = np.array([[960.0, 0.000000e+00, 960.5, 259.20001220703125],
                  [0.000000e+00, 959.3908081054688, 540.5, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

    # --- Define RPY and translation for frame transformation ---
    # Rotation: roll, pitch, yaw in degrees
    rpy_deg = [-90, 0, 179.7]  # e.g., rotate 45° around z-axis
    translation = np.array([0.0, -0.05, -0.1])  # Translate x, y, z
    #translation = np.array([0.0, -0.05, -0.1])  # Translate x, y, z


    # --- Pipeline ---
    points = read_bin_file(bin_path)
    transformed_points = transform_point_cloud(points, rpy_deg, translation)
    points_2d = project_lidar_to_image(transformed_points, P)
    show_projection(points_2d)
