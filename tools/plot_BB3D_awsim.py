#This script loads images from awsim_dataset, the predicted 3D bounding boxes from DSGN,
#  and visualizes the 3D boxes on the images.
import numpy as np
import cv2
import os

def read_kitti_detection(filename):
    """Parse a KITTI detection file into a list of detections."""
    detections = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue  # Skip invalid lines
            obj = {
                "type": fields[0],
                "truncated": float(fields[1]),
                "occluded": int(fields[2]),
                "alpha": float(fields[3]),
                "bbox": [float(x) for x in fields[4:8]],  # 2D bbox
                "dimensions": [float(x) for x in fields[8:11]],  # h, w, l
                "location": [float(x) for x in fields[11:14]],  # x, y, z
                "rotation_y": float(fields[14]),
                "score": float(fields[15]) if len(fields) > 15 else None
            }
            detections.append(obj)
    return detections

def read_kitti_calib(calib_file):
    """Read KITTI calibration file and return the P2 camera matrix."""
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                return P2
    raise ValueError("P2 matrix not found in calibration file.")

def compute_box_3d(dim, loc, ry):
    """Returns 3D box corners in camera coordinate system."""
    h, w, l = dim
    x, y, z = loc
    # 3D bounding box corners in object coordinate
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.array([x_corners, y_corners, z_corners])
    # Rotation
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = np.dot(R, corners)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    return corners_3d

def project_to_image(pts_3d, P):
    """Project 3D points to image plane."""
    n = pts_3d.shape[1]
    pts_3d_hom = np.vstack((pts_3d, np.ones((1, n))))
    pts_2d = np.dot(P, pts_3d_hom)
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2]

def draw_projected_box3d(img, qs, color=(0,255,0), thickness=5):
    """Draw 3d bounding box in image"""
    qs = qs.astype(np.int32).T
    for k in range(0,4):
        i,j = k,(k+1)%4
        cv2.line(img, tuple(qs[i]), tuple(qs[j]), color, thickness)
        cv2.line(img, tuple(qs[i+4]), tuple(qs[j+4]), color, thickness)
        cv2.line(img, tuple(qs[i]), tuple(qs[i+4]), color, thickness)
    return img

if __name__ == "__main__":
    import sys
    
    #data_path = "/home/arka/DSGN/data/awsim/training/"
    #data_path = "/home/arka/DSGN/data/awsim/testing/"
    data_path = "/home/arka/DSGN/data/awsim/testing_offline/"
    output_path = "/home/arka/ros2_ws/src/dsgn_offline/resource/awsim_output_offline/"
    #output_path = "/home/arka/DSGN/outputs/dsgn_12g_awsim_remote_downsample/kitti_output/"
    image_folder = data_path + "image_2/"
    calib_folder = data_path + "calib/"

    # List all image files (assume .png)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    for img_file in image_files:
        print(f"Processing {img_file}")
        img_path = os.path.join(image_folder, img_file)
        calib_path = os.path.join(calib_folder, img_file.replace('.png', '.txt'))
        detection_path = os.path.join(output_path, img_file.replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue

        try:
            P2 = read_kitti_calib(calib_path)
        except Exception as e:
            print(f"Error reading calibration file {calib_path}: {e}")
            continue
        
        if not os.path.exists(detection_path):
            print(f"Detection file {detection_path} not found, skipping.")
            continue

        detections = read_kitti_detection(detection_path)

        for det in detections:
            if det["type"].lower() != "car":  # Change class filter if needed
                continue
            corners_3d = compute_box_3d(det["dimensions"], det["location"], det["rotation_y"])
            corners_2d = project_to_image(corners_3d, P2)
            img = draw_projected_box3d(img, corners_2d)

        cv2.imshow("3D Boxes", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Press 'q' to quit early
            break

    cv2.destroyAllWindows()
