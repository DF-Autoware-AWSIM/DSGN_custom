import numpy as np
import cv2

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
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
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

def draw_projected_box3d(img, qs, color=(0,255,0), thickness=2):
    """Draw 3d bounding box in image"""
    qs = qs.astype(np.int32).T
    # 0-3: lower corners, 4-7: upper corners
    for k in range(0,4):
        i,j = k,(k+1)%4
        cv2.line(img, tuple(qs[i]), tuple(qs[j]), color, thickness)
        cv2.line(img, tuple(qs[i+4]), tuple(qs[j+4]), color, thickness)
        cv2.line(img, tuple(qs[i]), tuple(qs[i+4]), color, thickness)
    return img

if __name__ == "__main__":
    # Example usage
    data_path = "/home/arka/DSGN/data/kitti/training/"
    output_path = "/home/arka/DSGN/outputs/dsgn_12g_b/kitti_output/"
    image_file = data_path+"image_2/003973.png"
    calib_file = data_path+"calib/003973.txt"
    detection_file = output_path+"003973.txt"

    img = cv2.imread(image_file)
    P2 = read_kitti_calib(calib_file)
    detections = read_kitti_detection(detection_file)

    for det in detections:
        if det["type"].lower() != "car":  # Change this to visualize other classes
            continue
        corners_3d = compute_box_3d(det["dimensions"], det["location"], det["rotation_y"])
        corners_2d = project_to_image(corners_3d, P2)
        img = draw_projected_box3d(img, corners_2d)

    cv2.imshow("3D Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()