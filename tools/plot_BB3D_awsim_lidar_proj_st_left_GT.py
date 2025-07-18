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
                "bbox": [float(x) for x in fields[4:8]],
                "dimensions": [float(x) for x in fields[8:11]],
                "location": [float(x) for x in fields[11:14]],
                "rotation_y": float(fields[14]) if len(fields) > 14 else None
            }
            detections.append(obj)
    return detections

def compute_box_3d(dim, loc, ry):
    """Returns 3D box corners in camera coordinate system."""
    h, w, l = dim
    x, y, z = loc
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.array([x_corners, y_corners, z_corners])
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

def compute_2d_bbox(corners_3d_proj):
    x_coords = corners_3d_proj[0]
    y_coords = corners_3d_proj[1]
    xmin = np.min(x_coords)
    ymin = np.min(y_coords)
    xmax = np.max(x_coords)
    ymax = np.max(y_coords)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]

def compute_alpha(loc, rotation_y):
    x, _, z = loc
    alpha = rotation_y - np.arctan2(x, z)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return float(alpha)

def draw_projected_box3d(img, qs, color=(0,255,0), thickness=2):
    """Draw 3D bounding box in image."""
    qs = qs.astype(np.int32).T
    for k in range(0, 4):
        i, j = k, (k+1)%4
        cv2.line(img, tuple(qs[i]), tuple(qs[j]), color, thickness)
        cv2.line(img, tuple(qs[i+4]), tuple(qs[j+4]), color, thickness)
        cv2.line(img, tuple(qs[i]), tuple(qs[i+4]), color, thickness)
    return img

def draw_2d_bbox(image, bbox):
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
    return image

def get_sorted_file_list(folder, ext=".png"):
    return sorted([f for f in os.listdir(folder) if f.endswith(ext)])

if __name__ == "__main__":
    data_path = "/home/arka/ros2_ws/src/awsim_to_kitti/awsim_dataset2"
    image_folder = os.path.join(data_path, "image_2")
    label_folder = os.path.join(data_path, "label_2")

    BL_to_Cam_T = np.array([[-0.036, -0.999, 0.001, 0.031],
                            [-0.015, -0.000, -1.000, 1.913],
                            [0.999, -0.036, -0.015, -0.019],
                            [0.000, 0.000, 0.000, 1.000]])

    P2 = np.array([[960.0, 0.0, 960.5, 259.2],
                   [0.0, 959.39, 540.5, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])

    image_files = get_sorted_file_list(image_folder, ext=".png")

    for file_name in image_files:
        image_path = os.path.join(image_folder, file_name)
        label_path = os.path.join(label_folder, file_name.replace(".png", ".txt"))

        img = cv2.imread(image_path)
        detections = read_kitti_detection(label_path)

        for det in detections:
            corners_3d = compute_box_3d(det["dimensions"], det["location"], det["rotation_y"])
            corners_2d = project_to_image(corners_3d, P2)
            bbox_2d = compute_2d_bbox(corners_2d)
            alpha = compute_alpha(det["location"], det["rotation_y"])

            # Draw projected center point as blue circle
            loc_3d = np.array(det["location"]).reshape(3, 1)
            loc_3d_hom = np.vstack((loc_3d, np.ones((1, 1))))
            loc_2d = np.dot(P2, loc_3d_hom)
            loc_2d = (loc_2d[:2] / loc_2d[2]).astype(int)
            cv2.circle(img, (loc_2d[0, 0], loc_2d[1, 0]), radius=5, color=(255, 0, 0), thickness=-1)

            img = draw_projected_box3d(img, corners_2d)
            img = draw_2d_bbox(img, bbox_2d)

            print(f"File: {file_name} | Alpha: {alpha:.2f} | 2D Bounding Box: {bbox_2d}")

        cv2.imshow("3D Boxes", img)
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
