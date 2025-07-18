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

def read_lidar_detection():
    """Read a sample LiDAR detection file and return a list of detections."""
    # This is a placeholder function. Replace with actual LiDAR detection reading logic.
    detections = []
    '''obj1 = {
            "type": "car",
            "dimensions": [4.301588535308838, 1.9172251224517822, 1.415054440498352],  # l, w, h
            "location": [11.138276100158691, -1.7212547063827515, 0.68994140625],  # x, y, z
            "rotation_y": -2.8338977 #Minus because the direction is now flipped
        }'''
    '''obj1 = {
            "type": "car",# lidar_proj_test_scene2_2.png
            "dimensions": [4.438135623931885, 1.920973539352417, 1.434533715248108],  # l, w, h
            "location": [14.346869468688965, -6.608754634857178, 0.5009765625],  # x, y, z
            "rotation_y": 3.0967009 
        }'''
    '''obj1 = {
            "type": "car",# llidar_proj_test5.png
            "dimensions": [4.284818172454834, 1.8838183879852295, 1.4226752519607544],  # l, w, h
            "location": [11.586557388305664, -1.718598484992981, 0.7001953125],  # x, y, z
            "rotation_y": -2.95 
        }'''
    '''obj1 = {
            "type": "car",# lidar_proj_test_scene2_3.png
            "dimensions": [5.038232135772705, 1.936039686203003, 1.48007071018219],  # l, w, h
            "location": [22.537494659423828, 4.3530421257019043, 0.454833984375],  # x, y, z
            "rotation_y": 0.9165871 
        }'''

    obj1 = {
            "type": "car",# lidar_proj_test_scene2_1_re.png
            "dimensions": [5.083380699157715, 2.131521463394165, 1.8792248964309692],  # l, w, h
            "location": [7.967963695526123, -3.4081296920776367, 1.0205078125],  # x, y, z
            "rotation_y": -0.0158489 
        }
    detections.append(obj1)
    obj2 = {
            "type": "car",
            "dimensions": [4.955935478210449, 2.1252858638763428, 1.8902682065963745],  # l, w, h
            "location": [6.821245193481445, -6.733598232269287, 0.80126953125],  # x, y, z
            "rotation_y": -2.8588
        }
    #detections.append(obj2)
    obj3 = {
            "type": "car",
            "dimensions": [4.092596530914307, 1.8321107625961304, 1.399935007095337],  # l, w, h
            "location": [8.564994812011719, 1.7687451839447021, 0.5986328125],  # x, y, z
            "rotation_y": -2.8696998
        }
    #detections.append(obj3)

    return detections
        
    

def convert_lidar_to_camera(detections, BL_to_Cam_T):
    """Convert LiDAR detections to camera coordinate system."""
    detection_in_cam = []
    for det in detections:
        # Convert location from LiDAR to camera coordinates
        loc = np.array(det["location"] + [1.0])  # Add homogeneous coordinate
        loc_cam = np.dot(BL_to_Cam_T, loc)[:3]  # Apply transformation and drop homogeneous coordinate
        det_cam = {
            "type": det["type"],
            "dimensions": det["dimensions"],
            "location": loc_cam.tolist(),
            "rotation_y": det["rotation_y"]
        }
        detection_in_cam.append(det_cam)
    return detection_in_cam

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

def compute_box_3d_autoware(dim, loc, ry, BL_to_Cam_T):
    """
    Returns 3D box corners in coordinate system assuming the origin is at 
    the center of the bounding box (as in Autoware).
    
    Args:
        dim: tuple of (h, w, l) [height, width, length]
        loc: tuple of (x, y, z) [center position in 3D space]
        ry: rotation around Y-axis (in radians)

    Returns:
        corners_3d: (3, 8) array of 3D corner points
    """
    l_, w_, h_ = dim
    #w, m_l, m_h = dim
    #h = -m_h
    #l = -m_l
    x, y, z = loc
    R_new = BL_to_Cam_T[:3, :3]  # Extract rotation part from transformation matrix
    # Apply rotation to the dimensions
    new_dim = np.dot(R_new, np.array([l_, w_, h_]))
    l, w, h = new_dim[0], new_dim[1], new_dim[2]  # Unpack rotated dimensions
    # 3D bounding box corners centered at origin (geometric center)
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    corners = np.array([x_corners, y_corners, z_corners])

    # Rotation matrix around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Apply rotation then translation
    corners_3d = R @ corners
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
    data_path = "/home/arka/DSGN/tools/awsim_proj_data/"
    #output_path = "/home/arka/DSGN/outputs/dsgn_12g_b_2/awsim_output/"
    image_file = data_path+"/lidar_proj_test_scene2_1_re.png"
    #calib_file = data_path+"calib/000105.txt"
    #detection_file = output_path+"000105.txt"

    img = cv2.imread(image_file)
    #Transformation between traffic_light_left_camera/camera_optical_link and base_link
    BL_to_Cam_T = np.array([[-0.036, -0.999, 0.001, 0.048],
                 [-0.015, -0.000, -1.000, 1.913],
                 [0.999, -0.036, -0.015, -0.919],
                 [0.000, 0.000, 0.000, 1.000]])
    '''
    BL_to_Cam_T = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])'''
    P2 = np.array([[960.0, 0.000000e+00, 960.5, 0.000000e+00],
                  [0.000000e+00, 959.3908081054688, 540.5, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
    #P2 = read_kitti_calib(calib_file)
    #detections = read_kitti_detection(detection_file)
    detections = read_lidar_detection()
    detection_in_cam = convert_lidar_to_camera(detections, BL_to_Cam_T)
    for det in detection_in_cam:
        if det["type"].lower() != "car":  # Change this to visualize other classes
            continue
        corners_3d = compute_box_3d_autoware(det["dimensions"], det["location"], det["rotation_y"], BL_to_Cam_T)
        corners_2d = project_to_image(corners_3d, P2)
        img = draw_projected_box3d(img, corners_2d)

    cv2.imshow("3D Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
