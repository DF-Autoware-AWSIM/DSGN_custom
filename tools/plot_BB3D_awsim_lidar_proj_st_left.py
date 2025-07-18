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
   

    '''
    obj1 = {
            "type": "car",# lidar_proj_test_st_left_2.png
            "dimensions": [4.0686869621276855, 1.8655112981796265, 1.4836885929107666],  # l, w, h
            "location": [9.966870307922363, -1.4518797397613525, 0.7802734375],  # x, y, z
            "rotation_y": 0.1195761 
        }'''
    '''
    obj1 = {
            "type": "car",# lidar_proj_test_st_left_3.png
            "dimensions": [4.936614036560059, 2.0851969718933105, 1.9181616306304932],  # l, w, h
            "location": [9.755932807922363, 4.982495307922363, 0.88037109375],  # x, y, z
            "rotation_y": 0.1542599 
        }'''
    
    '''
    obj1 = {
            "type": "car",# lidar_proj_test_st_left_4.png
            "dimensions": [9.304226875305176, 2.826565742492676, 3.4801323413848877],  # l, w, h
            "location": [9.36624526977539, -4.066879749298096, 1.74609375],  # x, y, z
            "rotation_y": 0.2206828 
        }
    '''
    '''
    obj1 = {
            "type": "car",# lidar_proj_test_st_left_5.png
            "dimensions": [8.020745277404785, 2.8710780143737793, 3.44294810295105],  # l, w, h
            "location": [11.285932540893555, 4.259994983673096, 1.5771484375],  # x, y, z
            "rotation_y": 3.1130638 
        }'''
    '''
    obj1 = {
            "type": "car",# lidar_proj_test_st_left_6.png
            "dimensions": [8.588212966918945, 2.6984455585479736, 2.7963647842407227],  # l, w, h
            "location": [17.6328067779541, -6.164067268371582, 1.4169921875],  # x, y, z
            "rotation_y": -2.9973021 
        }'''

    obj1 = {
            "type": "car",# reverse-T_1.png
            "dimensions": [4.403597831726074, 2.0669500827789307, 1.727001667022705],  # l, w, h
            "location": [10.357728958129883, 0.42624521255493164, 0.87744140625],  # x, y, z
            "rotation_y": -0.0786049 
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
    l, w, h = np.abs(new_dim[0]), np.abs(new_dim[1]), np.abs(new_dim[2])  # Unpack rotated dimensions
    print(f"New Dimensions: {l,w,h}, Original Dimensions: {dim}")
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
    print(f"3D Corners in Camera Coordinates:\n{corners_3d}")
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
    """Draw 3d bounding box in image"""
    qs = qs.astype(np.int32).T
    # 0-3: lower corners, 4-7: upper corners
    for k in range(0,4):
        i,j = k,(k+1)%4
        cv2.line(img, tuple(qs[i]), tuple(qs[j]), color, thickness)
        cv2.line(img, tuple(qs[i+4]), tuple(qs[j+4]), color, thickness)
        cv2.line(img, tuple(qs[i]), tuple(qs[i+4]), color, thickness)
    return img

def draw_project_center(detections, image, P):  
    """
    Draw the center of the 3D bounding boxes on the image.
    detections: List of detections with 'location' key.
    image: The image on which to draw.
    P: Projection matrix.
    """
    for det in detections:
        loc = np.array(det["location"]) 
        loc = np.reshape(loc, (3, 1))  # Add homogeneous coordinate
        #print(loc.shape)
        #loc_cam = np.dot(BL_to_Cam_T, loc)[:3]  # Apply transformation and drop homogeneous coordinate
        center_2d = project_to_image(loc, P)
        print(f"Projected Center: {center_2d}")
        u, v = int(center_2d[0, 0]), int(center_2d[1, 0])
        cv2.circle(image, (u, v), 5, (255, 0, 0), -1)  # Draw center in blue
    return image

def draw_2d_bbox(image, bbox):
    """
    Draw a red 2D bounding box
    bbox: [xmin, ymin, xmax, ymax]
    """
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
    return image

if __name__ == "__main__":
    # Example usage
    data_path = "/home/arka/DSGN/tools/awsim_proj_data/"
    #output_path = "/home/arka/DSGN/outputs/dsgn_12g_b_2/awsim_output/"
    image_file = data_path+"/reverse_T_1.png"
    #calib_file = data_path+"calib/000105.txt"
    #detection_file = output_path+"000105.txt"

    img = cv2.imread(image_file)
    #Transformation between camera0/camera_optical_link and base_link
    #ros2 run tf2_ros tf2_echo camera0/camera_optical_link base_link
    BL_to_Cam_T = np.array([[-0.036, -0.999, 0.001, 0.031],
                 [-0.015, -0.000, -1.000, 1.913],
                 [0.999, -0.036, -0.015, -0.919],
                 [0.000, 0.000, 0.000, 1.000]])
    #ros2 run tf2_ros tf2_echo base_link camera0/camera_optical_link
    '''BL_to_Cam_T = np.array([[-0.036, -0.015, 0.999, 0.948],
                 [-0.999, -0.000, -0.036, -0.002],
                 [0.001, -1.000, -0.015, 1.899],
                 [0.000, 0.000, 0.000, 1.000]])'''
    '''
    BL_to_Cam_T = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])'''
    P2 = np.array([[960.0, 0.000000e+00, 960.5, 259.20001220703125],
                  [0.000000e+00, 959.3908081054688, 540.5, 0.000000e+00],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
    #P2 = read_kitti_calib(calib_file)
    #detections = read_kitti_detection(detection_file)
    detections = read_lidar_detection()
    detection_in_cam = convert_lidar_to_camera(detections, BL_to_Cam_T)
    draw_project_center(detection_in_cam, img, P2)
    for det in detection_in_cam:
        if det["type"].lower() != "car":  # Change this to visualize other classes
            continue
        corners_3d = compute_box_3d_autoware(det["dimensions"], det["location"], det["rotation_y"], BL_to_Cam_T)
        corners_2d = project_to_image(corners_3d, P2)
        bbox_2d = compute_2d_bbox(corners_2d)
        alpha = compute_alpha(det["location"], det["rotation_y"])
        img = draw_projected_box3d(img, corners_2d)
        img = draw_2d_bbox(img, bbox_2d)
        print(f" Alpha: {alpha:.2f}")
        print(f" 2D Bounding Box: {bbox_2d}")

    cv2.imshow("3D Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
