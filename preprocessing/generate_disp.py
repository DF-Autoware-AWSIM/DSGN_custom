import argparse
import os

import numpy as np
import scipy.misc as ssc

import kitti_util
import imageio
import cv2

DEPTH_AS_DISP = True

def generate_dispariy_from_velo(pc_velo, height, width, calib, depth_as_disp=False, baseline=0.54):
    #print('Height: {}, Width: {}'.format(height, width))
    #trans_3d = calib.project_velo_to_ref(pc_velo)
    #print(f"V2C Transformation:\n{calib.V2C}")
    # Save the transformed point cloud (trans_3d)
    #trans_3d_file = os.path.join(disparity_dir, predix + '_trans_3d.bin')
    #reflectance = np.ones((trans_3d.shape[0], 1), dtype=np.float32)
    #trans_3d_with_reflectance = np.hstack((trans_3d.astype(np.float32), reflectance))
    #trans_3d_with_reflectance.tofile(trans_3d_file)
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 1] < -2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for pt in imgfov_pts_2d:
        x, y = pt
        if 0 <= x < width and 0 <= y < height:
            img[y, x] = (0, 255, 0)  # green point
    #cv2.imshow("Projection Check", img)
    #cv2.waitKey(0)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    if depth_as_disp:
        return depth_map
    disp_map = (calib.f_u * baseline) / depth_map
    return disp_map

def rotate_pc_by_theta(pc, theta):
    """Rotate point cloud by theta around the z-axis."""
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return np.dot(pc, R.T)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    parser.add_argument('--split_file', type=str, default='~/Kitti/object/train.txt')
    parser.add_argument('--right_calib', action='store_true', default=False)
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    lidar_dir = args.data_path + '/velodyne/'
    calib_dir = args.data_path + '/calib/'
    image_dir = args.data_path + '/image_2/'
    if DEPTH_AS_DISP:
        disparity_dir = args.data_path + '/depth/'
    else:
        disparity_dir = args.data_path + '/disparity/'

    assert os.path.isdir(lidar_dir)
    assert os.path.isdir(calib_dir)
    assert os.path.isdir(image_dir)

    if not os.path.isdir(disparity_dir):
        os.makedirs(disparity_dir)

    lidar_files = [x for x in os.listdir(lidar_dir) if x[-3:] == 'bin']
    lidar_files = sorted(lidar_files)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]
    
    for fn in lidar_files:
        predix = fn[:-4]
        #print('File name {}'.format(file_names))
        #print('Processing {}'.format(predix))
        if predix not in file_names:
            print('Skip {}'.format(predix))
            #continue
        calib_file = '{}/{}.txt'.format(calib_dir, predix)
        calib = kitti_util.Calibration(calib_file, right_calib=args.right_calib)
        # load point cloud
        lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
        #lidar = rotate_pc_by_theta(lidar, np.pi)
        image_file = '{}/{}.png'.format(image_dir, predix)
        image = imageio.imread(image_file)
        height, width = image.shape[:2]
        print('calib baseline {}'.format(calib.baseline))
        disp = generate_dispariy_from_velo(lidar, height, width, calib, depth_as_disp=DEPTH_AS_DISP, baseline=calib.baseline)
        np.save(disparity_dir + '/' + predix + ('_r' if args.right_calib else ''), disp)
        print('Finish Disparity {}'.format(predix + ('_r' if args.right_calib else '')))
