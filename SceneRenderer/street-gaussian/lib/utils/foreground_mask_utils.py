import os
import sys
import numpy
import cv2
import numpy as np

def get_foreground_mask(foreground_img_path,conbined_img,output_path):
    foreground_img = cv2.imread(foreground_img_path)
    composition_img = cv2.imread(conbined_img)
    foreground_img_hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
    composition_img_hsv = cv2.cvtColor(composition_img, cv2.COLOR_BGR2HSV)
    if np.all(foreground_img == 255):
        mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=np.uint8)
        cv2.imwrite(output_path, mask)
        return
    mask = np.zeros((foreground_img.shape[0], foreground_img.shape[1], 4), dtype=np.uint8)
    mask[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) < 10] = np.concatenate((foreground_img, np.ones((foreground_img.shape[0], foreground_img.shape[1], 1), dtype=np.uint8) * 255), axis=2)[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) < 10]
    mask[np.linalg.norm(foreground_img_hsv - composition_img_hsv, axis=2) >= 10] = [0, 0, 0, 0]
    mask[np.all(foreground_img == 255, axis=2)] = [0, 0, 0, 0]
    cv2.imwrite(output_path, mask)

def get_foreground_mask_in_folder(folder):
    frame_list = [x.split('_')[0] for x in os.listdir(folder) if x.endswith('.png')]
    cam_id_list = [x.split('_')[1] for x in os.listdir(folder) if x.endswith('.png')]
    frame_list = list(set(frame_list))
    cam_id_list = list(set(cam_id_list))
    from joblib import Parallel, delayed
    Parallel(n_jobs=4)(delayed(get_foreground_mask)(os.path.join(folder, frame + '_' + cam_id + '_rgb_obj.png'), os.path.join(folder, frame + '_' + cam_id + '_rgb.png'), os.path.join(folder, frame + '_' + cam_id + '_rgb_obj_mask.png')) for frame in frame_list for cam_id in cam_id_list)
