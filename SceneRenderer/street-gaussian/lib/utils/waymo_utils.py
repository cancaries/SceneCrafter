import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm 
from lib.config import cfg
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from lib.utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from lib.utils.data_utils import get_val_frames
from lib.utils.graphics_utils import get_rays, sphere_intersection
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.datasets.base_readers import storePly, get_Sphere_Norm

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT',
}
image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        
    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)
    
    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:
        
        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)
    
    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]
    
    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses

# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])
    
    return obj_pose_vehicle, obj_pose_world

def make_agent_pose(ego_pose, loc, rot):
    rot_z = rot[2]
    c = math.cos(rot_z)
    s = math.sin(rot_z)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rot_matrix
    obj_pose_vehicle[:3, 3] = np.array(loc)
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world

def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location[0] - current_location[0], target_location[1] - current_location[1]])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(orientation), math.sin(orientation)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []    
    objects_info = {}

    if cfg.data.get('use_tracker', False):
        tracklet_path = os.path.join(datadir, 'track/track_info_castrack.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis_castrack.json')
    else:
        tracklet_path = os.path.join(datadir, 'track/track_info.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')
    scene_number = cfg.scene_number

    if scene_number is None:
        agent_dir = os.path.join(datadir, 'car_info/car_dict_sequence_500.json')
    else: 
        agent_dir = os.path.join(datadir, f'car_info/car_dict_sequence_{str(scene_number).zfill(3)}.json')

    print(f'Loading from : {tracklet_path}')
    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]
    
    f = open(tracklet_camera_vis_path, 'r')
    tracklet_camera_vis = json.load(f)

    with open(agent_dir, 'r') as agent_file:
        data = json.load(agent_file)
    
    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_cameras = 5
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    max_frame = len(data)
    more_frame_num = max(n_frames, max_frame)
    n_obj_in_frame = np.zeros(more_frame_num)
    max_track_id = 0

    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]
        
        if max_track_id < track_id:
            max_track_id = track_id

        if object_class in ['sign', 'misc']:
            continue
        
        cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))
        if len(join_cameras_list) == 0:
            continue
                
        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]
            objects_info[track_id]['height'] = float(tracklet[4])
            objects_info[track_id]['width'] = float(tracklet[5])
            objects_info[track_id]['length'] = float(tracklet[6])
            objects_info[track_id]['save'] = 0
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))
            
        tr_array = np.concatenate(
            [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[4:]).astype(np.float64)]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1
    
    agents_info = {}
    frame_vehicle_data = {}
    agents_id = []

    agent_id = max_track_id + 1
    
    for frame_id, frame in enumerate(data):
        frame_vehicle_data[frame_id] = {}
        ego_data={}
        for agent_name, agent_data in frame.items():
            ego_data['rot'] = frame['ego_vehicle']['rot'][2]
            ego_data['trans'] = np.array(frame['ego_vehicle']['loc']).astype(np.float64)
            if agent_name == "ego_vehicle":
                pass
            elif "static_agent" in agent_name:
                continue
            else:
                loc = np.array(agent_data['loc']).astype(np.float64)
                height = float(agent_data['bbox'][0])
                width = float(agent_data['bbox'][1])
                length = float(agent_data['bbox'][2])
                rot = float(agent_data['rot'][2])
                
                n_obj_in_frame[frame_id] += 1
                if agent_name not in agents_info and agent_name != "ego_vehicle":
                    agents_info[agent_name] = {
                        'id': agent_id,
                        'class': 'vehicle',
                        'class_label': 0,
                        'height' : height,
                        'width' :width,
                        'length' : length,
                        'loc' : loc,
                        'rot': rot,
                        'start_frame': frame_id,
                        'end_frame': frame_id,
                        'type' : agent_data['type']
                    }
                    agents_id.append(agent_id)
                    agent_id +=1
                frame_vehicle_data[frame_id][agents_info[agent_name]['id']] = dict()
                frame_vehicle_data[frame_id][agents_info[agent_name]['id']]['loc'] = loc
                frame_vehicle_data[frame_id][agents_info[agent_name]['id']]['rot'] = rot
                
                if is_within_distance(loc, ego_data['trans'], ego_data['rot'], 100, 120, d_angle_th_low=0):
                    frame_vehicle_data[frame_id][agents_info[agent_name]['id']]['visible'] = 1
                else:
                    frame_vehicle_data[frame_id][agents_info[agent_name]['id']]['visible'] = 1
                agent_array = np.concatenate(
                    [
                        np.array([frame_id]), 
                        np.array([agents_info[agent_name]['id']]), 
                        np.array([type]), 
                        np.array([height]), 
                        np.array([width]), 
                        np.array([length]), 
                        loc, 
                        np.array([rot])
                    ]
                )
                tracklets_ls.append(agent_array)
                if agents_info[agent_name]['end_frame'] < frame_id:
                    agents_info[agent_name]['end_frame'] = n_frames - 1

    for agent_name, agent_info in agents_info.items():
        track_id = agent_info['id']
        if track_id not in objects_info:
            objects_info[track_id] = {
                'track_id': track_id,
                'class': agent_info['class'],
                'class_label': agent_info['class_label'],
                'height': agent_info['height'],
                'width': agent_info['width'],
                'length': agent_info['length'],
                'save': 1,
                'start_frame': agent_info['start_frame'],
                'end_frame': agent_info['end_frame'],
                'type': agent_info['type']
            }
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], agent_info['height'])
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], agent_info['width'])
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], agent_info['length'])
            
    tracklets_array = np.array(tracklets_ls,dtype=object)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    
    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        # print(tracklet)
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:            
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world
  
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic and objects_info[key]['save'] == 0:
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)
            
    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0    
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)
    
    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info   
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False
        
        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale
        
        if 'start_frame' not in obj.keys():
            obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
            obj_frame_idx = obj_frame_idx.astype(np.int32)
            obj_frames = frames[obj_frame_idx]
            obj['start_frame'] = np.min(obj_frames)
            obj['end_frame'] = np.max(obj_frames)
        
        objects_info[key] = obj
     
    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )
    
    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )
    return objects_tracklets_world, objects_tracklets_vehicle, objects_info, agents_id, frame_vehicle_data

def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]
    
    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)
    
    if max_timestamp > frame_timestamps[1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)
        
    return tracklets, frame_timestamps
    
def generate_dataparser_outputs(
        datadir, 
        selected_frames=None, 
        build_pointcloud=True, 
        cameras=[0, 1, 2, 3, 4]
    ):
    

    
    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.png')))
    num_frames_all = len(image_filenames_all) // 5
    num_cameras = len(cameras)
    
    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)    

    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []
    
    ixts = []
    exts = []
    poses = []
    c2ws = []
    
    frames_timestamps = []
    cams_timestamps = []
        
    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )
    
    timestamp_path = os.path.join(datadir, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)
        
    for frame in range(start_frame, end_frame+1):
        frames_timestamps.append(timestamps['FRAME'][f'{frame:06d}'])
    
    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)
            
            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)
            
            camera_name = _label2camera[cam]
            timestamp = timestamps[camera_name][f'{frame:06d}']
            cams_timestamps.append(timestamp)
        
    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(), frames_timestamps.max())
 
    _, object_tracklets_vehicle, object_info, agents_id, frame_vehicle_data = get_obj_pose_tracking(
        datadir, 
        selected_frames, 
        ego_frame_poses,
        cameras,
    )

    light_direction_path = os.path.join(datadir, 'light_direction.json')
    with open(light_direction_path, 'r') as f:
        light_direction = json.load(f)
    
    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = timestamps['FRAME'][f'{object_start_frame:06d}'] - timestamp_offset - 0.1
        object_end_timestamp = timestamps['FRAME'][f'{object_end_frame:06d}'] - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)
    
    # print(frames_timestamps)
    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info 
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps
    
    result['agents_id'] = agents_id
    result['frame_vehicle_data'] = frame_vehicle_data

    result['light_direction'] = light_direction

    # get object bounding mask
    obj_bounds = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)    
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height], 
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext), 
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds         
    
    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from scripts.waymo.colmap_waymo_full import run_colmap_waymo
        run_colmap_waymo(result)
    
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)
        
        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.
                     
        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
        pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

        for i, frame in tqdm(enumerate(range(start_frame, end_frame+1))):
            idxs = list(range(i * num_cameras, (i+1) * num_cameras))
            cams_frame = [cams[idx] for idx in idxs]
            image_filenames_frame = [image_filenames[idx] for idx in idxs]
            
            raw_3d = pts3d_dict[frame]
            raw_2d = pts2d_dict[frame]
            
            # use the first projection camera
            points_camera_all = raw_2d[..., 0]
            points_projw_all = raw_2d[..., 1]
            points_projh_all = raw_2d[..., 2]

            # each point should be observed by at least one camera in camera lists
            mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)
            
            # get filtered LiDAR pointcloud position and color        
            points_xyz_vehicle = raw_3d[mask]

            # transfrom LiDAR pointcloud from vehicle frame to world frame
            ego_pose = ego_frame_poses[frame]
            points_xyz_vehicle = np.concatenate(
                [points_xyz_vehicle, 
                np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
            )
            points_xyz_world = points_xyz_vehicle @ ego_pose.T
            
            points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
            points_camera = points_camera_all[mask]
            points_projw = points_projw_all[mask]
            points_projh = points_projh_all[mask]

            for cam, image_filename in zip(cams_frame, image_filenames_frame):
                mask_cam = (points_camera == cam)
                image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.

                mask_projw = points_projw[mask_cam]
                mask_projh = points_projh[mask_cam]
                mask_rgb = image[mask_projh, mask_projw]
                points_rgb[mask_cam] = mask_rgb
        
            # filer points in tracking bbox
            points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

            for tracklet in object_tracklets_vehicle[i]:
                track_id = int(tracklet[0])
                if track_id >= 0:
                    obj_pose_vehicle = np.eye(4)                    
                    obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                    obj_pose_vehicle[:3, 3] = tracklet[1:4]
                    vehicle2local = np.linalg.inv(obj_pose_vehicle)
                    
                    points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                    points_xyz_obj = points_xyz_obj[..., :3]
                    
                    length = object_info[track_id]['length']
                    width = object_info[track_id]['width']
                    height = object_info[track_id]['height']
                    bbox = [[-length/2, -width/2, -height/2], [length/2, width/2, height/2]]
                    obj_corners_3d_local = bbox_to_corner3d(bbox)
                    
                    points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                    points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                    points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                    points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])
                    

            points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
            points_lidar_rgb = points_rgb[~points_xyz_obj_mask]
            
            points_xyz_dict['bkgd'].append(points_lidar_xyz)
            points_rgb_dict['bkgd'].append(points_lidar_rgb)
            
        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)                                
                elif k.startswith('obj'):
                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]
                        
                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb
                
                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
        try:
            if cfg.data.filter_colmap:
                points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                for i, ext in enumerate(exts):
                    # if frames_idx[i] not in train_frames:
                    #     continue
                    camera_position = c2ws[i][:3, 3]
                    radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                    mask = np.logical_or(radius < cfg.data.get('extent', 10), points_colmap_xyz[:, 2] < camera_position[2])
                    points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)        
                points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                points_colmap_rgb = points_colmap_rgb[points_colmap_mask]
            
            points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
            mask = points_colmap_dist < 2 * sphere_radius
            points_colmap_xyz = points_colmap_xyz[mask]
            points_colmap_rgb = points_colmap_rgb[mask]
        
            points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0) 
            points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
        except:
            print('No colmap pointcloud')
            points_bkgd_xyz = points_lidar_xyz
            points_bkgd_rgb = points_lidar_rgb
        
        points_xyz_dict['lidar'] = points_lidar_xyz
        points_rgb_dict['lidar'] = points_lidar_rgb
        points_xyz_dict['colmap'] = points_colmap_xyz
        points_rgb_dict['colmap'] = points_colmap_rgb
        points_xyz_dict['bkgd'] = points_bkgd_xyz
        points_rgb_dict['bkgd'] = points_bkgd_rgb
            
        result['points_xyz_dict'] = points_xyz_dict
        result['points_rgb_dict'] = points_rgb_dict
        
        for k in points_xyz_dict.keys():
            points_xyz = points_xyz_dict[k]
            points_rgb = points_rgb_dict[k]
            ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
            try:
                storePly(ply_path, points_xyz, points_rgb)
                print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
            except:
                print(f'failed to save pointcloud for {k}')
                continue
    return result
