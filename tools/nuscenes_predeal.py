import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import mmengine
import numpy as np
from pyquaternion import Quaternion
from os import path
# args 设置参数
def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    
    parser.add_argument(
        '-v','--version',
        choices=['v1.0-mini', 'v1.0-trainval'],
        default='v1.0-mini')
    
    args = parser.parse_args()
    return args

def predeal(version='v1.0-mini',dataroot='/myvectormapnet/datasets/nuScenes'):
    #加载数据集
    nusc  = NuScenes(version=version,dataroot=dataroot, verbose=True)
    # 划分训练集和验证集
    train_scenes = splits.mini_train
    val_scenes = splits.mini_val

    # 六个相机的key值
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT']
    #开始转换 
    train_samples, val_samples, test_samples = [], [], []
    for sample in mmengine.track_iter_progress(nusc.sample):
        # 读取sample中激光雷达外参
        lidar_sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calibration_data = nusc.get('calibrated_sensor',lidar_sample_data['calibrated_sensor_token'])
        # 读取sample中车身位姿
        ego_pose_record = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
        # 读取lidar文件路径 1.获取token，根据token查询路径
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # 获取场景scene 的token,根据token获取sence ,再从scene中读取 scene_name 
        scene_record = nusc.get('scene', sample['scene_token'])
        scene_name = scene_record['name']
        # 获取场景scene log 的token,再从log中读取位置location
        log_record = nusc.get('log', scene_record['log_token'])
        info = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'cams': {},
                'lidar2ego_translation': lidar_calibration_data['translation'],
                'lidar2ego_rotation': lidar_calibration_data['rotation'],
                'e2g_translation': ego_pose_record['translation'],
                'e2g_rotation': ego_pose_record['rotation'],
                'timestamp': sample['timestamp'],
                'location': log_record['location'],
                'scene_name': scene_name
        }
        # 加载 camera
        for cam in camera_types:
            # camera的token
            cam_token = sample['data'][cam]
            camera_sample_data = nusc.get('sample_data', cam_token)
            # 读取外参，和lidar差不多，但是这里要做一个逆变换？
            # TODO [?] 为什么做逆变换
            camera_calibration_data = nusc.get('calibrated_sensor', camera_sample_data['calibrated_sensor_token'])
            cam2ego_rotation = Quaternion(camera_calibration_data['rotation']).rotation_matrix
            cam2ego_translation = np.array(camera_calibration_data['translation'])
        
            ego2cam_rotation = cam2ego_rotation.T
            ego2cam_translation = ego2cam_rotation.dot(-cam2ego_translation)       

            transform_matrix = np.eye(4) #ego2cam
            transform_matrix[:3, :3] = ego2cam_rotation
            transform_matrix[:3, 3] = ego2cam_translation
            camera_info = dict(
                extrinsics=transform_matrix, # ego2cam
                intrinsics=camera_calibration_data['camera_intrinsic'],
                img_fpath=str(nusc.get_sample_data_path(camera_sample_data['token']))
            )
            info['cams'][cam] = camera_info
        if scene_name in train_scenes:
            train_samples.append(info)
        elif scene_name in val_scenes:
            val_samples.append(info)
        
    # 记录下这些数据
    # for training set
    info_path = path.join(dataroot, 'nuscenes_map_infos_train.pkl')
    print(f'saving training set to {info_path}')
    mmengine.dump(train_samples, info_path)
    # for val set
    info_path = path.join(dataroot, 'nuscenes_map_infos_val.pkl')
    print(f'saving validation set to {info_path}')
    mmengine.dump(val_samples, info_path)


if __name__ == '__main__':
    args = parse_args()
    predeal(dataroot=args.data_root, version=args.version)
