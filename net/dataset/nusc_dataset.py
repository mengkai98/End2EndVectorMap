from torch.utils.data import Dataset
from mmengine.registry import DATASETS
import mmengine
from mmengine.dataset import Compose
import numpy as np
@DATASETS.register_module()
class NuscenesDataset(Dataset):
    
    def __init__(self,data_root:str,predeal_file:str,modality:dict,pipeline_dict:dict,interval:int):
        """初始化数据集

        Args:
            dataroot (str): 数据集的根目录
            predeal_file (str): 预处理的文件
            modality (dict): 传感器的模态
            pipelines (dict): 数据集处理pipeline
            interval (int): 数据间隔
        """
        super().__init__()
        self.modality = modality
        self.samples = mmengine.load(predeal_file)[::interval]
        if pipeline_dict is not None:
            self.pipeline = Compose(pipeline_dict)
        else:
            self.pipeline = None
        
    def __getitem__(self, idx):
        sample =self.samples[idx]
        sample_token = sample['token']
        location = sample['location']
        #加载外参 3d空间点到像素平面
        ego2img_rts = []
        for c in sample['cams'].values():
            # 读取内外参
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            #内参×外参，外参把世界系下的点变换到相机系下，再用内参把相机系下的点投影到像素平面
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        data_dict = {
            # for nuscenes, the order is
            # 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            # 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            'sample_idx':sample_token,
            'location': location,
            # file  path
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks 内参
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam** 外参
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            # 世界系到像素平面
            'ego2img': ego2img_rts,
            # 'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            # 车身位置
            'ego2global_translation': sample['e2g_translation'],
            'ego2global_rotation': sample['e2g_rotation'],
        }
        # 使能lidar 则 更新进去lidar 的文件path
        if self.modality['use_lidar']:
            data_dict.update(
                dict(
                    pts_filename=sample['lidar_path'],
                )
            )
        # 配置 需要的数据
        if self.pipeline != None:
            data_dict = self.pipeline(data_dict)
        return data_dict

    def __len__(self):
        return len(self.samples)
