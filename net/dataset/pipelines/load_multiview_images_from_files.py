from mmengine.registry import TRANSFORMS
import mmcv
import numpy as np

@TRANSFORMS.register_module()
class LoadMultiViewImagesFromFiles:

    def __call__(self, data_dict):
        filename = data_dict['img_filenames']
        img = [mmcv.imread(name) for name in filename]

        data_dict['img'] = img
        # 为shape 赋初值
        data_dict['img_shape'] = [i.shape for i in img]
        data_dict['ori_shape'] = [i.shape for i in img]
        data_dict['pad_shape'] = [i.shape for i in img]
        # 通道数
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]

        return data_dict
    