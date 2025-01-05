from mmengine.registry import TRANSFORMS
import mmcv
import numpy as np

 
@TRANSFORMS.register_module()
class ResizeMultiViewImages:
    def __init__(self,size,change_intrinsics):
        self.size = size 
        self.change_intrinsics = change_intrinsics
    def __call__(self, data_dict):
        # resize后的图片 resize后的内参 和世界系到像素平面的变换
        new_imgs, post_intrinsics, post_ego2imgs = [], [], []
        for img,  cam_intrinsic, ego2img in zip(data_dict['img'], data_dict['cam_intrinsics'], data_dict['ego2img']):
            tmp, scaleW, scaleH = mmcv.imresize(img,
                                                # mmcv.imresize expect (w, h) shape
                                                (self.size[1], self.size[0]),
                                                return_scale=True)
            new_imgs.append(tmp)
            # 缩放变换矩阵
            rot_resize_matrix = np.array([
                [scaleW, 0,      0,    0],
                [0,      scaleH, 0,    0],
                [0,      0,      1,    0],
                [0,      0,      0,    1]])
            # 新内参
            post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
            # 新投影
            post_ego2img = rot_resize_matrix @ ego2img
            post_intrinsics.append(post_intrinsic)
            post_ego2imgs.append(post_ego2img)
        data_dict['img'] = new_imgs
        data_dict['img_shape'] = [img.shape for img in new_imgs]
        if self.change_intrinsics:
            data_dict.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs,
            })
        return data_dict
