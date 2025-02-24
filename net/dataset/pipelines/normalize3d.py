from mmengine.registry import TRANSFORMS
import numpy as np
import mmcv

@TRANSFORMS.register_module()
class Normalize3D(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=float)
        self.std = np.array(std, dtype=float)
        self.to_rgb = to_rgb

    def __call__(self, data_dict):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in data_dict.get('img_fields', ['img']):
            data_dict[key] = [mmcv.imnormalize(
                img, self.mean, self.std, self.to_rgb) for img in data_dict[key]]
        data_dict['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return data_dict
