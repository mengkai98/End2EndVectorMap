from mmengine.registry import TRANSFORMS
import mmcv
import numpy as  np

@TRANSFORMS.register_module()
class PadMultiViewImages(object):
    """Pad multi-view images and change intrinsics
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Changed keys is 'cam_intrinsics', 

    Args:
        size (tuple, optional): Fixed padding size, (h, w).
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0, change_intrinsics=False):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

        self.change_intrinsics = change_intrinsics

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        original_shape = [img.shape for img in results['img']]

        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = [mmcv.impad(
                    img, shape=self.size, pad_val=self.pad_val) for img in results[key]]
            elif self.size_divisor is not None:
                padded_img = [mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val) for img in results[key]]


            results[key] = padded_img

        if self.change_intrinsics:
            post_intrinsics, post_ego2imgs = [], []
            for img, oshape, cam_intrinsic, ego2img in zip(results['img'], original_shape, results['cam_intrinsics'], results['ego2img']):
                scaleW = img.shape[1] / oshape[1]
                scaleH = img.shape[0] / oshape[0]
  

                rot_resize_matrix = np.array([
                    [scaleW, 0,      0,    0],
                    [0,      scaleH, 0,    0],
                    [0,      0,      1,    0],
                    [0,      0,      0,    1]])

                post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
                post_ego2img = rot_resize_matrix @ ego2img
                post_intrinsics.append(post_intrinsic)
                post_ego2imgs.append(post_ego2img)

            results.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs,
            })

        results['img_shape'] = [img.shape for img in padded_img]
        results['img_fixed_size'] = self.size
        results['img_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


