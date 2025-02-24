from mmengine.registry import TRANSFORMS
import numpy as np
from shapely.geometry import LineString
@TRANSFORMS.register_module()
class PolygonizeLocalMapBbox(object):
    def __init__(self, 
                mode,
                threshold,
            canvas_size=(400, 200),
            coord_dim=2,
            # num_class=3,

            # centerline_mode='xyxy',
            # num_point=10,

            # debug=False,
            # test_mode=False,
            # flatten=True,
            ):
        self.coord_dim = coord_dim
        self.canvas_size = canvas_size
        self.mode = mode
        self.threshold = threshold


    def evaluate_line(self,polyline):

        edge = np.linalg.norm(polyline[1:] - polyline[:-1], axis=-1) # 沿着行计算每行的norm 即 两点之间的距离

        # 起始 和 终点
        start_end_weight = edge[(0, -1), ].copy()
        
        mid_weight = (edge[:-1] + edge[1:]) * .5

        pts_weight = np.concatenate(
            (start_end_weight[:1], mid_weight, start_end_weight[-1:]))

        denominator = pts_weight.sum()
        denominator = 1 if denominator == 0 else denominator

        pts_weight /= denominator

        # add weights for stop index
        pts_weight = np.repeat(pts_weight, 2)/2
        pts_weight = np.pad(pts_weight, ((0, 1)),
                            constant_values=1/(len(polyline)*2))

        return pts_weight
    
    def quantize_verts(self,
            verts,
            canvas_size=(200, 100),
            coord_dim=2,
    ):
        """将 [0,1]范围内的点转换为canvas大小。

        Args:
            verts (_type_): _description_
            canvas_size (tuple, optional): _description_. Defaults to (200, 100).
            coord_dim (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: 整数的点的坐标
        """
        min_range = 0
        max_range = 1
        range_quantize = np.array(canvas_size) - 1  # (0-199) = 200
        

        verts_ratio = (verts - min_range) / (
            max_range - min_range)
        verts_quantize = verts_ratio * range_quantize[:coord_dim]
        return verts_quantize.astype('int32')



    def format_polyline_map(self, vectors):

        polylines, polyline_masks, polyline_weights = [], [], []

        # quantilize each label's lines individually.
        for vector_data in vectors:

            polyline, valid_len, label = vector_data

            # and pad polyline.
            if label == 2: # contours 轮廓线
                polyline_weight = self.evaluate_line(polyline).reshape(-1)
            else: # 车道线和人行横道
                # polyline_weight 是一个两行，valid_len 列的数组
                polyline_weight = np.ones_like(polyline).reshape(-1)
                # 左边不pad，右边pad 1 
                polyline_weight = np.pad(
                    polyline_weight, ((0, 1),), constant_values=1.)
                polyline_weight = polyline_weight/polyline_weight.sum()

            #flatten and quantilized
            fpolyline = self.quantize_verts(
                polyline, self.canvas_size, self.coord_dim)
            # 转化成1维向量
            fpolyline = fpolyline.reshape(-1) 
            # reindex starting from 1, and add a zero stopping token(EOS),
            # 坐标里可能会出现0，但是不允许有0, 因为想把0作为停止符，因此所有的索引都从1开始
            fpolyline = np.pad(fpolyline + 1, ((0, 1),),constant_values= 0)
            # mask 
            fpolyline_msk = np.ones(fpolyline.shape, dtype=bool)
            polyline_masks.append(fpolyline_msk)
            polyline_weights.append(polyline_weight)
            # 记录了整形的 点序列，坐标从1开始，0表示这个序列的结束
        polylines.append(fpolyline)
        polyline_map = polylines
        polyline_map_mask = polyline_masks
        polyline_map_weights = polyline_weights

        return polyline_map, polyline_map_mask, polyline_map_weights
    def get_bbox(self,polyline_nd, threshold=6 ):
        '''
            polyline: seq_len, coord_dim
        '''
        polyline = LineString(polyline_nd)
        bbox = polyline.bounds
        minx, miny, maxx, maxy = bbox
        W, H = maxx-minx, maxy-miny

        if W < threshold or H < threshold:
            remain = (threshold - min(W, H))/2
            bbox = polyline.buffer(remain).envelope.bounds
            minx, miny, maxx, maxy = bbox

        bbox_np = np.array([[minx, miny], [maxx, maxy]])
        bbox_np = np.clip(bbox_np, 0., 1.)


        return bbox_np


    def format_keypoint(self, vectors,):

        kps, kp_labels = [], []
        qkps, qkp_masks = [], []

        for vector_data in vectors:

            polyline, valid_len, label = vector_data

            kp = self.get_bbox(polyline, self.threshold)
            kps.append(kp)
            kp_labels.append(label)

            # flatten and quantilized
            fkp = self.quantize_verts(kp, self.canvas_size, self.coord_dim)
            fkp = fkp.reshape(-1)

            # Reindex starting from 1, and add a class token,

            fkps_msk = np.ones(fkp.shape, dtype=bool)
            qkp_masks.append(fkps_msk)
            qkps.append(fkp)

        qkps = np.stack(qkps)
        qkp_msks = np.stack(qkp_masks)

        # format det
        kps = np.stack(kps, axis=0).astype(float)*self.canvas_size
        kp_labels = np.array(kp_labels)
        # restrict the boundary
        kps[..., 0] = np.clip(kps[..., 0], 0.1, self.canvas_size[0]-0.1)
        kps[..., 1] = np.clip(kps[..., 1], 0.1, self.canvas_size[1]-0.1)

        # nbox, boxsize(4)*coord_dim(2)
        kps = kps.reshape(kps.shape[0], -1)  # n*2
        # unflatten_seq(qkps)

        return kps, kp_labels, qkps, qkp_msks,


    def Polygonization(self, data_dict: dict):
        # vector 顶点 长度 label
        vectors = data_dict.pop('vectors')
        if not len(vectors):
            data_dict['polys'] = []
            return data_dict
        polyline_map, polyline_map_mask, polyline_map_weight = \
            self.format_polyline_map(vectors)
        keypoint, keypoint_label, qkeypoint, qkeypoint_mask = \
            self.format_keypoint(vectors)
        polys = {
            # for det
            'keypoint': keypoint,
            'det_label': keypoint_label,

            # for gen
            'gen_label': keypoint_label,
            'qkeypoint': qkeypoint,
            'qkeypoint_mask': qkeypoint_mask,

            # nlines(nbox) List[ seq_len*coord_dim ]
            'polylines': polyline_map,  # List[np.array]
            'polyline_masks': polyline_map_mask,  # List[np.array]
            'polyline_weights': polyline_map_weight,
        }

        # Format outputs
        data_dict['polys'] = polys
        return data_dict
    
    def __call__(self, data_dict: dict):

        data_dict = self.Polygonization(data_dict)
        return data_dict