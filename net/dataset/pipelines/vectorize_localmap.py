from mmengine.registry import TRANSFORMS
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from shapely import ops
from shapely.geometry import MultiPolygon
from shapely.strtree import STRtree
from shapely.geometry import LineString, MultiPolygon,box
@TRANSFORMS.register_module()
class VectorizeLocalMap(object):
    def __init__(self,data_root:str,patch_size,
                 fixed_num,
                 normalize= True,
                 padding = True,
                 max_len=30,
                 sample_pts = False,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment','lane'],
                class2label={
                     'ped_crossing': 0,
                     'divider': 1,
                     'contours': 2,
                     'others': -1,
                 }, 
                 ):
        
        self.class2label = class2label

        self.normalize = normalize
        self.padding = padding
        self.sample_pts = sample_pts
        self.patch_size = patch_size
        self.size = np.array([self.patch_size[1], self.patch_size[0]]) + 2
        self.max_len  =max_len
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
            'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.contour_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorers = {}
        self.process_func = {
            'ped_crossing': self.ped_geoms_to_vectors,
            'divider': self.line_geoms_to_vectors,
            'contours': self.poly_geoms_to_vectors,
            'centerline': self.line_geoms_to_vectors,
        }
        self.layer2class = {
            'ped_crossing': 'ped_crossing',
            'lane_divider': 'divider',
            'road_divider': 'divider',
            'road_segment': 'contours',
            'lane': 'contours',
        }
        self.fixed_num = fixed_num
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            # self.map_explorers[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

    def normalize_line(self, line):
        '''
            prevent extrime pts such as 0 or 1. 
        '''

        origin = -np.array([self.patch_size[1]/2, self.patch_size[0]/2])
        # for better learning
        line = line - origin
        line = line / self.size

        return line

    def ped_geoms_to_vectors(self, geoms: list):
        """_summary_

        Args:
            geoms (list): _description_

        Returns:
            _type_: _description_
        """
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for geom in geoms:
            for ped_poly in geom:
                # rect = ped_poly.minimum_rotated_rectangle
                ext = ped_poly.exterior
                # 不是逆时针方向，转换成逆时针
                if not ext.is_ccw:
                    ext.coords = list(ext.coords)[::-1]
                # 只要这个区域内的图形，取它俩的交集
                lines = ext.intersection(local_patch)
                # 把多边形转换成线
                if lines.geom_type != 'LineString':
                    lines = ops.linemerge(lines)

                # same instance but not connected.
                # 转换的操作和这个差不多
                if lines.geom_type != 'LineString':
                    ls = []
                    for l in lines.geoms:
                        ls.append(np.array(l.coords))

                    lines = np.concatenate(ls, axis=0)
                    lines = LineString(lines)

                results.append(lines)

        return results
    def line_geoms_to_vectors(self, geom):
        # XXX
        return geom
    def poly_geoms_to_vectors(self, polygon_geoms: list):

        results = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []

        for geom in polygon_geoms:
            for poly in geom:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        # 外围轮廓
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # since the start and end will disjoint
            # after applying the intersection.
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)
        # 内部轮廓
        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        return results
    def _geoms2pts(self, line, label, fixed_point_num):

        # if we still use the fix point
        if fixed_point_num > 0:
            remain_points = fixed_point_num - np.asarray(line.coords).shape[0]
            if remain_points < 0:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > fixed_point_num:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

                remain_points = fixed_point_num - \
                    np.asarray(line.coords).shape[0]
                if remain_points > 0:
                    line = self.pad_line_with_interpolated_line(
                        line, remain_points)

            elif remain_points > 0:

                line = self.pad_line_with_interpolated_line(
                    line, remain_points)

            v = line
            if not isinstance(v, np.ndarray):
                v = np.asarray(line.coords)

            valid_len = v.shape[0]
            print(1)
        elif self.padding:  # dynamic points

            if self.max_len < np.asarray(line.coords).shape[0]:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > self.max_len:
                    # 以0.2公差tolerance 一步步简化近似图形直到满足长度要求
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

            v = np.asarray(line.coords)
            valid_len = v.shape[0]

            pad_len = self.max_len - valid_len
            # 缺少的就填为0，这里应该是为了和网络对齐
            v = np.pad(v, ((0, pad_len), (0, 0)), 'constant')

        else:
            # dynamic points without padding
            line = line.simplify(0.2, preserve_topology=True)
            v = np.array(line.coords)
            valid_len = len(v)

        if self.normalize:
            # 归一化处理 normal 的方式很怪
            v = self.normalize_line(v)

        return v, valid_len
    
    def _geom_to_vectors(self, line_geom, label, vector_len, sample_pts=False):
        '''
            transfrom the geo type 2 line vectors
        '''
        line_vectors = {'vectors': [], 'length': []}
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    # l.geom_type == LineString
                    for l in line:
                        if sample_pts:
                            v, nl = self._sample_pts_from_line(
                                l, label, vector_len)
                        else:
                            v, nl = self._geoms2pts(l, label, vector_len)
                        line_vectors['vectors'].append(v.astype(float))
                        line_vectors['length'].append(nl)
                elif line.geom_type == 'LineString':
                    if sample_pts:
                        v, nl = self._sample_pts_from_line(
                            line, label, vector_len)
                    else:
                        v, nl = self._geoms2pts(line, label, vector_len)
                    line_vectors['vectors'].append(v.astype(float))
                    line_vectors['length'].append(nl)
                else:
                    raise NotImplementedError

        return line_vectors

    def get_global_patch(self, data_dict: dict):
        """以当前车辆为中心, 获取一个地图块 (map_path)

        Args:
            input_dict (dict): _description_

        Returns:
            _type_: _description_
        """
        # 地点,新加坡 etc。。。。
        location = data_dict['location']
        # 车身的位置
        ego2global_translation = data_dict['ego2global_translation']
        ego2global_rotation = data_dict['ego2global_rotation']
        # 取xy作为地图块的中心，长宽是初始化时传入的参数
        map_pose = ego2global_translation[:2]
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        # 计算yaw角 单位deg
        rotation = Quaternion(ego2global_rotation)
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        patch_params = (patch_box, patch_angle, location)
        return patch_params
        
    def retrive_geom(self, patch_params):
        patch_box, patch_angle, location = patch_params
        geoms_dict = {}
        # 拼接class，每一个class对应map中的一个layer
        layers = \
            self.line_classes + self.ped_crossing_classes + \
            self.contour_classes
        # 使用set去重
        layers = set(layers)
        geos = self.nusc_maps[location].get_map_geom(patch_box, patch_angle,layers)
        geoms_dict = dict()
        for geo in geos:
            geoms_dict[geo[0]] = geo[1]
            # geoms_dict[layer_name] = geoms

        return geoms_dict
    
    def union_ped(self, ped_geoms):

        def get_rec_direction(geom):
            # 计算这个多边形的最小包围矩形
            rect = geom.minimum_rotated_rectangle
            # 取矩形前三个顶点abc
            rect_v_p = np.array(rect.exterior.coords)[:3]
            # 计算顶点 a-b,b-c
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            # norm ,即可算出哪一个是长边
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()
            #返回长边的方向以及长度
            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        for i in range(len(final_pgeom)):
            if final_pgeom[i].geom_type != 'MultiPolygon':
                final_pgeom[i] = MultiPolygon([final_pgeom[i]])

        return final_pgeom

    def union_geoms(self, geoms_dict):
        """_summary_

        Args:
            geoms_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        customized_geoms_dict = {}

        #对图形进行合并 道路边界和车道的合并
        roads = geoms_dict['road_segment']
        lanes = geoms_dict['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        # 合并成轮廓线
        union_segments = ops.unary_union([union_roads, union_lanes])

        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        customized_geoms_dict['contours'] = ('contours', [union_segments, ])

        # ped
        #对人行横道合并
        geoms_dict['ped_crossing'] = self.union_ped(geoms_dict['ped_crossing'])
        # geoms_dict['ops_peds']=  [ops.unary_union(geoms_dict['ped_crossing'])]
        # 把剩余的特征一一赋值过来，这里面包含了合并过的人行道
        for layer_name, custom_class in self.layer2class.items():

            if custom_class == 'contours':
                continue
            
            customized_geoms_dict[layer_name] = (
                custom_class, geoms_dict[layer_name])
        # key:layer_name -> class:str,geometrys:list
        #layername: contours ped_crossing,lane_divider,road_divider
        return  customized_geoms_dict
    
    def convert2vec(self, geoms_dict: dict, sample_pts=False, override_veclen: int = None):

        vector_dict = {}
        for layer_name, (customized_class, geoms) in geoms_dict.items():
            
            # 全部处理成线
            line_strings = self.process_func[customized_class](geoms)

            vector_len = self.fixed_num[customized_class]
            if override_veclen is not None:
                vector_len = override_veclen
            ## 点和长度
            vectors = self._geom_to_vectors(
                line_strings, customized_class, vector_len, sample_pts)
            vector_dict.update({layer_name: (customized_class, vectors)})

        return vector_dict

    def vectorization(self, data_dict: dict):
        patch_params = self.get_global_patch(data_dict)
        geoms_dict = self.retrive_geom(patch_params)
        customized_geoms_dict = self.union_geoms(geoms_dict)
        data_dict['geoms_dict'] = geoms_dict
        vectors = []
        vectors_dict = self.convert2vec(customized_geoms_dict, self.sample_pts)
        # layername class vectors{point,len}
        for k, (custom_class, v) in vectors_dict.items():
            ## class name to int label
            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue
            # 合并成真值
            for vec, l in zip(v['vectors'], v['length']):

                vectors.append((vec, l, label))

        data_dict['vectors'] = vectors
        return data_dict

    def __call__(self, data_dict: dict):

        data_dict = self.vectorization(data_dict)

        return data_dict
    
