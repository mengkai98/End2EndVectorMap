from mmengine.registry import TRANSFORMS
@TRANSFORMS.register_module()
class PolygonizeLocalMapBbox(object):
        def __init__(self,
                 canvas_size=(400, 200),
                 coord_dim=2,
                 num_class=3,
                 mode='xyxy',
                 centerline_mode='xyxy',
                 num_point=10,
                 threshold=6/200,
                 debug=False,
                 test_mode=False,
                 flatten=True,
                 ):
                pass
        def Polygonization(self, data_dict: dict):
               pass
        def __call__(self, data_dict: dict):

            data_dict = self.Polygonization(data_dict)
            return data_dict