import os
import sys
import mmcv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon,MultiPolygon
from render_relative_map import render_relative_map,RenderDataset
import mmengine
from mmengine.registry import DATASETS
from mmengine.config import Config
work_dir = os.path.dirname(os.path.dirname(__file__))
if work_dir not in sys.path:
        sys.path.append(work_dir)
import net

def dataset_test():

    cfg = Config.fromfile("config/config.py")
    nuc_scenes = DATASETS.build(cfg["dataset"]["train"])
    # data0 = nuc_scenes[0]
    # geomes_dict0 = data0['geoms_dict']
    # render_relative_map(geomes_dict0)
    rd = RenderDataset()
    rd.render(nuc_scenes)

if __name__ == '__main__':
    # os.environ['DISPLAY']="192.168.93.1:0"
    print('\n\033[31m-----------------------START-----------------------\033[0m\n')
    dataset_test()
    print('\n\033[31m-----------------------END-----------------------\033[0m\n\n\n')
