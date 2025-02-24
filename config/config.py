roi_size = (60, 30)
num_points = 30
class2label = {
    'ped_crossing': 0,
    'divider': 1,
    'contours': 2,
    'others': -1,
    # 'centerline': 3,
}

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

canvas_size = (200, 100)
num_class = max(list(class2label.values()))+1
train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'), # 加载图片
    dict(type='ResizeMultiViewImages',          # resize
         size = (int(128*2), int((16/9*128)*2)), # H, W
         change_intrinsics=True,
         ),
    dict(
        type='VectorizeLocalMap',
        data_root='./datasets/nuScenes',
        patch_size=(roi_size[1],roi_size[0]),
        sample_pts=False,
        fixed_num={
            'ped_crossing': -1,
            'divider': -1,
            'contours': -1,
            'others': -1,
        },
        max_len=30,
        normalize=True,
        class2label = class2label 
    ),
    dict(
        type='PolygonizeLocalMapBbox',
        canvas_size=canvas_size,  # xy
        coord_dim=2,
        mode='xyxy',
        threshold=4/200,
    ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
]
# train_pipeline =None
input_modality = dict(
    use_lidar=True,
    use_camera=True
    )

dataset = dict(
    train=dict(
        type='NuscenesDataset',
        data_root='./datasets/nuScenes',
        predeal_file='./datasets/nuScenes/nuscenes_map_infos_train.pkl',
        modality=input_modality,
        pipeline_dict=train_pipeline,
        interval=1,
    ),
    val=dict(
        type='NuscenesDataset',
        data_root='./datasets/nuScenes',
        predeal_file='./datasets/nuScenes/nuscenes_map_infos_val.pkl',
        modality=input_modality,
        pipeline_dict=train_pipeline,
        interval=1,
    )
)
