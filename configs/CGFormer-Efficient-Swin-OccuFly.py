import os

def _env_int_list(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return [int(x.strip()) for x in raw.split(',') if x.strip()]

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_root = os.environ.get('OCCUFLY_DATASET_ROOT', 'repos/datasets/OccuFly_Dataset')
depth_root = os.environ.get('OCCUFLY_DEPTH_ROOT', os.path.join(os.path.dirname(data_root.rstrip(os.sep)), 'OccuFly_Predicted_DepthMaps'))
vjepa_depth_prior_teacher_root = os.environ.get('AERO_VJEPA_DEPTH_PRIOR_TEACHER_ROOT')
ckpt_root = os.environ.get('CGFORMER_CKPT_ROOT', os.path.join(_repo_root, 'data', 'checkpoints'))

camera_used = ['visual']
dataset_type = 'OccuFlyDataset'
point_cloud_range = [-48.0, -32.0, 0.0, 48.0, 32.0, 64.0]
occ_size = [192, 128, 128]

# Contiguous training ids. Raw OccuFly ids are remapped in OccuFlyDataset.
class_names = [
    'empty',
    'road', 'walkway', 'dirt', 'gravel', 'rock', 'grass', 'vegetation', 'tree',
    'ground_obstacle', 'person', 'bicycle', 'vehicle', 'water', 'building', 'roof',
    'cable', 'cable_tower', 'parking_lot', 'construction', 'crane', 'truck',
]
num_class = len(class_names)

# OccuFly non-empty class frequencies from dataset_notes.md, scaled to positive counts
# for CGFormer's 1 / log(freq + 0.001) class weighting.
occufly_class_frequencies = [
    1.0e9,
    1.8909e6, 2.0610e6, 2.3584e6, 1.4511e6, 0.0402e6,
    8.5614e6, 4.3121e6, 7.5479e6, 1.9605e6, 0.0001e6,
    0.0035e6, 0.5683e6, 1.7539e6, 62.1534e6, 2.2018e6,
    0.0018e6, 0.0047e6, 2.8415e6, 0.2741e6, 0.0059e6, 0.1105e6,
]

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0,
)

data_config = {
    'input_size': tuple(int(x) for x in os.environ.get('OCCUFLY_INPUT_SIZE', '512,768').split(',')),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.0,
}

train_pipeline = [
    dict(type='LoadOccuFlyImageFromFiles', data_config=data_config, load_stereo_depth=True,
         load_gt_depth=True, is_train=True, color_jitter=(0.4, 0.4, 0.4), clip_depth_max=128.0),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
         is_train=True, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ', 'vjepa_teacher', 'target_vjepa_teacher',
                                   'vjepa_depth_prior_teacher', 'vjepa_depth_prior_teacher_confidence'],
         meta_keys=['pc_range', 'occ_size', 'scene', 'altitude', 'sequence', 'frame_id',
                    'source_pose', 'target_pose', 'target_cam_intrinsic', 'target_frame_id',
                    'source_image_size', 'target_image_size',
                    'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape',
                    'gt_depths', 'vjepa_feature_path', 'target_vjepa_feature_path',
                    'vjepa_depth_prior_teacher_path']),
]

test_pipeline = [
    dict(type='LoadOccuFlyImageFromFiles', data_config=data_config, load_stereo_depth=True,
         load_gt_depth=True, is_train=False, color_jitter=None, clip_depth_max=128.0),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
         is_train=False, point_cloud_range=point_cloud_range),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ', 'vjepa_teacher', 'target_vjepa_teacher',
                                   'vjepa_depth_prior_teacher', 'vjepa_depth_prior_teacher_confidence'],
         meta_keys=['pc_range', 'occ_size', 'scene', 'altitude', 'sequence', 'frame_id',
                    'source_pose', 'target_pose', 'target_cam_intrinsic', 'target_frame_id',
                    'source_image_size', 'target_image_size',
                    'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape',
                    'gt_depths', 'vjepa_feature_path', 'target_vjepa_feature_path',
                    'vjepa_depth_prior_teacher_path']),
]

trainset_config = dict(
    type=dataset_type,
    data_root=data_root,
    depth_root=depth_root,
    gt_depth_root=data_root,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    altitudes=[30, 40, 50],
    test_mode=False,
    remap_labels=True,
    require_depth=True,
    vjepa_feature_root=os.environ.get('OCCUFLY_VJEPA_FEATURE_ROOT'),
    paired_training=os.environ.get('AERO_USE_LATENT_VIEW', '0') == '1' or os.environ.get('AERO_USE_OUTPUT_CONSISTENCY', '0') == '1',
    pair_frame_offset=int(os.environ.get('OCCUFLY_PAIR_FRAME_OFFSET', '1')),
    pair_search_radius=int(os.environ.get('OCCUFLY_PAIR_SEARCH_RADIUS', '5')),
    pair_max_translation_m=float(os.environ.get('OCCUFLY_PAIR_MAX_TRANSLATION_M', '30.0')),
    pair_require_projection=os.environ.get('OCCUFLY_PAIR_REQUIRE_PROJECTION', '1') == '1',
    pair_min_projected_voxels=int(os.environ.get('OCCUFLY_PAIR_MIN_PROJECTED_VOXELS', '1')),
    vjepa_depth_prior_teacher_root=vjepa_depth_prior_teacher_root,
    require_vjepa_depth_prior_teacher=os.environ.get('AERO_USE_VJEPA_DEPTH_PRIOR_TEACHER', '0') == '1',
)

valset_config = dict(
    type=dataset_type,
    data_root=data_root,
    depth_root=depth_root,
    gt_depth_root=data_root,
    pipeline=test_pipeline,
    split='val',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    altitudes=[30, 40, 50],
    test_mode=True,
    remap_labels=True,
    require_depth=True,
    vjepa_feature_root=os.environ.get('OCCUFLY_VJEPA_FEATURE_ROOT'),
    vjepa_depth_prior_teacher_root=vjepa_depth_prior_teacher_root,
    require_vjepa_depth_prior_teacher=False,
)

testset_config = dict(
    type=dataset_type,
    data_root=data_root,
    depth_root=depth_root,
    gt_depth_root=data_root,
    pipeline=test_pipeline,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    altitudes=[30, 40, 50],
    test_mode=True,
    remap_labels=True,
    require_depth=True,
    vjepa_feature_root=os.environ.get('OCCUFLY_VJEPA_FEATURE_ROOT'),
    vjepa_depth_prior_teacher_root=vjepa_depth_prior_teacher_root,
    require_vjepa_depth_prior_teacher=False,
)

data = dict(train=trainset_config, val=valset_config, test=testset_config)

train_dataloader_config = dict(
    batch_size=int(os.environ.get('CGFORMER_BATCH_SIZE_PER_GPU', '1')),
    num_workers=int(os.environ.get('CGFORMER_NUM_WORKERS', '8')),
    shuffle=os.environ.get('CGFORMER_TRAIN_SHUFFLE', '1') != '0',
)
test_dataloader_config = dict(batch_size=1, num_workers=int(os.environ.get('CGFORMER_NUM_WORKERS', '8')))

numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
volume_h = occ_size[0] // lss_downsample[0]
volume_w = occ_size[1] // lss_downsample[1]
volume_z = occ_size[2] // lss_downsample[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [0.5, 64.5, 0.5],
}

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = _dim_ // 2
_num_layers_self_ = 2
_num_points_self_ = 8

model = dict(
    type='CGFormer',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone',
                      checkpoint=os.path.join(ckpt_root, 'efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth')),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=34 if os.environ.get('AERO_USE_INMODEL_ALTITUDE_FRUSTUM_LIFTING', '0') == '1' else 33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
        altitude_conditioning=os.environ.get('AERO_USE_INMODEL_ALTITUDE_FRUSTUM_LIFTING', '0') == '1',
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=point_cloud_range,
        input_dimensions=[volume_h, volume_w, volume_z],
        data_config=data_config,
        init_cfg=None,
    ),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=volume_h,
        volume_w=volume_w,
        volume_z=volume_z,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        cross_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams=_num_cams_,
           encoder=dict(
               type='VoxFormerEncoder_DFA3D',
               num_layers=_num_layers_cross_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[dict(
                       type='DeformCrossAttention_DFA3D',
                       pc_range=point_cloud_range,
                       num_cams=_num_cams_,
                       deformable_attention=dict(
                           type='MSDeformableAttention3D_DFA3D',
                           embed_dims=_dim_,
                           num_points=_num_points_cross_,
                           num_levels=_num_levels_),
                       embed_dims=_dim_,
                   )],
                   ffn_cfgs=dict(
                       type='FFN', embed_dims=_dim_, feedforward_channels=1024,
                       num_fcs=2, ffn_drop=0., act_cfg=dict(type='ReLU', inplace=True)),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams=_num_cams_,
           use_level_embeds=False,
           use_cams_embeds=False,
           encoder=dict(
               type='VoxFormerEncoder',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[dict(type='DeformSelfAttention', embed_dims=_dim_, num_levels=1, num_points=_num_points_self_)],
                   ffn_cfgs=dict(type='FFN', embed_dims=_dim_, feedforward_channels=1024, num_fcs=2,
                                 ffn_drop=0., act_cfg=dict(type='ReLU', inplace=True)),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(type='LearnedPositionalEncoding', num_feats=_pos_dim_, row_num_embed=1024, col_num_embed=1024),
        mlp_prior=True,
    ),
    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        global_aggregator=dict(
            type='TPVGlobalAggregator',
            embed_dims=_dim_,
            split=[8, 8, 8],
            grid_size=[volume_h, volume_w, volume_z],
            global_encoder_backbone=dict(
                type='Swin',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                in_channels=128,
                patch_size=4,
                strides=[1, 2, 2, 2],
                frozen_stages=-1,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained', checkpoint=os.path.join(ckpt_root, 'swin_tiny_patch4_window7_224.pth'))),
            global_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True, track_running_stats=False),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False)),
        ),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(type='CustomResNet3D', numC_input=128, num_layer=[2, 2, 2], num_channels=[128, 128, 128], stride=[1, 2, 2]),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear', align_corners=False)),
        ),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg={
            'loss_voxel_ce_weight': 1.0,
            'loss_voxel_sem_scal_weight': 1.0,
            'loss_voxel_geo_scal_weight': 1.0,
        },
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=occufly_class_frequencies,
    ),
    aero_jepa=dict(
        use_vjepa_distill=os.environ.get('AERO_USE_VJEPA_DISTILL', '0') == '1',
        use_latent_view=os.environ.get('AERO_USE_LATENT_VIEW', '0') == '1',
        use_masked_3d=os.environ.get('AERO_USE_MASKED_3D', '0') == '1',
        use_output_consistency=os.environ.get('AERO_USE_OUTPUT_CONSISTENCY', '0') == '1',
        lambda_vjepa_distill=float(os.environ.get('AERO_LAMBDA_VJEPA_DISTILL', '0.0')),
        lambda_latent_view=float(os.environ.get('AERO_LAMBDA_LATENT_VIEW', '0.0')),
        lambda_masked_3d=float(os.environ.get('AERO_LAMBDA_MASKED_3D', '0.0')),
        lambda_output_consistency=float(os.environ.get('AERO_LAMBDA_OUTPUT_CONSISTENCY', '0.0')),
        require_vjepa_teacher=os.environ.get('AERO_REQUIRE_VJEPA_TEACHER', '0') == '1',
        vjepa_teacher_dim=int(os.environ.get('AERO_VJEPA_TEACHER_DIM', '1664')),
        context_channels=128,
        masked_3d_channels=128,
        m4_mask_type=os.environ.get('AERO_M4_MASK_TYPE', 'random'),
        m4_total_mask_ratio=float(os.environ.get('AERO_M4_TOTAL_MASK_RATIO', '0.20')),
        m4_max_mask_ratio=float(os.environ.get('AERO_M4_MAX_MASK_RATIO', '0.50')),
        m4_min_masked_voxels=int(os.environ.get('AERO_M4_MIN_MASKED_VOXELS', '1')),
        m4_random_ratio=float(os.environ.get('AERO_M4_RANDOM_RATIO', '0.15')),
        m4_block_num_min=int(os.environ.get('AERO_M4_BLOCK_NUM_MIN', '2')),
        m4_block_num_max=int(os.environ.get('AERO_M4_BLOCK_NUM_MAX', '8')),
        m4_block_size_min=_env_int_list('AERO_M4_BLOCK_SIZE_MIN', [3, 3, 2]),
        m4_block_size_max=_env_int_list('AERO_M4_BLOCK_SIZE_MAX', [12, 12, 6]),
        m4_column_num_min=int(os.environ.get('AERO_M4_COLUMN_NUM_MIN', '2')),
        m4_column_num_max=int(os.environ.get('AERO_M4_COLUMN_NUM_MAX', '8')),
        m4_column_size_min=_env_int_list('AERO_M4_COLUMN_SIZE_MIN', [2, 2]),
        m4_column_size_max=_env_int_list('AERO_M4_COLUMN_SIZE_MAX', [10, 10]),
        m4_frustum_num_min=int(os.environ.get('AERO_M4_FRUSTUM_NUM_MIN', '1')),
        m4_frustum_num_max=int(os.environ.get('AERO_M4_FRUSTUM_NUM_MAX', '4')),
        m4_frustum_angle_width_min=float(os.environ.get('AERO_M4_FRUSTUM_ANGLE_WIDTH_MIN', '5')),
        m4_frustum_angle_width_max=float(os.environ.get('AERO_M4_FRUSTUM_ANGLE_WIDTH_MAX', '25')),
        m4_frustum_depth_min=float(os.environ.get('AERO_M4_FRUSTUM_DEPTH_MIN', '0.2')),
        m4_frustum_depth_max=float(os.environ.get('AERO_M4_FRUSTUM_DEPTH_MAX', '1.0')),
        m4_mix_random_ratio=float(os.environ.get('AERO_M4_MIX_RANDOM_RATIO', '0.30')),
        m4_mix_block_ratio=float(os.environ.get('AERO_M4_MIX_BLOCK_RATIO', '0.30')),
        m4_mix_column_ratio=float(os.environ.get('AERO_M4_MIX_COLUMN_RATIO', '0.25')),
        m4_mix_frustum_ratio=float(os.environ.get('AERO_M4_MIX_FRUSTUM_RATIO', '0.15')),
        m4_ignore_index=int(os.environ.get('AERO_M4_IGNORE_INDEX', '255')),
        output_consistency_confidence=float(os.environ.get('AERO_OUTPUT_CONSISTENCY_CONFIDENCE', '0.6')),
        altitude_lifting_variant=os.environ.get('AERO_ALTITUDE_LIFTING_VARIANT', 'none'),
        use_gt_metric_depth_teacher=os.environ.get('AERO_USE_GT_METRIC_DEPTH_TEACHER', '0') == '1',
        use_vjepa_depth_prior_teacher=os.environ.get('AERO_USE_VJEPA_DEPTH_PRIOR_TEACHER', '0') == '1',
        teacher_used_at_inference=os.environ.get('AERO_TEACHER_USED_AT_INFERENCE', '0') == '1',
        lambda_depth_gt_kl=float(os.environ.get('AERO_LAMBDA_DEPTH_GT_KL', '0.0')),
        lambda_depth_gt_l1=float(os.environ.get('AERO_LAMBDA_DEPTH_GT_L1', '0.0')),
        lambda_depth_vjepa_kl=float(os.environ.get('AERO_LAMBDA_DEPTH_VJEPA_KL', '0.0')),
        lambda_depth_vjepa_l1=float(os.environ.get('AERO_LAMBDA_DEPTH_VJEPA_L1', '0.0')),
        teacher_depth_soft_target=os.environ.get('AERO_TEACHER_DEPTH_SOFT_TARGET', 'gaussian_log_depth'),
        teacher_depth_sigma_log=float(os.environ.get('AERO_TEACHER_DEPTH_SIGMA_LOG', '0.10')),
    ),
)

learning_rate = float(os.environ.get('CGFORMER_LR', '3e-4'))
training_steps = int(os.environ.get('CGFORMER_TRAINING_STEPS', '25000'))
optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=float(os.environ.get('CGFORMER_WEIGHT_DECAY', '0.01')),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=float(os.environ.get('AERO_BACKBONE_LR_MULT', '0.3')), decay_mult=1.0),
            'depth_net': dict(lr_mult=1.0, decay_mult=1.0),
            'img_view_transformer': dict(lr_mult=1.0, decay_mult=1.0),
            'vjepa_projector': dict(lr_mult=3.0, decay_mult=1.0),
            'latent_view_projector': dict(lr_mult=3.0, decay_mult=1.0),
            'masked_3d_head': dict(lr_mult=3.0, decay_mult=1.0),
            'bn': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
        }
    ),
)
lr_scheduler = dict(type='OneCycleLR', max_lr=learning_rate, total_steps=training_steps + 10,
                    pct_start=0.05, cycle_momentum=False, anneal_strategy='cos', interval='step', frequency=1)
grad_clip = dict(max_norm=1.0, norm_type=2)
load_from = os.environ.get('CGFORMER_LOAD_FROM', os.path.join(ckpt_root, 'efficientnet-seg-depth.pth'))
