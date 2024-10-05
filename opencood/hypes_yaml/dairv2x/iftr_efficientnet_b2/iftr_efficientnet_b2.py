plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 1]
voxel_size = [0.4, 0.4, 4]

class_names = ['car']
num_classes = len(class_names)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 256
bev_w_ = 256
num_query = 600
queue_length = 4        # each sequence contains `queue_length` frames.

max_cav = 2
Ncams = 1
num_cams = max_cav * Ncams

model = dict(
    type='IFTR',
    use_grid_mask=True,
    order='hwl',                        # 必须和数据集后处理中的参数保持一致
    pool='avg',                         # 得到 use_2d_det_as_query 的方式: AvgPooling vs MaxPooling. 可选: avg or other
    embed_dims=_dim_,
    video_test_mode=True,
    img_backbone=dict(
        type='Efficientnet',
        out_c=_dim_,
        downsample=8,
        scale="b2",
        frozen_block=-1,
        frozen_neck=False),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=_dim_,
        num_depth_layers=1,
        loss_weight=0.2,
    ),
    pts_bbox_head=dict(
        type='IFTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=num_query,
        num_classes=num_classes,
        in_channels=_dim_,
        code_size=8,            # (cx, cy, w, l, cz, h, sin(theta), cos(theta), vx, vy)
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # the loss weights of code_size
        sync_cls_avg_factor=False,          # 分布训练时, 是否同步损失
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            num_cams=num_cams,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='IFTREncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='IFA',
                    attn_cfgs=[
                        dict(
                            type='SelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='MVFA',
                            embed_dims=_dim_,
                            num_cams=num_cams,
                            pc_range=point_cloud_range,
                            use_key_padding_mask=True,         # 是否只对前景进行交互
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=150,
            voxel_size=voxel_size,
            num_classes=num_classes,
            score_threshold=0.05),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[256, 256, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=1,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range
        )
    )
    )
)
