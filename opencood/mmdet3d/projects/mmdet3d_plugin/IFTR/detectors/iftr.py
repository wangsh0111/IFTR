# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Modified by Shaohong Wang
# ---------------------------------------------

import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..modules.grid_mask import GridMask
from opencood.utils import box_utils as box_utils


@DETECTORS.register_module()
class IFTR(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 order='hwl',
                 pool='avg',
                 embed_dims=256,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_backbone_resume=None,
                 depth_branch=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(IFTR, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head, img_roi_head,
            img_rpn_head, train_cfg, test_cfg, pretrained)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        self.pc_range = pts_bbox_head["transformer"]["encoder"]["pc_range"]

        self.order = order
        self.pool = pool

        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )
        self.view_cone_mlp = nn.Sequential(
            nn.Linear(9, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims)
        )

        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None

        if img_backbone_resume is not None:
            self.img_backbone.load_state_dict(
                torch.load(img_backbone_resume, map_location='cpu'), strict=False)
            print(f"img backbone weights is resumed from {img_backbone_resume}")

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points_2d', 'img_metas'))
    def view_cone_encoding(self, reference_points_2d, img_metas, predefine_depth=1):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        rots, trans, post_rots, post_trans, intrins, pairwise_t_matrix, record_len, img_shape = img_metas['rots'], \
            img_metas['trans'], img_metas['post_rots'], img_metas['post_trans'], img_metas['intrins'], \
            img_metas["pairwise_t_matrix"], img_metas["record_len"], img_metas['img_shape']

        B, num_cam, num_query, D = reference_points_2d.shape[:4]  # B, num_cam, num_query, D, 2(D=2)
        reference_points = reference_points_2d.permute(3, 0, 1, 2, 4)  # D, B, num_cam, num_query, 2(D=2)
        torch_ones = reference_points_2d.new_ones(D, B, num_cam, num_query, 1) * predefine_depth
        reference_points = torch.cat((reference_points, torch_ones), dim=-1)  # D, B, num_cam, num_query, 3(D=2)

        reference_points[..., 0] *= img_shape[1]
        reference_points[..., 1] *= img_shape[0]

        # undo post-transformation
        reference_points -= post_trans.view(1, B, num_cam, 1, 3).type_as(reference_points)
        reference_points = torch.inverse(post_rots).view(1, B, num_cam, 1, 3, 3).type_as(reference_points).matmul(
            reference_points.unsqueeze(-1)).squeeze(-1)

        # (u, v, d) --> (du, dv, d)
        reference_points = torch.cat((reference_points[:, :, :, :, :2] * reference_points[:, :, :, :, 2:3],
                                      reference_points[:, :, :, :, 2:3]), 4)  # D, B, num_cam, num_query, 3(D=2)

        # add optical center coordinates
        torch_zeros = reference_points_2d.new_zeros(1, B, num_cam, num_query, 3)
        reference_points = torch.cat((reference_points, torch_zeros), dim=0)  # D, B, num_cam, num_query, 3(D=3)
        D = reference_points.shape[0]

        # (du, dv, d) --> (x, y, z)
        combine = rots.matmul(torch.inverse(intrins)).type_as(reference_points)
        reference_points = combine.view(1, B, num_cam, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points += trans.view(1, B, num_cam, 1, 3).type_as(reference_points)
        reference_points = reference_points.permute(1, 2, 3, 0, 4)  # B, num_cam, num_query, D, 3(D=3)

        # regroup_reference_points
        B, L = pairwise_t_matrix.shape[:2]
        regroup_reference_points = reference_points.new_zeros(
            B, L, num_cam, num_query, D, 3)  # B, L, num_cam, num_query, D, 3(D=3)
        cum_sum_record_len = torch.cat(
            [record_len.new_zeros(1), torch.cumsum(record_len, dim=0)])

        for b in range(B):
            regroup_reference_points[b, :record_len[b], ...] = \
                reference_points[cum_sum_record_len[b]:cum_sum_record_len[b + 1]]  # B, L, num_cam, num_query, D, 3(D=3)
        reference_points = regroup_reference_points.permute(4, 0, 1, 2, 3, 5)  # D, B, L, num_cam, num_query, 3(D=3)
        reference_points = reference_points.transpose(-2, -1)  # D, B, L, num_cam, 3, num_query(D=3)
        torch_ones = reference_points.new_ones(*reference_points.shape[:4], 1, reference_points.shape[-1])
        reference_points = torch.cat((reference_points, torch_ones), dim=-2)  # D, B, L, num_cam, 4, num_query(D=3)

        # agent self coordinate system --> ego self coordinate system
        ego = 0  # Default 0 as ego
        transformation_matrix = pairwise_t_matrix[:, :, ego, :, :]  # B, L, L, 4, 4 --> B, L, 4, 4
        transformation_matrix = transformation_matrix.type_as(reference_points)  # float64 --> float32
        reference_points = transformation_matrix.view(1, B, L, 1, 4, 4).matmul(reference_points)
        reference_points = reference_points[:, :, :, :, :3, :].transpose(-2, -1)  # D, B, L, num_cam, num_query, 3
        reference_points = reference_points.permute(1, 2, 3, 4, 0, 5)  # B, L, num_cam, num_query, D, 3

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points

    # This function must use fp32!!!
    @force_fp32(apply_to=('object_bbx_center', 'object_bbx_mask', 'feats', 'img_metas'))
    def regroup_feats_and_mask(self, single_label_dict, feats, img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        rots, trans, post_rots, post_trans, intrins, pairwise_t_matrix, record_len, img_shape = img_metas['rots'], \
            img_metas['trans'], img_metas['post_rots'], img_metas['post_trans'], img_metas['intrins'], \
            img_metas["pairwise_t_matrix"], img_metas["record_len"], img_metas['img_shape']

        if 'object_2d_bbx_center_single' in single_label_dict:
            object_bbx_center = single_label_dict['object_2d_bbx_center_single'].float()
            object_bbx_mask = single_label_dict['object_2d_bbx_mask_single']

            B, num_cam, num_box, _ = object_bbx_center.shape
            reference_points = object_bbx_center.view(B, num_cam, num_box, 2, 2).permute(3, 0, 1, 2, 4)
            reference_points_mask = object_bbx_mask.view(B, num_cam, num_box, 1, 1).repeat(1, 1, 1, 2, 1).\
                permute(3, 0, 1, 2, 4).type(torch.bool)

            D, B, num_cam, num_query, _ = reference_points.size()

            torch_ones = reference_points.new_ones(D, B, num_cam, num_query, 1)
            reference_points = torch.cat((reference_points, torch_ones), dim=-1)

        else:
            object_bbx_center = single_label_dict['object_bbx_center_single']
            object_bbx_mask = single_label_dict['object_bbx_mask_single']

            bbx_num = torch.sum(object_bbx_mask, dim=1, dtype=torch.int32)
            B, num_box, _ = object_bbx_center.shape
            object_bbx_center = object_bbx_center.view(B * num_box, -1)
            object_bbx_corners = box_utils.boxes_to_corners_3d(object_bbx_center, order=self.order).view(B, num_box, 8, 3)

            reference_points = object_bbx_corners.permute(2, 0, 1, 3)
            D, B, num_query = reference_points.size()[:3]
            num_cam = rots.shape[1]

            reference_points = reference_points.view(  # D, B, num_cam, num_query, 3
                D, B, 1, num_query, 3).repeat(1, 1, num_cam, 1, 1)
            reference_points_mask = reference_points.new_ones((
                D, B, num_cam, num_query, 1), dtype=torch.bool)  # D, B, num_cam, num_query, 1
            for b in range(B):
                reference_points_mask[:, b, :, bbx_num[b]:, :] = False

            # --------------- coordinate transformation ---------------
            # [du, dv, d]^T = intrins * rots^(-1) * ([x,y,z]^T - trans)
            reference_points -= trans.view(1, B, num_cam, 1, 3).type_as(reference_points)
            combine = intrins.matmul(torch.inverse(rots)).type_as(reference_points)
            reference_points = combine.view(1, B, num_cam, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)

            eps = 1e-5
            reference_points_mask = (reference_points[..., 2:3] > eps) & reference_points_mask

            # (du, dv, d) --> (u, v, d)
            reference_points[..., 0:2] = reference_points[..., 0:2] / torch.maximum(
                reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3]) * eps)

        # 数据增强及预处理对像素的变化: (u0, v0, d0) = post_rots^(-1) * ([u, v, d] - post_trans)
        # (u, v, d) = post_rots * [u0, v0, d0] + post_trans
        reference_points = post_rots.view(1, B, num_cam, 1, 3, 3).type_as(reference_points).matmul(
            reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points += post_trans.view(1, B, num_cam, 1, 3).type_as(reference_points)
        reference_points = reference_points[..., :2]  # D, B, num_cam, num_query, 2

        reference_points[..., 0] /= img_shape[1]
        reference_points[..., 1] /= img_shape[0]
        # --------------- coordinate transformation ---------------

        # --------------- 2D box corners points and mask  ---------------
        reference_points_mask = (reference_points_mask & (reference_points[..., 1:2] > 0.0)
                                 & (reference_points[..., 1:2] < 1.0) & (reference_points[..., 0:1] < 1.0)
                                 & (reference_points[..., 0:1] > 0.0))

        reference_points = reference_points.permute(1, 2, 3, 0, 4)  # B, num_cam, num_query, D, 2
        reference_points_mask = reference_points_mask.permute(1, 2, 3, 0, 4)  # B, num_cam, num_query, D, 1

        reference_points_2d = torch.zeros_like(reference_points[:, :, :, :2, :])
        reference_points_2d[:, :, :, 0, :], _ = reference_points.min(dim=-2)
        reference_points_2d[:, :, :, 1, :], _ = reference_points.max(dim=-2)
        reference_points_2d = reference_points_2d.clamp(0, 1)  # B, num_cam, num_query, 2, 2

        reference_points_2d_mask = reference_points_mask.any(dim=-2).view(
            B, num_cam, num_query, 1, 1).repeat(1, 1, 1, 2, 1)  # B, num_cam, num_query, 2, 1
        # --------------- 2D box corners points and mask  ---------------

        # --------------- view cone encoding; regroup img feats, 2D box corners points and mask ---------------
        cone_coding = self.view_cone_encoding(reference_points_2d, img_metas)  # B, L, num_cam, num_query, 3, 3
        cone_coding[..., 0:1] = cone_coding[..., 0:1] / (self.pc_range[3] - self.pc_range[0])
        cone_coding[..., 1:2] = cone_coding[..., 1:2] / (self.pc_range[4] - self.pc_range[1])
        cone_coding[..., 2:3] = cone_coding[..., 2:3] / (self.pc_range[5] - self.pc_range[2])

        B, L = pairwise_t_matrix.shape[:2]
        regroup_reference_points_2d_mask = reference_points_2d_mask.new_zeros(B, L, num_cam, num_query, 2, 1)
        regroup_reference_points_2d = reference_points_2d.new_zeros(B, L, num_cam, num_query, 2, 2)
        regroup_feats = feats.new_zeros(B, L, *feats.shape[1:])  # B, L, num_cam, fc, fh, fw
        cum_sum_record_len = torch.cat([record_len.new_zeros(1), torch.cumsum(record_len, dim=0)])

        for b in range(B):
            regroup_reference_points_2d_mask[b, :record_len[b], ...] = \
                reference_points_2d_mask[cum_sum_record_len[b]:cum_sum_record_len[b + 1]]
            regroup_reference_points_2d[b, :record_len[b], ...] = \
                reference_points_2d[cum_sum_record_len[b]:cum_sum_record_len[b + 1]]
            regroup_feats[b, :record_len[b], ...] = feats[cum_sum_record_len[b]:cum_sum_record_len[b + 1]]

        reference_points_2d = regroup_reference_points_2d * regroup_reference_points_2d_mask
        reference_points_2d_mask = regroup_reference_points_2d_mask[...]  # B, L, num_cam, num_query, 2, 1
        # --------------- view cone encoding; regroup img feats, 2D box corners points and mask ---------------

        # --------------- get feats mask, obj feats list and obj feats cone encoding list ---------------
        fc, fh, fw = feats.shape[-3:]
        feats_mask = reference_points_2d_mask.new_zeros(B, L, num_cam, fh, fw)
        obj_feats = [[] for _ in range(B)]
        obj_feats_cone_coding = [[] for _ in range(B)]
        selected_indices = torch.nonzero(reference_points_2d_mask[:, :, :, :, 1, 0] > 0.)

        for index in selected_indices:
            b, i, j, y = index.tolist()
            selected_bbx = reference_points_2d[b, i, j, y, ...]
            x1, y1 = selected_bbx[0][0].item(), selected_bbx[0][1].item()
            x2, y2 = selected_bbx[1][0].item(), selected_bbx[1][1].item()
            x1, y1, x2, y2 = int(x1 * fw + 0.5), int(y1 * fh + 0.5), int(x2 * fw + 0.5), int(y2 * fh + 0.5)
            feats_mask[b, i, j, y1:y2, x1:x2].fill_(True)
            selected_obj_feats = regroup_feats[b, i, j, :, y1:y2, x1:x2].to(dtype=torch.float64)

            if self.pool == 'avg':
                selected_obj_feats /= (y2 - y1) * (x2 - x1)
                selected_obj_feats = selected_obj_feats.sum(dim=(1, 2))
            else:
                selected_obj_feats = selected_obj_feats.max(dim=(1, 2))[0]

            obj_feats[b].append(selected_obj_feats.type_as(reference_points))
            obj_feats_cone_coding[b].append(cone_coding[b, i, j, y, ...].flatten())

        target_obj_feats = [self.query_mlp(torch.stack(sublist, dim=0).type_as(feats))
                            if len(sublist) != 0 else feats.new_zeros(0, fc)
                            for sublist in obj_feats]
        target_obj_feats_cone_coding = [self.view_cone_mlp(torch.stack(sublist, dim=0).type_as(feats))
                                        if len(sublist) != 0 else feats.new_zeros(0, fc)
                                        for sublist in obj_feats_cone_coding]

        ego = 0
        feats_mask[:, ego, ...].fill_(True)     # B, L, num_cam, fh, fw
        # --------------- get feats mask, obj feats list and obj feats cone encoding list ---------------

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return regroup_feats, feats_mask, target_obj_feats, target_obj_feats_cone_coding

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        if img is not None:
            num_car, Ncams, imC, imH, imW = img.shape
            img = img.view(num_car * Ncams, imC, imH, imW)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            out_feats = self.img_backbone(img)
            out_feats = out_feats.view(num_car, Ncams, *out_feats.shape[-3:])

            return out_feats

        else:
            return None

    def extract_feat(self, img, single_label_dict, return_depth=False, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        img_feats_reshaped = []
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch([img_feats], focal=img_metas['focal'])
        else:
            depths = None

        regroup_feats, feats_mask, obj_feats, obj_feats_cone_coding = self.regroup_feats_and_mask(
            single_label_dict, img_feats, img_metas
        )
        B, L, Ncams, fC, fH, fW = regroup_feats.shape
        regroup_img_feats = regroup_feats.view(B, L * Ncams, fC, fH, fW)
        img_feats_reshaped.append(regroup_img_feats)

        feats_mask = feats_mask.view(B * L * Ncams, fH, fW).flatten(1)

        return img_feats_reshaped, feats_mask, obj_feats, obj_feats_cone_coding, depths

    def forward_pts_train(self, img_feats, label_dict, img_metas, prev_bev, **kwargs):
        """Forward function
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev, **kwargs)

        object_bbx_center = label_dict["object_bbx_center"]
        object_bbx_mask = label_dict["object_bbx_mask"]

        device = object_bbx_mask.device
        gt_bboxes_3d, gt_labels_3d = [], []
        for i in range(object_bbx_mask.shape[0]):
            selected_indices = torch.where(object_bbx_mask[i] > 0.)[0]
            selected_bbx = object_bbx_center[i, selected_indices].clone().detach().to(
                dtype=torch.float32, device=device)

            gt_bboxes_3d.append(selected_bbx)
            gt_labels_3d.append(torch.zeros(selected_bbx.shape[0], dtype=torch.int64, device=device))

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating my_code time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        image_inputs_dict = kwargs['ego']['image_inputs']
        img = image_inputs_dict['imgs']

        single_label_dict = kwargs['ego']["label_dict_single"]
        label_dict = {
            'object_bbx_center': kwargs['ego']['object_bbx_center'],
            'object_bbx_mask': kwargs['ego']['object_bbx_mask'],
            'object_bbx_center_single': single_label_dict['object_bbx_center_single'],
            'object_bbx_mask_single': single_label_dict['object_bbx_mask_single']
        }

        rots, trans, post_rots, post_trans, intrins, focal, pairwise_t_matrix, record_len = \
            image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['post_rots'],\
                image_inputs_dict['post_trans'], image_inputs_dict['intrins'], image_inputs_dict['focal'],\
                kwargs['ego']['pairwise_t_matrix'], kwargs['ego']['record_len']

        img_metas = {
            'rots': rots, 'trans': trans, 'post_rots': post_rots, 'post_trans': post_trans, 'intrins': intrins,
            'focal': focal, 'pairwise_t_matrix': pairwise_t_matrix,
            'record_len': record_len, 'img_shape': img.size()[3:]
        }

        img_feats, feats_mask, obj_feats, obj_feats_cone_coding, depths = self.extract_feat(
            img=img, single_label_dict=single_label_dict, return_depth=True, img_metas=img_metas)

        losses_pts = self.forward_pts_train(
            img_feats=img_feats, label_dict=label_dict, img_metas=img_metas, prev_bev=None, feats_mask=feats_mask,
            obj_feats=obj_feats, obj_feats_cone_coding=obj_feats_cone_coding
        )

        if depths is not None and "depth_maps" in image_inputs_dict:
            losses_pts["loss_dense_depth"] = self.depth_branch.loss(
                depths, [image_inputs_dict["depth_maps"]]
            )

        losses = sum(losses_pts.values())

        return losses_pts, losses

    def forward_test(self, img_metas=None, img=None, **kwargs):
        image_inputs_dict = kwargs['ego']['image_inputs']
        img = image_inputs_dict['imgs']

        single_label_dict = kwargs['ego']["label_dict_single"]

        rots, trans, post_rots, post_trans, intrins, focal, pairwise_t_matrix, record_len = \
            image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['post_rots'], \
                image_inputs_dict['post_trans'], image_inputs_dict['intrins'], image_inputs_dict['focal'], \
                kwargs['ego']['pairwise_t_matrix'], kwargs['ego']['record_len']

        img_metas = {
            'rots': rots, 'trans': trans, 'post_rots': post_rots, 'post_trans': post_trans, 'intrins': intrins,
            'focal': focal.view(-1, 1), 'pairwise_t_matrix': pairwise_t_matrix,
            'record_len': record_len, 'img_shape': img.size()[3:]
        }

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        new_prev_bev, bbox_results = self.simple_test(
            img=img, single_label_dict=single_label_dict, img_metas=img_metas, prev_bev=None)

        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, img_feats, img_metas, prev_bev=None, rescale=False, **kwargs):
        """Test function"""
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, **kwargs)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img=None, single_label_dict=None, img_metas=None, prev_bev=None, rescale=False):
        """
        Test function without augmentaiton.
        """
        img_feats, feats_mask, obj_feats, obj_feats_cone_coding, _ = self.extract_feat(
            img=img, single_label_dict=single_label_dict, img_metas=img_metas)

        bbox_list = [dict() for i in range(img_metas["pairwise_t_matrix"].shape[0])]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev, rescale=rescale,
            feats_mask=feats_mask, obj_feats=obj_feats, obj_feats_cone_coding=obj_feats_cone_coding
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return new_prev_bev, bbox_list
