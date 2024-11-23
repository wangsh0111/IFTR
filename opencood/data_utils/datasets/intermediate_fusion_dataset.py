# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
import cv2
from opencood.utils import box_utils as box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils import common_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)


def fill_multiscale_depth_maps(points, image_width, image_height, strides):
    """
    将雷达点云数据填充到多尺度特征图中

    参数：
    - points: 形状为 [N, 4, 3] 的雷达点云数据，其中每行包含 (u, v, depth)。
    - image_width: 原始图像的宽度。
    - image_height: 原始图像的高度。
    - scales: 多尺度列表，每个尺度表示一个缩放因子。

    返回：
    - depth_maps_list: 包含每个摄像头的多尺度深度图的列表，每个深度图是一个 PyTorch 张量。
    """
    depth_maps_list = []

    for stride in strides:
        depth_maps = []
        scaled_width = int(image_width / stride)
        scaled_height = int(image_height / stride)

        for cam in range(points.shape[1]):
            cam_points = points[:, cam, :]
            sorted_indices = torch.argsort(cam_points[:, 2], descending=True)
            cam_points = cam_points[sorted_indices]

            u = cam_points[:, 0]
            v = cam_points[:, 1]
            depth = cam_points[:, 2]

            depth_map = torch.full((scaled_height, scaled_width), float('inf'))

            u_img = (u * (scaled_width - 1)).long()
            v_img = (v * (scaled_height - 1)).long()

            valid_mask = (u_img >= 0) & (u_img < scaled_width) & (v_img >= 0) & (v_img < scaled_height)
            u_img = u_img[valid_mask]
            v_img = v_img[valid_mask]
            depth_valid = depth[valid_mask]

            depth_map[v_img, u_img] = torch.minimum(depth_map[v_img, u_img], depth_valid)

            # 将无穷大深度值转换为0或其他值，以适应显示或进一步处理
            depth_map[depth_map == float('inf')] = 0
            depth_maps.append(depth_map)

        depth_maps = torch.stack(depth_maps, dim=0)
        depth_maps_list.append(depth_maps)

    return depth_maps_list


def getIntermediateFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.train = train
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                else params['fusion']['args']['proj_first']

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/my_code pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)   # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

            # generate targets label single GT, note the reference pose is itself.
            if self.params["fusion"]["dataset"] == 'dairv2x' and self.label_type == 'camera':
                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                    [selected_cav_base], selected_cav_base['params']['lidar_pose']
                )
                tmp_object_list = selected_cav_base['params']['vehicles_single_front']
                object_2d_np = np.zeros((1, self.params["postprocess"]["max_num"], 4))
                mask_2d_np = np.zeros((1, self.params["postprocess"]["max_num"]))

                j = 0
                for i, item in enumerate(tmp_object_list):
                    if item["type"] in ['Car', 'Van', 'Bus', 'Truck']:
                        object_2d_np[0][j][:] = item["2d_box"]["xmin"], item["2d_box"]["ymin"], \
                                             item["2d_box"]["xmax"], item["2d_box"]["ymax"]
                        mask_2d_np[0][j] = 1
                        j += 1

                selected_cav_processed.update({
                    "single_object_2d_bbx_center": object_2d_np,
                    "single_object_2d_bbx_mask": mask_2d_np
                })

            else:
                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                    [selected_cav_base], selected_cav_base['params']['lidar_pose']
                )

            selected_cav_processed.update({
                                "single_object_bbx_center": object_bbx_center,
                                "single_object_bbx_mask": object_bbx_mask
            })

            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                orin_imgs = []
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    opencv_image = np.array(img)
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                    orin_imgs.append(torch.from_numpy(opencv_image))

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                orin_imgs, imgs, intrins, extrinsics, rots, trans, post_rots, post_trans = \
                    torch.stack(orin_imgs), torch.stack(imgs), torch.stack(intrins), torch.stack(extrinsics), \
                        torch.stack(rots), torch.stack(trans), torch.stack(post_rots), torch.stack(post_trans)
                focal_combine = post_rots.matmul(intrins)
                focal = focal_combine[:, 0, 0]

                selected_cav_processed.update(
                    {
                    "image_inputs": 
                        {
                            "orin_imgs": orin_imgs,     # [Ncam, 3or4, H, W]
                            "imgs": imgs,               # [Ncam, 3or4, H, W]
                            "intrins": intrins,
                            "extrinsics": extrinsics,
                            "rots": rots,
                            "trans": trans,
                            "post_rots": post_rots,
                            "post_trans": post_trans,
                            "focal": focal,
                        }
                    }
                )

                if self.train and self.load_lidar_file:
                    key_points = torch.from_numpy(lidar_np[:, :3])
                    num_points = key_points.shape[0]
                    num_cam, _, h, w = imgs.shape

                    key_points = key_points.view(  # num_points, num_cam, 3
                        num_points, 1, 3).repeat(1, num_cam, 1)

                    # d[u,v,1]^T = intrins * rots^(-1) * ([x,y,z]^T - trans)
                    key_points -= trans.view(1, num_cam, 3)
                    combine = intrins.matmul(torch.inverse(rots))
                    key_points = combine.view(1, num_cam, 3, 3).matmul(key_points.unsqueeze(-1)).squeeze(-1)

                    key_points[..., 0:2] = key_points[..., 0:2] / torch.clamp(
                        key_points[..., 2:3], min=1e-5)

                    # 数据增强及预处理对像素的变化: (u0, v0, d0) = post_rots^(-1) * ([u, v, d] - post_trans)
                    # (u, v, d) = post_rots * [u0, v0, d0] + post_trans
                    key_points = post_rots.view(1, num_cam, 3, 3).matmul(key_points.unsqueeze(-1)).squeeze(-1)
                    key_points += post_trans.view(1, num_cam, 3)

                    # num_points, 3(u, v, d)
                    key_points[..., 0] /= w
                    key_points[..., 1] /= h

                    depth_maps = fill_multiscale_depth_maps(
                        points=key_points, image_width=w, image_height=h, strides=[8, ])

                    selected_cav_processed["image_inputs"].update({"depth_maps": depth_maps, })

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], ego_pose_clean
            )

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )

            return selected_cav_processed

        def get_item_ego_car(self, base_data_dict, idx):
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_lidar_pose_clean = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                    ego_cav_base = cav_content
                    break

            assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            object_stack = []
            object_id_stack = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            single_object_2d_bbx_center_list = []
            single_object_2d_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []

            projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt(
                    (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                    (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])   # 6dof pose
                cav_id_list.append(cav_id)

            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)

            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)

                if self.load_camera_file:
                    agents_image_inputs.append(selected_cav_processed['image_inputs'])

                if self.visualize:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])

                single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

                if self.params["fusion"]["dataset"] == 'dairv2x':
                    cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
                    transformation_matrix = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

                    # convert center to corner
                    object_bbx_center = selected_cav_processed['object_bbx_center']
                    object_bbx_corner = box_utils.boxes_to_corners_3d(
                        object_bbx_center, self.post_processor.params['order'])
                    projected_object_bbx_corner = box_utils.project_box3d(
                        object_bbx_corner, transformation_matrix)

                    object_stack.append(projected_object_bbx_corner)
                    object_id_stack += selected_cav_processed['object_ids']

                    single_object_2d_bbx_center_list.append(selected_cav_processed['single_object_2d_bbx_center'])
                    single_object_2d_bbx_mask_list.append(selected_cav_processed['single_object_2d_bbx_mask'])

                else:
                    object_stack.append(selected_cav_processed['object_bbx_center'])
                    object_id_stack += selected_cav_processed['object_ids']

            # generate single view GT label
            single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
            single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))

            single_object_2d_bbx_center = torch.from_numpy(np.array(single_object_2d_bbx_center_list))
            single_object_2d_bbx_mask = torch.from_numpy(np.array(single_object_2d_bbx_mask_list))

            processed_data_dict['ego'].update({
                "single_object_bbx_center_torch": single_object_bbx_center,
                "single_object_bbx_mask_torch": single_object_bbx_mask,
                "single_object_2d_bbx_center_torch": single_object_2d_bbx_center,
                "single_object_2d_bbx_mask_torch": single_object_2d_bbx_mask
            })

            if self.params["fusion"]["dataset"] == 'dairv2x':
                if len(cav_id_list) > 1:
                    veh_corners_np = object_stack[0]
                    inf_corners_np = object_stack[1]
                    inf_polygon_list = list(common_utils.convert_format(inf_corners_np))
                    veh_polygon_list = list(common_utils.convert_format(veh_corners_np))
                    iou_thresh = 0.05

                    gt_from_inf = []
                    for i in range(len(inf_polygon_list)):
                        inf_polygon = inf_polygon_list[i]
                        ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_from_inf.append(inf_corners_np[i])

                    if len(gt_from_inf):
                        gt_from_inf = np.stack(gt_from_inf)
                        gt_bboxes_3d = np.vstack([veh_corners_np, gt_from_inf])
                    else:
                        gt_bboxes_3d = veh_corners_np
                else:
                    gt_bboxes_3d = np.vstack(object_stack)

                object_stack = box_utils.corner_to_center(
                    gt_bboxes_3d, self.post_processor.params['order'])

                # make sure bounding boxes across all frames have the same number
                object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
                mask = np.zeros(self.params['postprocess']['max_num'])

                object_bbx_center[:object_stack.shape[0], :] = object_stack
                mask[:object_stack.shape[0]] = 1
                instance_id = [0] * object_stack.shape[0]

            else:
                # exclude all repetitive objects
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

                # make sure bounding boxes across all frames have the same number
                object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
                mask = np.zeros(self.params['postprocess']['max_num'])
                object_bbx_center[:object_stack.shape[0], :] = object_stack
                mask[:object_stack.shape[0]] = 1
                instance_id = [object_id_stack[i] for i in unique_indices]

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                 'object_bbx_mask': mask,
                 'object_ids': instance_id,
                 'cav_num': cav_num,
                 'pairwise_t_matrix': pairwise_t_matrix,
                 'lidar_poses_clean': lidar_poses_clean,
                 'lidar_poses': lidar_poses
                 })

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx, 'cav_id_list': cav_id_list})

            return processed_data_dict

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            data_dict_batch = OrderedDict()
            ego_id = -1
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
            assert ego_id != -1

            if self.train:
                cloned_dict = copy.deepcopy(base_data_dict)

                if self.params["fusion"]["dataset"] == 'dairv2x' and self.label_type == 'camera':
                    probability = random.random()
                    if probability < 0.5:
                        data_dict_batch.update(
                            {'ego': self.get_item_ego_car(cloned_dict, idx)['ego']}
                        )
                    else:
                        tmp_data = OrderedDict()
                        tmp_data[0] = OrderedDict()
                        tmp_data[1] = OrderedDict()
                        tmp_data[0].update(cloned_dict[1])
                        tmp_data[1].update(cloned_dict[0])
                        tmp_data[0]['ego'] = True
                        tmp_data[1]['ego'] = False
                        data_dict_batch.update(
                            {'ego': self.get_item_ego_car(tmp_data, idx)['ego']}
                        )

                else:
                    data_dict_batch.update(
                        {'ego': self.get_item_ego_car(cloned_dict, idx)['ego']}
                    )

            else:
                for cav_id, cav_content in base_data_dict.items():
                    cav_content['ego'] = False

                for cav_id, cav_content in base_data_dict.items():
                    if cav_id == -1:
                        continue

                    cloned_dict = copy.deepcopy(base_data_dict)
                    cloned_dict[cav_id]['ego'] = True

                    value = cloned_dict.pop(cav_id, None)
                    cloned_dict = {cav_id: value, **cloned_dict}

                    if cav_id == ego_id:
                        data_dict_batch['ego'] = {}
                        data_dict_batch['ego'].update(self.get_item_ego_car(cloned_dict, idx)['ego'])
                    else:
                        data_dict_batch[f"{cav_id}"] = {}
                        data_dict_batch[f"{cav_id}"].update(self.get_item_ego_car(cloned_dict, idx)['ego'])

            return data_dict_batch

        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            image_inputs_list = []

            # used to record different scenario
            record_len = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            object_bbx_center_single = []
            object_bbx_mask_single = []
            object_2d_bbx_center_single = []
            object_2d_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses'])     # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs'])
                
                record_len.append(ego_dict['cav_num'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])
                object_2d_bbx_center_single.append(ego_dict['single_object_2d_bbx_center_torch'])
                object_2d_bbx_mask_single.append(ego_dict['single_object_2d_bbx_mask_torch'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')
                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'record_len': record_len,
                'object_ids': object_ids[0],
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_pose_clean': lidar_pose_clean,
                'lidar_pose': lidar_pose,
            })

            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            output_dict['ego']['label_dict_single'] = {}
            output_dict['ego']['label_dict_single'].update({
                "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
            })

            if self.params["fusion"]["dataset"] == 'dairv2x':
                output_dict['ego']['label_dict_single'].update(
                    {
                        "object_2d_bbx_center_single": torch.cat(object_2d_bbx_center_single, dim=0),
                        "object_2d_bbx_mask_single": torch.cat(object_2d_bbx_mask_single, dim=0),
                    }
                )
                output_dict['ego']['label_dict_single'].update(
                    {
                        "object_2d_bbx_center_single": torch.cat(object_2d_bbx_center_single, dim=0),
                        "object_2d_bbx_mask_single": torch.cat(object_2d_bbx_mask_single, dim=0),
                    }
                )

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"

            output_dict = {}

            for cav_id, cav_content in batch[0].items():
                cloned_batch = [{'ego': {}}]
                cloned_batch[0]['ego'] = copy.deepcopy(cav_content)
                output_dict.update(
                    {f"{cav_id}": self.collate_batch_train(cloned_batch)['ego']}
                )

            ego_lidar_pose = output_dict['ego']['lidar_pose'][0]
            ego_lidar_pose_clean = output_dict['ego']['lidar_pose_clean'][0]

            for cav_id, cav_content in output_dict.items():
                cav_lidar_pose = cav_content['lidar_pose'][0]
                cav_lidar_pose_clean = cav_content['lidar_pose_clean'][0]

                transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

                cav_content.update(
                    {'transformation_matrix': torch.from_numpy(transformation_matrix),
                     'transformation_matrix_clean': torch.from_numpy(transformation_matrix_clean)}
                )

                cav_content.update({
                    "sample_idx": batch[0][f"{cav_id}"]['sample_idx'],
                    "cav_id_list": batch[0][f"{cav_id}"]['cav_id_list']
                })

            if output_dict is None:
                return None

            return output_dict

        def post_process(self, data_dict, output_dict):
            data_dict = {'ego': data_dict['ego']}
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float().cuda()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float().cuda()
            data_dict['ego'].update({
                'transformation_matrix': transformation_matrix_torch,
                'transformation_matrix_clean': transformation_matrix_clean_torch})
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_boxes3d = box_utils.boxes_to_corners_3d(
                output_dict['ego'][0]['pts_bbox']['boxes_3d'], order=self.params['postprocess']['order'])
            pred_score = output_dict['ego'][0]['pts_bbox']['scores_3d']

            return pred_boxes3d, pred_score, gt_box_tensor

        def post_process_once_nms(self, data_dict, output_dict):
            tmp_data_dict = {'ego': copy.deepcopy(data_dict['ego'])}
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float().cuda()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float().cuda()
            tmp_data_dict['ego'].update({
                'transformation_matrix': transformation_matrix_torch,
                'transformation_matrix_clean': transformation_matrix_clean_torch})
            gt_box_tensor = self.post_processor.generate_gt_bbx(tmp_data_dict)

            pred_box3d_list, pred_box2d_list = [], []
            for cav_id, cav_content in data_dict.items():
                assert cav_id in output_dict
                transformation_matrix = cav_content['transformation_matrix'].float()

                pred_boxes3d = output_dict[cav_id][0]['pts_bbox']['boxes_3d'].cuda()
                pred_score = output_dict[cav_id][0]['pts_bbox']['scores_3d'].cuda()

                if len(pred_boxes3d) != 0:
                    boxes3d_corner = box_utils.boxes_to_corners_3d(
                        pred_boxes3d, order=self.params['postprocess']['order'])
                    projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)

                    projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                    boxes2d_score = torch.cat((projected_boxes2d, pred_score.unsqueeze(1)), dim=1)

                    pred_box2d_list.append(boxes2d_score)
                    pred_box3d_list.append(projected_boxes3d)

            if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
                pred_box3d_tensor, scores = None, None

            else:
                # shape: (N, 5)
                pred_box2d_list = torch.vstack(pred_box2d_list)
                # scores
                scores = pred_box2d_list[:, -1]
                # predicted 3d bbx
                pred_box3d_tensor = torch.vstack(pred_box3d_list)
                # remove large bbx
                keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
                keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
                keep_index = torch.logical_and(keep_index_1, keep_index_2)

                pred_box3d_tensor = pred_box3d_tensor[keep_index]
                scores = scores[keep_index]

                keep_index = box_utils.nms_rotated(
                    pred_box3d_tensor, scores, self.params['postprocess']['nms_thresh'])

                pred_box3d_tensor = pred_box3d_tensor[keep_index]
                scores = scores[keep_index]

                pred_box3d_np = pred_box3d_tensor.cpu().numpy()

                pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(
                    pred_box3d_np, self.params['postprocess']['gt_range'], order=None, return_mask=True)

                pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
                scores = scores[mask]
                assert scores.shape[0] == pred_box3d_tensor.shape[0]

            return pred_box3d_tensor, scores, gt_box_tensor

    return IntermediateFusionDataset
