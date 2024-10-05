import os
from collections import OrderedDict
import torch
import numpy as np
from torch.utils.data import Dataset
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils import box_utils


class DAIRV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
    
    def reinitialize(self):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # pose of agent 
        lidar_to_novatel = read_json(os.path.join(
            self.root_dir, 'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(
            self.root_dir, 'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        virtuallidar_to_world = read_json(os.path.join(
            self.root_dir, 'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(
            virtuallidar_to_world, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['params']['vehicles_all'] = read_json(
            os.path.join(self.root_dir, frame_info['cooperative_label_path']))
        data[1]['params']['vehicles_all'] = []  # we only load cooperative label in vehicle side

        data[0]['params']['vehicles_front'] = read_json(os.path.join(
            self.root_dir, frame_info['cooperative_label_path'].replace("label_world", "label_world_backup")))
        data[1]['params']['vehicles_front'] = []  # we only load cooperative label in vehicle side

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(read_json(
                os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(read_json(os.path.join(
                self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))
            
            data[1]['camera_data']= load_camera_data([os.path.join(
                self.root_dir, frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(read_json(
                os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(read_json(
                os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))

        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(
                self.root_dir, frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(
                self.root_dir, frame_info["infrastructure_pointcloud_path"]))

        # Label for single side
        data[0]['params']['vehicles_single_all'] = read_json(os.path.join(
            self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(
            self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))

        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(
            self.root_dir, 'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(
            self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))

        return data

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(
            cav_contents, reference_lidar_pose)

    def generate_object_center_camera(self, object_bbx_center, object_bbx_mask, object_ids, transformation_matrix_clean):
        """
        reference lidar 's coordinate 
        """
        object_bbx_center = torch.from_numpy(object_bbx_center)
        object_bbx_mask = torch.from_numpy(object_bbx_mask)

        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        transformation_matrix = transformation_matrix_clean
        object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        # convert center to corner
        object_bbx_corner = box_utils.boxes_to_corners_3d(
            object_bbx_center, self.post_processor.params['order'])
        # self.post_processor.params['max_num']
        projected_object_bbx_corner = box_utils.project_box3d(object_bbx_corner.float(), transformation_matrix)
        gt_box3d_list.append(projected_object_bbx_corner)

        # append the corresponding ids
        object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        gt_box3d_selected_indices = [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range. with z dim
        gt_box3d_np = gt_box3d_tensor.cpu().numpy()
        gt_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(
            gt_box3d_np, self.post_processor.params['gt_range'], order=None, return_mask=True)

        object_id_list = list(np.array(object_id_list)[mask])
        gt_box3d_np = box_utils.corner_to_center(gt_box3d_np, self.post_processor.params['order'])

        object_np = np.zeros((self.post_processor.params['max_num'], 7))
        mask = np.zeros(self.post_processor.params['max_num'])

        object_np[:len(gt_box3d_np)] = gt_box3d_np
        mask[:len(gt_box3d_np)] = 1

        return object_np, mask, object_id_list
                                                        
    ### Add new func for single side
    def generate_object_center_single(self, cav_contents, reference_lidar_pose, **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic
