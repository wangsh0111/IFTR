import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mmdet3d'))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        default='/data2/wsh/paper_project/IFTR-main/opencood/'
                                'logs/dair_iftr_efficientnet_b2_bs_2x1_2024_09_29_13_33_46/',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str, default='iftr',
                        help='iftr or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=50,
                        help='interval of saving visualization')
    parser.add_argument('--wo_od', action='store_true',
                        help="whether ues 2d detector to filter foreground")
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['intermediate', 'iftr']
    hypes = yaml_utils.load_yaml(None, opt)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    print(hypes['validate_dir'])

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    print('--------------------Creating Model--------------------')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('--------------------Loading Model from checkpoint--------------------')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    # setting noise
    np.random.seed(303)

    # build dataset for each noise setting
    print('--------------------Dataset Building--------------------')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset, batch_size=1, num_workers=4, collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False, pin_memory=False, drop_last=False
    )

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    infer_info = opt.fusion_method + opt.note
    if opt.wo_od:
        infer_info = opt.fusion_method + opt.note + '_wo_od'
    else:
        infer_info = opt.fusion_method + opt.note + '_od'


    for i, batch_data in enumerate(data_loader):
        print(f"[{infer_info}][{i}/{len(data_loader)}][val]")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if not opt.wo_od:
                for cav_item in batch_data.values():
                    orin_imgs = cav_item['image_inputs']['orin_imgs']
                    max_num = hypes["postprocess"]["max_num"]
                    object_2d_bbx_center_single = torch.zeros((orin_imgs.shape[1], max_num, 4)).to(device)
                    object_2d_bbx_mask_single = torch.zeros((orin_imgs.shape[1], max_num)).to(device)
                    for idx, img in enumerate(orin_imgs[0]):
                        outputs = predictor(img.cpu().numpy())
                        pred_classes = outputs["instances"].pred_classes
                        pred_boxes = outputs["instances"].pred_boxes.tensor

                        target_classes = torch.tensor([2, 5, 7]).to(device)
                        valid_mask = torch.isin(pred_classes, target_classes)
                        valid_bbx = pred_boxes[valid_mask]
                        num_bbx = min(valid_bbx.shape[0], max_num)

                        object_2d_bbx_center_single[idx, :num_bbx, :] = valid_bbx
                        object_2d_bbx_mask_single[idx, :num_bbx] = 1.
                        cav_item['label_dict_single']['object_2d_bbx_center_single'][0, idx, ...] = \
                            object_2d_bbx_center_single
                        cav_item['label_dict_single']['object_2d_bbx_mask_single'][0, idx, ...] = \
                            object_2d_bbx_mask_single

            if opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'iftr':
                infer_result = inference_utils.inference_iftr_fusion(
                    batch_data, model, opencood_dataset)
            else:
                raise NotImplementedError('Only intermediate fusion and iftr fusion are supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')

                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)

                inference_utils.save_prediction_gt(
                    pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], i, npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                vis_3d_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                simple_vis.visualize(
                    infer_result, batch_data['ego']['origin_lidar'][0], hypes['postprocess']['gt_range'],
                    vis_3d_save_path, method='3d', left_hand=left_hand
                )

                vis_bev_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                print(batch_data['ego'].keys())
                simple_vis.visualize(
                    infer_result, batch_data['ego']['origin_lidar'][0], hypes['postprocess']['gt_range'],
                    vis_bev_save_path, method='bev', left_hand=left_hand
                )

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, infer_info)
    print(f"AP30: {ap30}; AP50: {ap50}; AP70: {ap70}")


if __name__ == '__main__':
    main()
