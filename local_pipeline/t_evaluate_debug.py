# 호모그래피 매칭 후 좌표 결과를 database 와 비교함. 
# 터미널 output 용 
# databse center utm GT query

# 현재 database center 값이 이상한 것으로 추측. 

import numpy as np
import os
import torch
import argparse
from model.network import STHN
from utils import setup_seed
import datasets_4cor_img as datasets
import torchvision
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb

def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)

        # ---- checkpoint load ----
        if not args.train_ue_method == "train_only_ue_raw_input":
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG'].keys()):
                model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
            for key in list(model_med['netG'].keys()):
                if key.startswith('module'):
                    del model_med['netG'][key]
            model.netG.load_state_dict(model_med['netG'], strict=False)

        if args.use_ue:
            if args.eval_model_ue is not None:
                model_med = torch.load(args.eval_model_ue, map_location='cuda:0')
            for key in list(model_med['netD'].keys()):
                model_med['netD'][key.replace('module.','')] = model_med['netD'][key]
            for key in list(model_med['netD'].keys()):
                if key.startswith('module'):
                    del model_med['netD'][key]
            model.netD.load_state_dict(model_med['netD'])

        if args.two_stages:
            if args.eval_model_fine is None:
                model_med = torch.load(args.eval_model, map_location='cuda:0')
                for key in list(model_med['netG_fine'].keys()):
                    model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
                for key in list(model_med['netG_fine'].keys()):
                    if key.startswith('module'):
                        del model_med['netG_fine'][key]
                model.netG_fine.load_state_dict(model_med['netG_fine'])
            else:
                model_med = torch.load(args.eval_model_fine, map_location='cuda:0')
                for key in list(model_med['netG'].keys()):
                    model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
                for key in list(model_med['netG'].keys()):
                    if key.startswith('module'):
                        del model_med['netG'][key]
                model.netG_fine.load_state_dict(model_med['netG'], strict=False)

        model.setup()
        model.netG.eval()
        if args.use_ue:
            model.netD.eval()
        if args.two_stages:
            model.netG_fine.eval()
    else:
        model = None

    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')

    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)


def evaluate_SNet(model, val_dataset, batch_size=0, args=None, wandb_log=False):
    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)
    total_ce = torch.empty(0)

    torch.cuda.empty_cache()

    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        if i_batch >= 20:   # 앞 배치만 확인
            break

        img1, img2, flow_gt, H, query_utm, database_utm, index, pos_index = [x for x in data_blob]

        # ========================
        # DB 중심 좌표
        # ========================
        db_center = database_utm[0, 0].cpu().numpy()
        print(f"\n[Batch {i_batch}] Database Center UTM: {db_center}")

        # ========================
        # GT 좌표
        # ========================
        gt_center = query_utm[0, 0].cpu().numpy()
        print(f"GT Query UTM: {gt_center}")

        if not args.identity:
            model.set_input(img1, img2, flow_gt)
            with torch.no_grad():
                model.forward()
            four_pred = model.four_pred  # (B,2,2,2)

            # ------------------------
            # Predicted 4 corners
            # ------------------------
            S = int(args.resize_width)
            four_point_org_single = torch.zeros((1, 2, 2, 2))
            four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0])
            four_point_org_single[:, :, 0, 1] = torch.tensor([S - 1, 0])
            four_point_org_single[:, :, 1, 0] = torch.tensor([0, S - 1])
            four_point_org_single[:, :, 1, 1] = torch.tensor([S - 1, S - 1])

            src_pts = four_point_org_single.flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)
            dst_pts = (four_pred[0].cpu().detach() + four_point_org_single) \
                            .flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)

            print("Predicted 4 corners (pixel coords):")
            for k, pt in enumerate(dst_pts):
                print(f" Corner {k}: {pt}")

            # ------------------------
            # Predicted Center (pixel)
            # ------------------------
            H_cv = cv2.getPerspectiveTransform(src_pts, dst_pts)
            center_px = np.array([S/2, S/2, 1], dtype=np.float32).reshape(3, 1)
            pred_center_px = (H_cv @ center_px).ravel()
            pred_center_px /= pred_center_px[2]

            print(f"Predicted Center (pixel): {pred_center_px[:2]}")

            # ------------------------
            # Predicted Center (UTM 변환)
            # ------------------------
            scale = args.val_positive_dist_threshold / args.database_size
            db_center_px = np.array([args.database_size/2, args.database_size/2])
            offset_px = pred_center_px[:2] - db_center_px
            pred_center_utm = db_center + offset_px * scale

            print(f"Predicted Center (UTM approx): {pred_center_utm}")

    print("\n✅ Debug evaluation finished.")


if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if args.identity:
        pass
    else:
        args.save_dir = join(
            "test",
            args.save_dir,
            args.eval_model.split("/")[-2] if args.eval_model is not None else args.eval_model_ue.split("/")[-2],
            f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    logging.debug(args)
    wandb_log = False   # wandb 끔
    test(args, wandb_log)


"""
python3 local_pipeline/t_evaluate_debug.py \
  --datasets_folder t_datasets \
  --dataset_name 3131_datasets \
  --eval_model pretrained_models/1536_two_stages/STHN.pth \
  --val_positive_dist_threshold 512 \
  --lev0 \
  --database_size 1536 \
  --corr_level 4 \
  --test \
  --num_workers 0 \
  --batch_size 1
"""