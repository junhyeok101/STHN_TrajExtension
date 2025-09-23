# 이는 ground truth 를 점으로 찍고, matching 결과를 점과 점들을 이어 경로로 표시한 plot 을 저장하는 파일
# input : 재구성한 데이터셋 
# t_outputs/compare에 output 저장


import numpy as np
import os
import torch
import argparse
from model.network import STHN
from utils import setup_seed
import datasets_4cor_img as datasets
from tqdm import tqdm
import cv2
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import matplotlib.pyplot as plt

def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)

        # ---- checkpoint load ----
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.','')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=False)

        if args.two_stages:
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.','')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'])

        model.setup()
        model.netG.eval()
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

    pred_centers, gt_centers = [], []

    os.makedirs("t_outputs", exist_ok=True)

    torch.cuda.empty_cache()

    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        if i_batch >= 200:   # 최대 200개까지만 확인
            break

        img1, img2, flow_gt, H, query_utm, database_utm, index, pos_index = [x for x in data_blob]

        db_center = database_utm[0].cpu().numpy()
        gt_center = query_utm[0].cpu().numpy()

        if not args.identity:
            model.set_input(img1, img2, flow_gt)
            with torch.no_grad():
                model.forward()
            four_pred = model.four_pred  # (B,2,2,2)

            # Predicted corners
            S = int(args.resize_width)
            four_point_org_single = torch.zeros((1, 2, 2, 2))
            four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0])
            four_point_org_single[:, :, 0, 1] = torch.tensor([S - 1, 0])
            four_point_org_single[:, :, 1, 0] = torch.tensor([0, S - 1])
            four_point_org_single[:, :, 1, 1] = torch.tensor([S - 1, S - 1])

            src_pts = four_point_org_single.flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)
            dst_pts = (four_pred[0].cpu().detach() + four_point_org_single) \
                        .flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)

            H_cv = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Predicted center (pixel)
            center_pix = np.array([[S/2, S/2]], dtype=np.float32)
            center_pix = np.expand_dims(center_pix, axis=0)
            center_pred = cv2.perspectiveTransform(center_pix, H_cv)[0, 0]

            # Pixel → UTM 변환
            scale = (args.database_size / args.resize_width) 

            # 이부분!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # 1. 픽셀 중심으로부터의 변위(변화량) 계산: 결과는 [delta_x, delta_y] 형태
            pixel_offset = center_pred - (S / 2)

            # 2. 픽셀 변위를 UTM 변위로 변환하면서 축 보정
            #    - 이미지의 Y축(아래로 +)과 UTM의 Y축(북쪽으로 +) 방향이 반대이므로, Y 변화량에 -1을 곱합니다.
            utm_offset_x = pixel_offset[1] * scale
            utm_offset_y = pixel_offset[0] * scale  # Y축 방향 보정

            # 3. 최종 UTM 좌표 계산
            #    db_center의 X에는 X 변화량을, Y에는 Y 변화량을 정확히 더합니다.
            pred_center_utm = db_center + np.array([utm_offset_x, utm_offset_y])

            pred_centers.append(pred_center_utm)
            gt_centers.append(gt_center)

            # Debug 출력
            print(f"[Batch {i_batch}] Database Center UTM: {db_center}")
            print(f"GT Query UTM: {gt_center}")
            print("Predicted 4 corners (pixel coords):")
            for k in range(4):
                print(f" Corner {k}: {dst_pts[k]}")
            print(f"Predicted Center (pixel): {center_pred}")
            print(f"Predicted Center (UTM approx): {pred_center_utm}\n")



    # --- Trajectory plot ---
    pred_arr, gt_arr = np.array(pred_centers), np.array(gt_centers)
    pred_arr = np.squeeze(pred_arr)
    gt_arr = np.squeeze(gt_arr)

    # --- Trajectory plot 1 (ordered line) ---
    plt.figure(figsize=(8, 8))
    plt.scatter(gt_arr[:, 0], gt_arr[:, 1], c="green", marker="o", label="GT Trajectory")
    plt.plot(pred_arr[:, 0], pred_arr[:, 1], "rx--", linewidth=0.8, markersize=5, label="Pred Trajectory")
    plt.ylabel("UTM Y")
    plt.legend()
    plt.title("GT vs Predicted Trajectory (Sequential Pred)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig("t_outputs/trajectory_ordered.png", dpi=200)
    plt.close()

    # --- Trajectory plot 2 (numbered points) ---
    plt.figure(figsize=(10, 10))
    plt.scatter(gt_arr[:, 0], gt_arr[:, 1], c="green", marker="o", label="GT Trajectory")
    for i, (x, y) in enumerate(gt_arr):
        plt.text(x + 5, y + 5, str(i), color="green", fontsize=7)

    plt.scatter(pred_arr[:, 0], pred_arr[:, 1], c="red", marker="x", label="Pred Trajectory")
    for i, (x, y) in enumerate(pred_arr):
        plt.text(x + 5, y + 5, str(i), color="red", fontsize=7)

    plt.ylabel("UTM Y")
    plt.legend()
    plt.title("GT vs Predicted Trajectory (Sequential Pred with Index)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig("t_outputs/trajectory_ordered_numbered.png", dpi=200)
    plt.close()



if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if not args.identity:
        args.save_dir = join(
            "test",
            args.save_dir,
            args.eval_model.split("/")[-2] if args.eval_model is not None else "none",
            f"{args.dataset_name}-{start_time.strftime('%Y  -%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    wandb_log = False  # 🚫 wandb 로그 끔
    test(args, wandb_log)


"""python3 local_pipeline/t_evaluate_debug_plot.py \
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