import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt

from model.network import STHN
import datasets_4cor_img as datasets
import parser
import commons
from utils import setup_seed


def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)

        # ---- checkpoint load ----
        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.', '')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=False)

        if args.two_stages:
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.', '')] = model_med['netG_fine'][key]
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

    # ---- dataset ----
    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')

    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)


def evaluate_SNet(model, val_dataset, batch_size=0, args=None, wandb_log=False):
    assert batch_size > 0, "batchsize > 0"

    pred_centers, gt_centers = [], []
    os.makedirs("t_outputs", exist_ok=True)

    # 로그 파일 초기화
    log_path = "t_outputs/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)

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
            four_pred = model.four_pred

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
            pixel_offset = center_pred - (S / 2)
            utm_offset_x = pixel_offset[1] * scale
            utm_offset_y = pixel_offset[0] * scale
            pred_center_utm = db_center + np.array([utm_offset_x, utm_offset_y])

            pred_centers.append(pred_center_utm)
            gt_centers.append(gt_center)

            # ---- 로그 저장 ----
            log_text = []
            log_text.append(f"[Batch {i_batch}] Database Center UTM: {db_center}")
            log_text.append(f"GT Query UTM: {gt_center}")
            log_text.append("Predicted 4 corners (pixel coords):")
            for k in range(4):
                log_text.append(f" Corner {k}: {dst_pts[k]}")
            log_text.append(f"Predicted Center (pixel): {center_pred}")
            log_text.append(f"Predicted Center (UTM approx): {pred_center_utm}\n")

            print("\n".join(log_text))
            with open(log_path, "a") as f:
                f.write("\n".join(log_text) + "\n")

    # --- Trajectory plot ---
    pred_arr, gt_arr = np.array(pred_centers), np.array(gt_centers)

    plt.figure(figsize=(10, 10))
    pred_arr = np.squeeze(pred_arr)
    gt_arr = np.squeeze(gt_arr)

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

    save_path = "t_outputs/trajectory_ordered_numbered.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f" Sequential Pred trajectory plot saved at {save_path}")
    print(f" 로그 저장 완료: {log_path}")


if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if not args.identity:
        args.save_dir = join(
            "test",
            args.save_dir,
            args.eval_model.split("/")[-2] if args.eval_model is not None else "none",
            f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    wandb_log = False
    test(args, wandb_log)
