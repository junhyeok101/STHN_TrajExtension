#test() 함수가 데이터셋을 준비해서 → 최종적으로 evaluate_SNet()을 호출하고, 여기서 모델 forward와 메트릭 계산/시각화를 전부 수행하는 구조.
# 재구성 한 데이터셋 가지고 매칭하고 매칭 결과 사진으로 전부 저장함. 
# input 재구성한 데이터셋
# output t_output , mace 와 같은 평가 지표

import numpy as np
import os
import torch
import argparse
from model.network import STHN
from utils import save_overlap_img, save_img, setup_seed, save_overlap_bbox_img
import datasets_4cor_img as datasets
import scipy.io as io
import torchvision
import numpy as np
import time
from tqdm import tqdm
import cv2
import kornia.geometry.transform as tgm
import matplotlib.pyplot as plt
from plot_hist import plot_hist_helper
import torch.nn.functional as F
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import wandb
import platform

def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)
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
    total_flow = torch.empty(0)
    total_ce = torch.empty(0)
    total_mace_conf_error = torch.empty(0)
    timeall = []
    mace_conf_list = []
    if args.generate_test_pairs:
        test_pairs = torch.zeros(len(val_dataset.dataset), dtype=torch.long)

    # 메모리 파편화 완화
    torch.cuda.empty_cache()

    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        if i_batch >=200:   # 🔥 20배치만 평가
            break
        img1, img2, flow_gt, H, query_utm, database_utm, index, pos_index = [x for x in data_blob]
        if args.generate_test_pairs:
            test_pairs[index] = pos_index

        dc = query_utm - database_utm   # (B, 1, 2)
        if i_batch < 200:  # 앞 배치만 출력
            print(f"[Batch {i_batch}] DC (query_utm - database_utm): {dc}")

            # 각 쌍별로 방향 + 거리 계산
            for j in range(dc.shape[0]):
                dx, dy = dc[j, 0].tolist()

                # 방향 해석
                direction_x = "위쪽" if dx < 0 else "아래쪽"
                direction_y = "왼쪽" if dy < 0 else "오른쪽"

                # 거리 계산 (L2 norm)
                d = (dx**2 + dy**2) ** 0.5

                print(f"GT, 중심 위성 중심 좌표 기준 : {direction_x}으로 {abs(dx):.1f}px, {direction_y}으로 {abs(dy):.1f}px → 총 {d:.2f}px 떨어져 있음")



        if i_batch == 0:
            logging.info("Check the reproducibility by UTM:")
            logging.info(f"the first 5th query UTMs: {query_utm[:5]}")
            logging.info(f"the first 5th database UTMs: {database_utm[:5]}")

        if i_batch % 1000 == 0:
            save_img(torchvision.utils.make_grid((img1)),
                     args.save_dir + "/b1_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')
            save_img(torchvision.utils.make_grid((img2)),
                     args.save_dir + "/b2_epoch_" + str(i_batch).zfill(5) + "_finaleval_" + '.png')
            torch.cuda.empty_cache()

        if not args.identity:
            # ✅ set_input: 인자 수 맞춤
            model.set_input(img1, img2, flow_gt)

            # 텐서를 numpy로
            q_img = img1[0].permute(1, 2, 0).cpu().numpy()
            d_img = img2[0].permute(1, 2, 0).cpu().numpy()
            # [0,1] -> [0,255]
            q_img = (q_img * 255).astype(np.uint8)
            d_img = (d_img * 255).astype(np.uint8)

            # 원본 시각화 크기
            h, w = q_img.shape[:2]

        if args.train_ue_method != 'train_only_ue_raw_input':
            if not args.identity:
                # ✅ 평가: 그래디언트 비활성화 (OOM 완화)
                with torch.no_grad():
                    model.forward()
                four_pred = model.four_pred  # (B,2,2,2)
            else:
                four_pred = torch.zeros((flow_gt.shape[0], 2, 2, 2))

            # ----- 시각화용 호모그래피 & 겹치기 -----
            if not args.identity and i_batch < 200  and False:   # ✅ 앞의 10개만 저장
                # 모델 좌표계 해상도 (항상 여기서 H를 계산)
                S = int(args.resize_width)

                # (S,S)로 리사이즈해서 H에 맞춘다
                q_small = cv2.resize(q_img, (S, S))
                d_small = cv2.resize(d_img, (S, S))

                # 원 코너 (S,S) 기준, CE와 동일한 전개 방식 사용
                four_point_org_single = torch.zeros((1, 2, 2, 2))
                four_point_org_single[:, :, 0, 0] = torch.tensor([0, 0])
                four_point_org_single[:, :, 0, 1] = torch.tensor([S - 1, 0])
                four_point_org_single[:, :, 1, 0] = torch.tensor([0, S - 1])
                four_point_org_single[:, :, 1, 1] = torch.tensor([S - 1, S - 1])

                # CE와 동일한 플래튼/퍼뮤트로 (4,2) 점 집합 생성
                src_pts = four_point_org_single.flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)
                dst_pts = (four_pred[0].cpu().detach() + four_point_org_single) \
                            .flatten(2).permute(0, 2, 1)[0].numpy().astype(np.float32)

                # 4점 호모그래피 & 워핑 (모델 좌표계)
                H_cv = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_small = cv2.warpPerspective(d_small, H_cv, (S, S))

                # 알파 블렌딩
                alpha = 0.5
                overlay_small = cv2.addWeighted(q_small, 1 - alpha, warped_small, alpha, 0)

                # 보기 좋게 원래 크기로 업샘플하여 저장
                d_big       = cv2.resize(d_small, (w, h))
                overlay_big = cv2.resize(overlay_small, (w, h))

                vis3 = np.hstack([q_img, d_big, overlay_big])
                save_dir = "t_outputs/match_images"   # 원하는 폴더명
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"match_{i_batch}.png")
                cv2.imwrite(save_path, cv2.cvtColor(vis3, cv2.COLOR_RGB2BGR))
                print(f"Saved {save_path}") 



            # ----- 메트릭 계산 -----
            # flow 4-corners
            flow_4cor = torch.zeros((flow_gt.shape[0], 2, 2, 2))
            flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
            flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
            flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
            flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
            flow_ = (flow_4cor) ** 2
            flow_ = ((flow_[:, 0, :, :] + flow_[:, 1, :, :]) ** 0.5)
            flow_vec = torch.mean(torch.mean(flow_, dim=1), dim=1)

            mace_ = (flow_4cor - four_pred.cpu().detach()) ** 2
            mace_ = ((mace_[:, 0, :, :] + mace_[:, 1, :, :]) ** 0.5)
            mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)

            total_mace = torch.cat([total_mace, mace_vec], dim=0)
            final_mace = torch.mean(total_mace).item()
            total_flow = torch.cat([total_flow, flow_vec], dim=0)
            final_flow = torch.mean(total_flow).item()

            # CE (kornia로 중심점 오프셋)
            four_point_org_single_w = torch.zeros((1, 2, 2, 2))
            four_point_org_single_w[:, :, 0, 0] = torch.Tensor([0, 0])
            four_point_org_single_w[:, :, 0, 1] = torch.Tensor([args.resize_width - 1, 0])
            four_point_org_single_w[:, :, 1, 0] = torch.Tensor([0, args.resize_width - 1])
            four_point_org_single_w[:, :, 1, 1] = torch.Tensor([args.resize_width - 1, args.resize_width - 1])

            four_point_1 = four_pred.cpu().detach() + four_point_org_single_w
            four_point_org = four_point_org_single_w.repeat(four_point_1.shape[0], 1, 1, 1).flatten(2).permute(0, 2, 1).contiguous()
            four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
            four_point_gt = flow_4cor.cpu().detach() + four_point_org_single_w
            four_point_gt = four_point_gt.flatten(2).permute(0, 2, 1).contiguous()

            H_k = tgm.get_perspective_transform(four_point_org, four_point_1)
            center_T = torch.tensor([args.resize_width / 2 - 0.5, args.resize_width / 2 - 0.5, 1]).unsqueeze(1).unsqueeze(0).repeat(H_k.shape[0], 1, 1)
            w_ = torch.bmm(H_k, center_T).squeeze(2)
            center_pred_offset = w_[:, :2] / w_[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)

            H_gt = tgm.get_perspective_transform(four_point_org, four_point_gt)
            w_gt = torch.bmm(H_gt, center_T).squeeze(2)
            center_gt_offset = w_gt[:, :2] / w_gt[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)

            ce_ = (center_pred_offset - center_gt_offset) ** 2
            ce_ = ((ce_[:, 0] + ce_[:, 1]) ** 0.5)
            ce_vec = ce_
            total_ce = torch.cat([total_ce, ce_vec], dim=0)
            final_ce = torch.mean(total_ce).item()

            if args.vis_all:
                save_dir = os.path.join(args.save_dir, 'vis')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                if not args.two_stages:
                    save_overlap_bbox_img(img1, model.fake_warped_image_2,
                                          save_dir + f'/train_overlap_bbox_{i_batch}.png',
                                          four_point_gt, four_point_1)
                else:
                    four_point_org_single_ori = torch.zeros((1, 2, 2, 2))
                    four_point_org_single_ori[:, :, 0, 0] = torch.Tensor([0, 0])
                    four_point_org_single_ori[:, :, 0, 1] = torch.Tensor([args.database_size - 1, 0])
                    four_point_org_single_ori[:, :, 1, 0] = torch.Tensor([0, args.database_size - 1])
                    four_point_org_single_ori[:, :, 1, 1] = torch.Tensor([args.database_size - 1, args.database_size - 1])
                    four_point_bbox = model.flow_bbox.cpu().detach() + four_point_org_single_ori
                    alpha = args.database_size / args.resize_width
                    four_point_bbox = four_point_bbox.flatten(2).permute(0, 2, 1).contiguous() / alpha
                    save_overlap_bbox_img(img1, model.fake_warped_image_2,
                                          save_dir + f'/train_overlap_bbox_{i_batch}.png',
                                          four_point_gt, four_point_1,
                                          crop_bbox=four_point_bbox)

        if not args.identity and args.use_ue:
            with torch.no_grad():
                conf_pred = model.predict_uncertainty(GAN_mode=args.GAN_mode)
            conf_vec = torch.mean(conf_pred, dim=[1, 2, 3])
            if args.GAN_mode == "macegan" and args.D_net != "ue_branch":
                mace_conf_error_vec = F.l1_loss(conf_vec.cpu(), torch.exp(args.ue_alpha * mace_vec))
            elif args.GAN_mode == "vanilla_rej":
                flow_bool = torch.ones_like(flow_vec)
                alpha = args.database_size / args.resize_width
                flow_bool[flow_vec >= (args.rej_threshold / alpha)] = 0.0
                mace_conf_error_vec = F.binary_cross_entropy(conf_vec.cpu(), flow_bool)
            total_mace_conf_error = torch.cat([total_mace_conf_error, mace_conf_error_vec.reshape(1)], dim=0)
            final_mace_conf_error = torch.mean(total_mace_conf_error).item()
            if args.GAN_mode == "macegan" and args.D_net != "ue_branch":
                for i in range(len(mace_vec)):
                    mace_conf_list.append((mace_vec[i].item(), conf_vec[i].item()))
            elif args.GAN_mode == "vanilla_rej":
                for i in range(len(flow_vec)):
                    mace_conf_list.append((flow_vec[i].item(), conf_vec[i].item()))

    if not args.train_ue_method == "train_only_ue_raw_input":
        logging.info(f"MACE Metric: {final_mace}")
        logging.info(f'CE Metric: {final_ce}')
        print(f"MACE Metric: {final_mace}")
        print(f'CE Metric: {final_ce}')
        if wandb_log:
            wandb.log({"test_mace": final_mace})
            wandb.log({"test_ce": final_ce})
    if args.use_ue:
        mace_conf_list = np.array(mace_conf_list)
        plt.figure()
        plt.scatter(mace_conf_list[:, 0], mace_conf_list[:, 1], s=1)
        x = np.linspace(0, 100, 400)
        y = np.exp(args.ue_alpha * x)
        plt.plot(x, y, label='f(x) = exp(-0.1x)', color='red')
        plt.legend()
        plt.savefig(args.save_dir + f'/final_conf.png')
        plt.close()
        plt.figure()
        n, bins, patches = plt.hist(x=mace_conf_list[:, 1], bins=np.linspace(0, 1, 20))
        logging.info(n)
        plt.close()
        logging.info(f"MACE CONF ERROR Metric: {final_mace_conf_error}")
        if wandb_log:
            wandb.log({"test_mace_conf_error": final_mace_conf_error})

    #logging.info(np.mean(np.array(timeall[1:-1])))
    io.savemat(args.save_dir + '/resmat', {'matrix': total_mace.numpy()})
    np.save(args.save_dir + '/resnpy.npy', total_mace.numpy())
    io.savemat(args.save_dir + '/flowmat', {'matrix': total_flow.numpy()})
    np.save(args.save_dir + '/flownpy.npy', total_flow.numpy())
    plot_hist_helper(args.save_dir)


    # 결과 저장 경로 생성
    os.makedirs("t_outputs", exist_ok=True)

    metrics_path = os.path.join("t_outputs", "metrics.txt")

    # 실행 시간 계산
    end_time = datetime.now()
    elapsed_time = end_time - start_time  # timedelta 객체
    elapsed_str = str(elapsed_time).split(".")[0]  # 초 단위까지만

    with open(metrics_path, "w") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"MACE: {final_mace:.6f}\n")
        f.write(f"CE:   {final_ce:.6f}\n")
        f.write(f"Flow Mean: {final_flow:.6f}\n\n")

        f.write("=== Model Info ===\n")
        f.write(f"Eval Model       : {args.eval_model}\n")
        f.write(f"Two Stages       : {args.two_stages}\n\n")

        f.write("=== Dataset Info ===\n")
        f.write(f"Dataset Name     : {args.dataset_name}\n")
        f.write(f"Database Size    : {args.database_size}\n")
        f.write(f"Positive Thres   : {args.val_positive_dist_threshold}\n")
        f.write(f"Correlation Lvl  : {args.corr_level}\n")
        f.write(f"Generate Pairs   : {args.generate_test_pairs}\n\n")

        f.write("=== Runtime Settings ===\n")
        f.write(f"Batch Size       : {args.batch_size}\n")
        f.write(f"Num Workers      : {args.num_workers}\n")
        f.write(f"Lev0             : {args.lev0}\n")
        f.write(f"Start Time       : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time         : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed Time     : {elapsed_str}\n\n")  # ⬅️ 영어로 걸린시간 추가

        f.write("=== Data Augmentation ===\n")
        f.write(f"Augment Type     : {args.augment}\n")
        f.write(f"Rotate Max       : {args.rotate_max:.2f} rad ({np.degrees(args.rotate_max):.2f} deg)\n")
        f.write(f"Resize Max       : {args.resize_max}\n")
        f.write(f"Perspective Max  : {args.perspective_max}\n\n")

        f.write("=== System Info ===\n")
        f.write(f"Python Version   : {platform.python_version()}\n")
        f.write(f"PyTorch Version  : {torch.__version__}\n")
        f.write(f"CUDA Version     : {torch.version.cuda}\n")
        f.write(f"GPU : NVIDIA Tesla V100\n")  # 수정: Tesla V100 으로 저장


        f.write("===Query와 database 선택 ===\n")
        f.write("Query와 가까운 Database 후보들을 거리순으로 정렬한다. \n")
        f.write("후보가 충분히 많으면 5등~10등 사이에서 무작위로 하나를 선택한다. \n")
        f.write("후보가 적을 경우에는 가장 가까운 1등 후보를 선택한다. = 똑같은 중심좌표 \n")
        f.write("항상 같은 짝이 아니라 여러 순위의 후보를 쓰게 되어 매칭 다양성을 확보 \n")

        
    print(f"[INFO] Metrics saved at {metrics_path}")

   # 🔥 추가: MACE / CE 시각화 저장
    mace_vals = total_mace.cpu().numpy()
    ce_vals   = total_ce.cpu().numpy() 


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
    wandb_log = False
    if wandb_log:
        wandb.init(project="STHN-eval", entity="aeaaea898-yonsei-university", config=vars(args))
    test(args, wandb_log)

    

"""python3 local_pipeline/t_evaluate.py   --datasets_folder t_datasets   --dataset_name 3131_datasets   --eval_model pretrained_models/1536_two_stages/STHN.pth   --val_positive_dist_threshold 512   --lev0   --database_size 1536   --corr_level 4   --test   --num_workers 0   --batch_size 1"""


"""python3 local_pipeline/t_evaluate.py \
  --datasets_folder t_datasets \
  --dataset_name 3131_datasets \
  --eval_model pretrained_models/1536_two_stages/STHN.pth \
  --val_positive_dist_threshold 512 \
  --lev0 \
  --database_size 1536 \
  --corr_level 4 \
  --test \
  --num_workers 2 \
  --batch_size 1 \
  --augment img \
  --rotate_max 15 \
  --resize_max 0.1 \
  --perspective_max 5
"""



"""python3 local_pipeline/t_evaluate.py \
  --datasets_folder t_datasets \
  --dataset_name 3131_datasets \
  --eval_model pretrained_models/1536_two_stages/STHN.pth \
  --val_positive_dist_threshold 512 \
  --lev0 \
  --database_size 1536 \
  --corr_level 4 \
  --test \
  --num_workers 2 \
  --batch_size 1 \
  --augment img \
  --rotate_max 5 \
  --resize_max 0 \
  --perspective_max 0
"""