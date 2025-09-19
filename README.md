# STHN-TrajExtension
Extension of STHN model for UAV–Satellite geo-localization with single-line trajectory dataset from original dataset 
## 🇰🇷 한글 요약
이 프로젝트는 STHN 모델을 기반으로, 직접 멀티라인 trajectory 데이터를 단일 trajectory로 단순화하여 재가공 후, local matching 성능과 좌표 오차를 평가한 확장 연구 코드입니다

---

## 📌 Overview
This repository contains my extension work based on the [STHN model](https://github.com/arplaboratory/STHN).  
The main idea is to process the original STHN dataset to create a **single-line trajectory dataset**,  
then evaluate how well the pretrained STHN performs **local matching** on this simplified data.  

Key points:
- Dataset preprocessing (multi-line → single-line trajectory)
- Local matching with pretrained STHN
- Evaluation of coordinate errors vs. ground truth
- Trajectory visualization

---

## 📂 Structure
