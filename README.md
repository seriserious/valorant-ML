# Valorant 마커 이상 탐지 프로젝트

파이토치(Pytorch)를 활용한 LSTM AutoEncoder 기반의 이상 탐지 시스템입니다. 발로란트(Valorant) 플레이 영상에서 빨간 마커(적 머리 위치)를 추출하고, 궤적 재구성 오차를 이용해 비정상적인 움직임(핵 의심)을 감지합니다.

## 📋 목차

* [프로젝트 개요](#프로젝트-개요)
* [주요 기능](#주요-기능)
* [설치 및 실행](#설치-및-실행)
* [사용 예시](#사용-예시)
* [프로젝트 구조](#프로젝트-구조)
* [환경 설정](#환경-설정)
* [리포팅(보고서)](#리포팅보고서)
* [기여 안내](#기여-안내)
* [라이선스](#라이선스)

## 🚀 프로젝트 개요

이 리포지토리는 **LSTM AutoEncoder**를 이용해 정상적인 발로란트 마커 궤적을 학습하고, 재구성 에러가 큰 시퀀스를 이상(핵 의심)으로 감지하는 시스템을 제공합니다.

## ⚙️ 주요 기능

* **전처리(Preprocessing)**: 영상 프레임에서 빨간 마커 좌표를 추출·정규화
* **시퀀스 생성**: 30프레임 단위로 좌표를 묶어 시계열 데이터 생성
* **비지도 학습**: 정상 궤적만으로 AutoEncoder 학습 (라벨 불필요)
* **이상 탐지**: 재구성 MSE 기준(평균 + 3σ) 초과 시 이상으로 판단

## 🛠️ 설치 및 실행

```bash
# 1. 리포지토리 클론
git clone https://github.com/yourusername/valorant-ML.git
cd valorant-ML

# 2. 가상환경 생성 및 활성화
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt
```

## 🎮 사용 예시

1. **학습 및 이상 탐지**

   ```bash
   python main.py
   ```
2. **개별 영상 이상 탐지**

   ```bash
   python -c "from detect.detect_anomaly import detect_anomaly; detect_anomaly('data/clip_007.mp4', model_path='trained_model.pth')"
   ```

## 📂 프로젝트 구조

```
Project/
├── data/                   # 빨간 마커 처리된 영상 클립
├── preprocess/             # 마커 추출 및 시퀀스 생성 모듈
├── model/                  # LSTM AutoEncoder 정의
├── train/                  # AutoEncoder 학습 스크립트
├── detect/                 # 이상 탐지(inference) 모듈
├── report/                 # 생성된 HTML·PNG 보고서
├── outputs/                # 로그 및 중간 결과
├── main.py                 # 전체 파이프라인 실행 스크립트
├── generate_report.py      # 보고서 자동 생성 스크립트
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 문서 (한글)
```

## ⚙️ 환경 설정

* **window\_size**: `preprocess` 모듈의 기본 시퀀스 길이(30) 변경 가능
* **threshold**: `detect.detect_anomaly()`에서 평균 + kσ 방식으로 자동 계산 또는 고정값 설정


