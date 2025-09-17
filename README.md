# Enhanced D2SA Amodal Instance Segmentation

## 📋 개요

이 프로젝트는 D2SA (Densely Annotated 2D Scene Annotation) 데이터셋을 사용하여 **Amodal Instance Segmentation**을 수행하는 시스템입니다. EfficientSAM을 기반으로 한 베이스라인 모델을 구현하고, 종합적인 평가 메트릭과 시각화 도구를 제공합니다.

## 🔧 주요 개선사항

### 1. D2SA Annotation 형식 완전 지원
- **Amodal mask**: 전체 객체 영역 (가려진 부분 포함)
- **Visible mask**: 실제로 보이는 영역
- **Invisible mask**: 가려진 영역 (선택적)
- **Occlusion metrics**: 가림 비율 및 깊이 정보

### 2. 종합적인 평가 메트릭
- **mIoU**: 평균 Intersection over Union
- **AP@0.5, AP@0.75**: Average Precision at different thresholds
- **AR@100**: Average Recall at max 100 detections
- **COCO 메트릭**: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100
- **Occlusion 특화 메트릭**: Invisible reconstruction IoU, Completion rate

### 3. 체크포인트 관리 시스템
- 자동 최고 성능 모델 저장
- 정기적인 체크포인트 저장 및 정리
- 모델 복원 기능
- Early stopping 지원

### 4. 고급 시각화 도구
- GT vs 예측 마스크 비교
- Amodal, Visible, Invisible 영역 시각화
- 학습 곡선 자동 생성
- 배치별 예측 결과 저장

## 📁 파일 구조

```
Zero_shot_AIS/
├── train_amodal_completion.py  # 메인 학습 스크립트
├── d2sa_dataset.py            # D2SA 데이터셋 클래스
├── baseline_model.py          # EfficientSAM 기반 베이스라인 모델
├── evaluation_metrics.py      # 평가 메트릭 구현
├── utils.py                   # 체크포인트 및 시각화 유틸리티
├── test_setup.py             # 시스템 테스트 스크립트
└── README.md                 # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정 확인

먼저 시스템이 올바르게 설정되었는지 테스트:

```bash
python test_setup.py
```

### 2. 데이터셋 경로 설정

`train_amodal_completion.py`에서 데이터셋 경로를 확인/수정:

```python
D2SA_ROOT = "/root/datasets/D2SA"
ANNOTATION_FILE = os.path.join(D2SA_ROOT, "D2S_amodal_training_rot0.json")
IMAGE_DIR = os.path.join(D2SA_ROOT, "images")
```

### 3. 학습 시작

```bash
python train_amodal_completion.py
```

### 4. 결과 확인

학습 완료 후 `output_enhanced/` 디렉토리에서 결과 확인:

```
output_enhanced/
├── Baseline Model/
│   ├── epoch_1/
│   │   ├── masks/           # 예측 마스크
│   │   └── visualizations/  # 시각화 결과
│   └── logs/               # 평가 메트릭
├── checkpoints/            # 모델 체크포인트
└── training_curves.png     # 학습 곡선
```

## 📊 평가 메트릭 설명

### 기본 메트릭
- **mIoU**: 모든 샘플의 IoU 평균
- **AP@0.5**: IoU 임계값 0.5에서의 Average Precision
- **AP@0.75**: IoU 임계값 0.75에서의 Average Precision
- **AR@100**: 최대 100개 검출에서의 Average Recall

### COCO 표준 메트릭
- **AP**: IoU 0.5:0.95 범위의 평균 AP
- **AP50**: IoU 0.5에서의 AP
- **AP75**: IoU 0.75에서의 AP
- **APs/APm/APl**: 작은/중간/큰 객체의 AP
- **AR1/AR10/AR100**: 최대 1/10/100 검출에서의 AR

### Occlusion 특화 메트릭
- **Invisible Reconstruction mIoU**: 가려진 영역 재구성 성능
- **Occlusion Completion Rate**: 전체 객체 완성도
- **Number of Occluded Objects**: 가려진 객체 수

## ⚙️ 하이퍼파라미터

주요 설정값들:

```python
EPOCHS = 6                    # 학습 에포크 수
BATCH_SIZE = 1               # 배치 크기
LEARNING_RATE = 1e-4         # 학습률
PATIENCE = 3                 # Early stopping patience
max_samples = 100            # 디버깅용 샘플 제한
```

## 🔍 주요 특징

### 1. D2SA Annotation 완전 지원
- RLE 및 Polygon 형식 자동 처리
- Amodal, Visible, Invisible 마스크 모두 활용
- Occlusion rate 및 depth 정보 처리

### 2. 안정적인 학습 프로세스
- 바운딩 박스 유효성 검사
- 예외 처리 및 오류 복구
- GPU 메모리 효율적 사용

### 3. 종합적인 결과 분석
- 에포크별 상세 메트릭 저장
- 시각적 비교 결과 생성
- 학습 과정 추적 및 분석

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```python
   BATCH_SIZE = 1  # 배치 크기 감소
   max_samples = 50  # 샘플 수 제한
   ```

2. **데이터셋 경로 오류**
   ```bash
   # 경로 확인
   ls /root/datasets/D2SA/
   ```

3. **의존성 설치**
   ```bash
   pip install torch torchvision pycocotools matplotlib scikit-learn opencv-python
   ```

### 로그 확인

문제 발생 시 다음 로그들을 확인:
- `output_enhanced/logs/`: 평가 메트릭 로그
- 콘솔 출력: 실시간 학습 진행 상황
- 체크포인트 로그: `output_enhanced/checkpoints/checkpoint_log.json`

## 📈 성능 개선 팁

1. **하이퍼파라미터 튜닝**
   - Learning rate 조정
   - 배치 크기 최적화
   - Early stopping patience 조정

2. **데이터 증강**
   - 추가적인 이미지 변환
   - 다양한 스케일링

3. **모델 아키텍처**
   - EfficientSAM 파라미터 조정
   - 추가 레이어 실험

## 📝 라이센스

이 프로젝트는 연구 목적으로 개발되었습니다. D2SA 데이터셋 및 EfficientSAM의 해당 라이센스를 준수해주세요.

## 🤝 기여

버그 리포트나 개선 제안은 언제든 환영합니다!

---

**Happy Training! 🚀** 