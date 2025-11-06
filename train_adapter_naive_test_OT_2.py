'''''
LoRA 어댑터 기반 경량 미세 조정 학습
- visible_mask_decoder와 amodal_mask_decoder 모두 LoRA 어댑터 적용
- 인코더는 동결하여 기존 SAM 지식 보존
- 전체 파라미터의 1% 미만만 학습하여 과적합 방지
- 다양한 학습 전략 지원 (amodal_core, joint, amodal_refine, visible_only, amodal_only)
- 임베딩 공간 OT (Optimal Transport) 손실 함수 통합
  - Total loss = BCE(amodal) + BCE(visible) + α * Consistency + β * OT_embedding
  - OT_embedding: Visible과 Amodal 디코더의 쿼리 임베딩 간 구조적 매칭
  - Q×Q 매트릭스로 계산하여 픽셀 공간 대비 계산 효율성 극대화
  - 시맨틱 정보를 포함한 의미적 일관성 강제
'''''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import numpy as np
from pycocotools.coco import COCO
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 matplotlib 사용
import matplotlib.pyplot as plt
import cv2

# 커스텀 모듈 임포트
from d2sa_dataset import D2SADataset
from mp3d_dataset import MP3DDataset
from baseline_model_adapter_naive_test_OT_2 import BaselineModel  # LoRA 기반 모델 (Text RA + Cross-Attention 제거, 임베딩 공간 OT Loss)
# from baseline_model_adapter_text_RA_revision_CA_test import BaselineModel  # LoRA 기반 모델 (Cross-Attention + Attention Map 시각화)
# from baseline_model_adapter_text_RA_revision_CA import BaselineModel  # LoRA 기반 모델 (Cross-Attention 사용)
# from baseline_model_adapter_naive import BaselineModel  # Naive LoRA 기반 모델 사용
from evaluation_metrics import AmodalEvaluationMetrics, compute_occlusion_metrics
from utils import CheckpointManager, VisualizationUtils, denormalize_image, create_output_directories, save_metrics_to_file
from lora_adapter import count_parameters  # 파라미터 수 계산용

# --- 하이퍼파라미터 설정 ---
EPOCHS = 20     # EPOCHS=6
BATCH_SIZE = 4  # 배치 크기 증가 (메모리 허용 시)
LEARNING_RATE_STEP1 = 2e-4  # Step 1: 핵심 개념 학습용 학습률 (더 높게)
LEARNING_RATE_STEP2 = 1e-4  # Step 2: 정제 및 일관성 학습용 학습률
# PATIENCE = 5  # Early stopping을 위한 patience (더 관대하게)
PATIENCE = 8  # Early stopping을 위한 patience (20 epoch에 맞게 조정: 약 40% 수준)

# LoRA 설정
LORA_RANK = 16  # LoRA rank (낮을수록 더 적은 파라미터)
LORA_ALPHA = 16.0  # LoRA scaling factor

# 일관성 제약 가중치
CONSISTENCY_WEIGHT = 0.2  # Step 2에서 일관성 손실 가중치

# OT (Optimal Transport) 손실 가중치 (임베딩 공간)
OT_WEIGHT = 0.1  # OT 손실의 중요도를 조절하는 하이퍼파라미터 (α)
OT_SINKHORN_BLUR = 0.05  # Sinkhorn 알고리즘의 엔트로피 정규화 계수
OT_DISTANCE_TYPE = 'l2'  # 거리 계산 방식 ('l2', 'cosine', 'l1')

# --- 데이터셋 설정 ---
# 사용할 데이터셋 선택: 'd2sa' 또는 'mp3d'
DATASET_NAME = 'mp3d'  # 'd2sa' 또는 'mp3d'

# 데이터셋 경로 설정
if DATASET_NAME == 'd2sa':
    D2SA_ROOT = "/root/datasets/D2SA"
    # Training Dataset
    TRAIN_ANNOTATION_FILES = [
        os.path.join(D2SA_ROOT, "D2S_amodal_augmented.json")
    ]
    # Validation Dataset: 별도 validation set
    VAL_ANNOTATION_FILE = os.path.join(D2SA_ROOT, "D2S_amodal_validation.json")
    IMAGE_DIR = os.path.join(D2SA_ROOT, "images")
    OUTPUT_DIR = "./outputs/MP3D_amodal_visible_both_adapter_naive_OT_251105"
    
elif DATASET_NAME == 'mp3d':
    MP3D_ROOT = "/root/datasets/MP3D"
    OUTPUT_DIR = "./outputs/MP3D_amodal_visible_both_adapter_naive_OT_embedding__20epochs_251105"
    MAX_SAMPLES = None
    
else:
    raise ValueError(f"지원하지 않는 데이터셋: {DATASET_NAME}. 지원되는 데이d터셋: 'd2sa', 'mp3d'")

# --- 디바이스 설정 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용할 디바이스: {device}")

# --- 데이터 변환 설정 ---
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Training Dataset 및 DataLoader 초기화 ---
print(f"Training 데이터셋 초기화 중... (데이터셋: {DATASET_NAME})")
if DATASET_NAME == 'd2sa':
    train_dataset = D2SADataset(
        annotation_file=TRAIN_ANNOTATION_FILES,  # Training + Augmented
        image_dir=IMAGE_DIR, 
        transform=transform,
        max_samples=1000
    )
    val_dataset = D2SADataset(
        annotation_file=VAL_ANNOTATION_FILE,  # Validation set
        image_dir=IMAGE_DIR, 
        transform=transform,
        max_samples=500
    )
elif DATASET_NAME == 'mp3d':
    train_dataset = MP3DDataset(
        data_root=MP3D_ROOT,
        transform=transform,
        max_samples=MAX_SAMPLES,
        split='val'
    )
    val_dataset = MP3DDataset(
        data_root=MP3D_ROOT,
        transform=transform,
        max_samples=MAX_SAMPLES,
        split='train'
    )

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"Training set: {len(train_dataset)}개의 샘플 로드 완료.")

# --- Validation Dataset 및 DataLoader 초기화 ---
print("Validation 데이터셋 초기화 중...")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Validation set: {len(val_dataset)}개의 샘플 로드 완료.")

# --- 손실 함수 정의 ---
criterion = nn.BCEWithLogitsLoss()

# --- 통합된 출력 디렉토리 구조 생성 ---
# checkpoints, logs, visualizations 폴더를 루트에 생성
checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
logs_dir = os.path.join(OUTPUT_DIR, "logs")
visualizations_dir = os.path.join(OUTPUT_DIR, "visualizations")

for dir_path in [checkpoints_dir, logs_dir, visualizations_dir]:
    os.makedirs(dir_path, exist_ok=True)

print(f"통합된 출력 디렉토리 구조 생성:")
print(f"  - Checkpoints: {checkpoints_dir}")
print(f"  - Logs: {logs_dir}")
print(f"  - Visualizations: {visualizations_dir}")

# --- 체크포인트 매니저 초기화 ---
checkpoint_manager_baseline = CheckpointManager(
    checkpoint_dir=checkpoints_dir,
    max_checkpoints=5
)

# --- COCO GT 객체 생성 ---
import tempfile

# D2SA와 MP3D 데이터셋의 경우 COCO GT 생성
if DATASET_NAME in ['d2sa', 'mp3d']:
    # Training set용 COCO GT 생성
    train_coco_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    train_coco_data = {
        'info': {
            'description': 'Training dataset for amodal instance segmentation',
            'version': '1.0',
            'year': 2024
        },
        'images': train_dataset.coco['images'],
        'annotations': train_dataset.coco['annotations'], 
        'categories': train_dataset.coco['categories']
    }
    json.dump(train_coco_data, train_coco_file, indent=2)
    train_coco_file.close()

    train_coco_gt = COCO(train_coco_file.name)
    print(f"Training COCO GT 생성 완료: {len(train_dataset.coco['images'])}개 이미지, {len(train_dataset.coco['annotations'])}개 어노테이션")

    # Validation set용 COCO GT 생성
    val_coco_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    val_coco_data = {
        'info': {
            'description': 'Validation dataset for amodal instance segmentation',
            'version': '1.0',
            'year': 2024
        },
        'images': val_dataset.coco['images'],
        'annotations': val_dataset.coco['annotations'], 
        'categories': val_dataset.coco['categories']
    }
    json.dump(val_coco_data, val_coco_file, indent=2)
    val_coco_file.close()

    val_coco_gt = COCO(val_coco_file.name)
    print(f"Validation COCO GT 생성 완료: {len(val_dataset.coco['images'])}개 이미지, {len(val_dataset.coco['annotations'])}개 어노테이션")
else:
    # 지원하지 않는 데이터셋
    train_coco_gt = None
    val_coco_gt = None
    train_coco_file = None
    val_coco_file = None
    print("지원하지 않는 데이터셋: COCO GT 생성 생략")

# --- 학습 기록 ---
train_losses = []
val_metrics_history = []
best_miou = 0.0
patience_counter = 0

# --- 더 효율적인 LoRA 학습 함수 ---
def train_model(model, dataloader, optimizer, criterion, epoch):
    """
    Joint 학습: Visible과 Amodal LoRA 모두 학습
    - BCE 손실 + OT 손실 조합 사용
    - Total loss = BCE(amodal) + BCE(visible) + α * Consistency + β * OT(amodal)
    """
    model.train()
    running_loss = 0.0
    running_amodal_loss = 0.0
    running_visible_loss = 0.0
    running_consistency_loss = 0.0
    running_ot_loss = 0.0
    
    print(f"\n--- 에포크 {epoch} Joint 학습 시작 (임베딩 공간 OT 손실 포함) ---")
    print(f"총 배치 수: {len(dataloader)}")
    print(f"OT 손실 가중치 (α): {OT_WEIGHT}, 거리 타입: {OT_DISTANCE_TYPE}")
    
    # 현재 학습 가능한 파라미터 수 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"현재 학습 가능한 파라미터 수: {trainable_params:,}")
    
    for i, (images, bboxes, texts, amodal_masks, visible_masks, _, annotations) in enumerate(dataloader):
        if i % 10 == 0:  # 더 자주 출력
            print(f"  배치 {i+1}/{len(dataloader)} 처리 중...")
        
        try:
            images = images.to(device)
            bboxes = bboxes.to(device)
            amodal_masks = amodal_masks.to(device)
            visible_masks = visible_masks.to(device)

            optimizer.zero_grad()

            # 모델 예측 (임베딩도 함께 반환)
            pred_amodal, _, pred_visible, _, amodal_embeddings, visible_embeddings = model(
                images, bboxes, return_embeddings=True
            )

            # Joint 학습: Amodal과 Visible 모두 학습
            # 1. BCE 손실
            loss_amodal_bce = criterion(pred_amodal, amodal_masks)
            loss_visible_bce = criterion(pred_visible, visible_masks)
            
            # 2. 일관성 손실 계산 (amodal이 visible을 포함해야 함)
            consistency_loss, consistency_details = model.compute_amodal_visible_consistency_loss(
                pred_amodal, pred_visible
            )
            
            # 3. 임베딩 공간 OT 손실 계산 (구조적 매칭)
            # Visible과 Amodal 디코더의 쿼리 임베딩 간의 의미적 일관성 강제
            # Q×Q 매트릭스로 계산하여 계산 효율성 극대화
            ot_loss_embedding = model.compute_optimal_transport_loss_embedding(
                visible_embeddings,
                amodal_embeddings,
                sinkhorn_blur=OT_SINKHORN_BLUR,
                distance_type=OT_DISTANCE_TYPE
            )
            
            # Total loss = BCE(amodal) + BCE(visible) + Consistency + α * OT_embedding
            # OT_embedding: Visible과 Amodal 임베딩 간의 구조적 매칭 손실
            total_loss = (
                loss_amodal_bce + 
                loss_visible_bce + 
                CONSISTENCY_WEIGHT * consistency_loss + 
                OT_WEIGHT * ot_loss_embedding
            )
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_amodal_loss += loss_amodal_bce.item()
            running_visible_loss += loss_visible_bce.item()
            running_consistency_loss += consistency_loss.item()
            running_ot_loss += ot_loss_embedding.item()
            
            if (i + 1) % 20 == 0:
                print(f"  Batch {i+1}/{len(dataloader)}, Total Loss: {total_loss.item():.4f}")
                print(f"    - Amodal BCE: {loss_amodal_bce.item():.4f}, Visible BCE: {loss_visible_bce.item():.4f}")
                print(f"    - Consistency: {consistency_loss.item():.4f}, OT_embedding: {ot_loss_embedding.item():.4f}")
        
        except Exception as e:
            print(f"  배치 {i+1} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

    avg_loss = running_loss / len(dataloader)
    avg_amodal_loss = running_amodal_loss / len(dataloader)
    avg_visible_loss = running_visible_loss / len(dataloader)
    avg_consistency_loss = running_consistency_loss / len(dataloader)
    avg_ot_loss = running_ot_loss / len(dataloader)
    
    print(f"--- 에포크 {epoch} 평균 손실 ---")
    print(f"  Total: {avg_loss:.4f}")
    print(f"  Amodal BCE: {avg_amodal_loss:.4f}, Visible BCE: {avg_visible_loss:.4f}")
    print(f"  Consistency: {avg_consistency_loss:.4f}, OT: {avg_ot_loss:.4f}")
    return avg_loss

# --- 평가 함수 ---
def evaluate_model(model, dataloader, criterion, model_name, epoch, coco_gt=None):
    """모델 평가 함수"""
    model.eval()
    running_loss = 0.0
    
    # 통합된 디렉토리 구조 사용
    epoch_logs_dir = os.path.join(logs_dir, f"epoch_{epoch}")
    os.makedirs(epoch_logs_dir, exist_ok=True)
    
    evaluator_amodal = AmodalEvaluationMetrics(prefix="amodal_")
    evaluator_visible = AmodalEvaluationMetrics(prefix="visible_")
    
    print(f"\n--- {model_name} 에포크 {epoch} 평가 시작 ---")
    print(f"Validation 데이터셋 크기: {len(dataloader)} 배치")
    print(f"COCO GT 사용 여부: {'Yes' if coco_gt is not None else 'No'}")

    with torch.no_grad():
        for i, (images, bboxes, texts, amodal_masks, visible_masks, _, annotations) in enumerate(dataloader):
            if i % 50 == 0:  # 진행 상황 출력
                print(f"  평가 진행: {i}/{len(dataloader)} 배치")
            
            images, bboxes = images.to(device), bboxes.to(device)
            amodal_masks, visible_masks = amodal_masks.to(device), visible_masks.to(device)

            try:
                # 배치 단위로 정보 전달 (평가 시에는 Text RA 저장하지 않음)
                # 모델 예측 (순수 LoRA, Text RA 없음)
                # 평가 시에는 임베딩이 필요 없으므로 기본 반환값만 사용
                pred_amodal_logits, iou_amodal, pred_visible_logits, iou_visible = model(images, bboxes, return_embeddings=False)

                loss_amodal = criterion(pred_amodal_logits, amodal_masks)
                loss_visible = criterion(pred_visible_logits, visible_masks)
                running_loss += (loss_amodal + loss_visible).item()

                pred_amodal_masks = torch.sigmoid(pred_amodal_logits)
                pred_visible_masks = torch.sigmoid(pred_visible_logits)
                
                for j in range(images.size(0)):
                    ann_dict = {k: v[j].item() for k, v in annotations.items() if torch.is_tensor(v[j])}
                    ann_dict['file_name'] = annotations['file_name'][j]

                    pred_amodal_np = pred_amodal_masks[j].squeeze().cpu().numpy()
                    pred_visible_np = pred_visible_masks[j].squeeze().cpu().numpy()
                    gt_amodal_np = amodal_masks[j].squeeze().cpu().numpy()
                    gt_visible_np = visible_masks[j].squeeze().cpu().numpy()

                    evaluator_amodal.add_batch([pred_amodal_np], [gt_amodal_np], [ann_dict], [iou_amodal[j].item()])
                    evaluator_visible.add_batch([pred_visible_np], [gt_visible_np], [ann_dict], [iou_visible[j].item()])
                    
            except Exception as e:
                print(f"  배치 {i} 처리 중 오류: {e}")
                continue

    print(f"  평가 데이터 처리 완료: {len(evaluator_amodal.iou_scores)}개 샘플")
    
    avg_loss = running_loss / len(dataloader)
    
    # COCO GT가 없는 경우 기본 메트릭만 계산
    if coco_gt is None:
        print("  COCO GT가 없어 기본 메트릭만 계산합니다.")
        metrics_amodal = evaluator_amodal.compute_basic_metrics()
        metrics_visible = evaluator_visible.compute_basic_metrics()
    else:
        print("  COCO 메트릭 계산 중...")
        try:
            metrics_amodal = evaluator_amodal.print_metrics(coco_gt, epoch)
            metrics_visible = evaluator_visible.print_metrics(coco_gt, epoch)
        except Exception as e:
            print(f"  COCO 메트릭 계산 실패: {e}")
            print("  기본 메트릭으로 fallback합니다.")
            metrics_amodal = evaluator_amodal.compute_basic_metrics()
            metrics_visible = evaluator_visible.compute_basic_metrics()
    
    all_metrics = {**metrics_amodal, **metrics_visible, 'eval_loss': avg_loss}
    
    metrics_path = os.path.join(epoch_logs_dir, f"metrics_epoch_{epoch}.json")
    save_metrics_to_file(all_metrics, metrics_path)
    
    print(f"--- {model_name} 에포크 {epoch} 평가 완료 ---")
    return avg_loss, all_metrics

# --- 최종 시각화 함수 ---
def create_final_visualizations(model, dataloader, output_dir, max_samples=None):
    """최종 평가 시 전체 데이터셋에 대한 시각화 생성"""
    model.eval()
    
    final_vis_dir = os.path.join(output_dir, "visualizations", "final_evaluation")
    os.makedirs(final_vis_dir, exist_ok=True)
    
    if max_samples is None:
        print(f"\n--- 최종 시각화 생성 시작 (전체 데이터셋) ---")
    else:
        print(f"\n--- 최종 시각화 생성 시작 (최대 {max_samples}개 샘플) ---")
    
    sample_count = 0
    with torch.no_grad():
        for i, (images, bboxes, texts, amodal_masks, visible_masks, _, annotations) in enumerate(dataloader):
            if max_samples is not None and sample_count >= max_samples:
                break
                
            images, bboxes = images.to(device), bboxes.to(device)
            amodal_masks, visible_masks = amodal_masks.to(device), visible_masks.to(device)

            # 모델 예측 (평가 시에는 임베딩 불필요)
            pred_amodal_logits, iou_amodal, pred_visible_logits, iou_visible = model(images, bboxes, return_embeddings=False)
            pred_amodal_masks = torch.sigmoid(pred_amodal_logits)
            pred_visible_masks = torch.sigmoid(pred_visible_logits)
            
            for j in range(images.size(0)):
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
                ann_dict = {k: v[j].item() for k, v in annotations.items() if torch.is_tensor(v[j])}
                ann_dict['file_name'] = annotations['file_name'][j]
                # class_name은 문자열이므로 별도로 추가
                if 'class_name' in annotations:
                    ann_dict['class_name'] = annotations['class_name'][j]

                pred_amodal_np = pred_amodal_masks[j].squeeze().cpu().numpy()
                pred_visible_np = pred_visible_masks[j].squeeze().cpu().numpy()
                gt_amodal_np = amodal_masks[j].squeeze().cpu().numpy()
                gt_visible_np = visible_masks[j].squeeze().cpu().numpy()

                # IoU 계산
                def compute_iou(pred, gt, threshold=0.5):
                    pred_binary = (pred > threshold).astype(np.uint8)
                    gt_binary = gt.astype(np.uint8)
                    intersection = np.logical_and(pred_binary, gt_binary).sum()
                    union = np.logical_or(pred_binary, gt_binary).sum()
                    return intersection / (union + 1e-8)
                
                amodal_iou_score = compute_iou(pred_amodal_np, gt_amodal_np)
                visible_iou_score = compute_iou(pred_visible_np, gt_visible_np)

                # 시각화 저장 (inference 스타일)
                vis_path = os.path.join(final_vis_dir, f"final_sample_{ann_dict['image_id']}_{ann_dict['id']}.png")
                create_visualization_inference_style(
                    images[j], bboxes[j].cpu().numpy(), gt_amodal_np, gt_visible_np,
                    pred_amodal_np, pred_visible_np, texts[j], 
                    amodal_iou_score, visible_iou_score, vis_path
                )
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    if max_samples is not None:
                        print(f"  시각화 진행: {sample_count}/{max_samples}")
                    else:
                        print(f"  시각화 진행: {sample_count}개 완료")

    print(f"--- 최종 시각화 생성 완료: {sample_count}개 샘플 저장됨 ---")
    print(f"시각화 저장 경로: {final_vis_dir}")

def create_visualization_inference_style(image_tensor, bbox, gt_amodal, gt_visible, 
                                        pred_amodal, pred_visible, text, amodal_iou, visible_iou, save_path):
    """inference 스타일의 시각화 생성"""
    # 이미지 텐서를 numpy로 변환
    image_array = denormalize_image(image_tensor)
    
    # 마스크 이진화
    pred_amodal_binary = (pred_amodal > 0.5).astype(np.float32)
    pred_visible_binary = (pred_visible > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 원본 이미지
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title('Original Image')
    rect = plt.Rectangle((bbox[0], bbox[1]), 
                       bbox[2]-bbox[0], bbox[3]-bbox[1], 
                       fill=False, color='red', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].axis('off')
    
    # 원본 이미지 크기에 맞게 마스크들을 리사이즈
    original_height, original_width = image_array.shape[:2]
    
    # GT 마스크들을 원본 이미지 크기로 리사이즈
    gt_amodal_resized = cv2.resize(gt_amodal, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    gt_visible_resized = cv2.resize(gt_visible, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 예측된 마스크들도 원본 이미지 크기로 리사이즈
    pred_amodal_resized = cv2.resize(pred_amodal_binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    pred_visible_resized = cv2.resize(pred_visible_binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # GT Amodal (원본 이미지 위에)
    axes[0, 1].imshow(image_array)
    axes[0, 1].imshow(gt_amodal_resized, alpha=0.6, cmap='Reds')
    axes[0, 1].set_title('GT Amodal')
    axes[0, 1].axis('off')
    
    # Pred Amodal (원본 이미지 위에)
    axes[0, 2].imshow(image_array)
    axes[0, 2].imshow(pred_amodal_resized, alpha=0.6, cmap='Blues')
    axes[0, 2].set_title(f'Pred Amodal\nIoU: {amodal_iou:.3f}')
    axes[0, 2].axis('off')
    
    # GT Visible (원본 이미지 위에)
    axes[1, 0].imshow(image_array)
    axes[1, 0].imshow(gt_visible_resized, alpha=0.6, cmap='Reds')
    axes[1, 0].set_title('GT Visible')
    axes[1, 0].axis('off')
    
    # Pred Visible (원본 이미지 위에)
    axes[1, 1].imshow(image_array)
    axes[1, 1].imshow(pred_visible_resized, alpha=0.6, cmap='Blues')
    axes[1, 1].set_title(f'Pred Visible\nIoU: {visible_iou:.3f}')
    axes[1, 1].axis('off')
    
    # Invisible 영역 (원본 이미지 위에)
    invisible_mask = np.clip(pred_amodal_resized - pred_visible_resized, 0, 1)
    axes[1, 2].imshow(image_array)
    axes[1, 2].imshow(invisible_mask, alpha=0.6, cmap='Oranges')
    axes[1, 2].set_title('Invisible Region')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'{text}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# --- 메인 LoRA 학습 및 평가 루프 ---
if __name__ == '__main__':
    print("\n=== LoRA 어댑터 기반 Amodal Instance Segmentation 학습 시작 ===")
    print(f"사용 데이터셋: {DATASET_NAME.upper()}")
    print(f"\nLoRA 설정: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    
    # --- 순수 LoRA 모델, 옵티마이저, 체크포인트 매니저 초기화 ---
    model = BaselineModel(
        lora_rank=LORA_RANK, 
        lora_alpha=LORA_ALPHA
    ).to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE_STEP1
    )
    
    checkpoint_manager = CheckpointManager(checkpoints_dir)
    
    # 파라미터 수 출력
    param_stats = count_parameters(model, trainable_only=False)
    print(f"\n=== LoRA 모델 파라미터 분석 ===")
    print(f"전체 파라미터: {param_stats['total_params']:,}")
    print(f"학습 가능한 파라미터: {param_stats['trainable_params']:,}")
    print(f"학습 가능한 비율: {param_stats['trainable_ratio']:.2%}")
    print("=" * 50)

    train_losses, val_metrics_history = [], []
    best_amodal_miou = 0.0
    best_visible_miou = 0.0
    best_combined_score = 0.0  # amodal + visible mIoU 조합 점수
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\n에포크 {epoch}/{EPOCHS}\n{'='*60}")

        # Joint 학습 적용
        train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        
        eval_loss, metrics = evaluate_model(model, val_dataloader, criterion, "LoRA BaselineModel", epoch, val_coco_gt)
        val_metrics_history.append(metrics)
        
        # 다양한 성능 지표 추적
        current_amodal_miou = metrics.get('amodal_mIoU', 0.0)
        current_visible_miou = metrics.get('visible_mIoU', 0.0)
        current_combined_score = current_amodal_miou + current_visible_miou
        
        # 개선된 성능 판단: amodal mIoU 또는 combined score가 개선되면 best로 간주
        is_best = (current_amodal_miou > best_amodal_miou) or (current_combined_score > best_combined_score)
        
        if is_best:
            best_amodal_miou = max(best_amodal_miou, current_amodal_miou)
            best_visible_miou = max(best_visible_miou, current_visible_miou)
            best_combined_score = max(best_combined_score, current_combined_score)
            patience_counter = 0
            print(f"✓ 성능 개선! amodal mIoU: {current_amodal_miou:.4f}, visible mIoU: {current_visible_miou:.4f}")
        else:
            patience_counter += 1
            print(f"성능 개선 없음 (patience: {patience_counter}/{PATIENCE})")
        
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, metrics, is_best, "naive_lora_model")
        
        # curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
        # VisualizationUtils.save_training_curves(train_losses, val_metrics_history, curves_path)
        
        print(f"\n현재 최고 성능:")
        print(f"  - amodal mIoU: {best_amodal_miou:.4f}")
        print(f"  - visible mIoU: {best_visible_miou:.4f}")
        print(f"  - combined score: {best_combined_score:.4f}")
        
        # 조기 종료 조건 완화: 더 많은 에포크 허용
        if patience_counter >= PATIENCE and epoch >= 5:  # 최소 5 에포크는 학습
            print(f"\n{PATIENCE} 에포크 동안 성능 개선이 없어 학습을 조기 종료합니다.")
            break

    print(f"\n=== LoRA 학습 완료 ===\n최종 최고 amodal mIoU: {best_amodal_miou:.4f}")
    
    # 최고 성능 모델 로드 및 최종 평가
    best_checkpoint = checkpoint_manager.get_best_checkpoint_path("cross_attention_model")
    if best_checkpoint:
        print(f"\n최고 성능 Cross-Attention 모델로 최종 평가: {best_checkpoint}")
        checkpoint_manager.load_checkpoint(model, optimizer, best_checkpoint)
        final_loss, final_metrics = evaluate_model(
            model, val_dataloader, criterion, 
            "LoRA Baseline Model", "final", val_coco_gt
        )
        
        print(f"\n=== LoRA 모델 최종 평가 결과 ===")
        print(f"mIoU: {final_metrics.get('mIoU', 0):.4f}")
        print(f"AP: {final_metrics.get('AP', 0):.4f}")
        print(f"AP50: {final_metrics.get('AP50', 0):.4f}")
        print(f"AP75: {final_metrics.get('AP75', 0):.4f}")
        print(f"AR100: {final_metrics.get('AR@100', 0):.4f}")
        
        # 최종 시각화 생성 (학습 데이터 전체)
        create_final_visualizations(model, train_dataloader, OUTPUT_DIR, max_samples=None)
    
    print(f"\n=== LoRA 어댑터 기반 경량 미세 조정 완료 ===")
    print(f"모든 결과가 {OUTPUT_DIR}에 저장되었습니다.")
    
    # 임시 COCO 파일 정리 (D2SA, MP3D 데이터셋인 경우에만)
    if DATASET_NAME in ['d2sa', 'mp3d']:
        try:
            os.unlink(train_coco_file.name)
            os.unlink(val_coco_file.name)
            print("임시 COCO 파일 정리 완료")
        except:
            pass
