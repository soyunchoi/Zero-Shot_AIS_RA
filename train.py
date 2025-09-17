'''''
Amodal / Visible decoder 따로 학습
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

# 커스텀 모듈 임포트
from d2sa_dataset import D2SADataset
from pix2gestalt_dataset import Pix2GestaltDataset
from baseline_model import BaselineModel
from evaluation_metrics import AmodalEvaluationMetrics, compute_occlusion_metrics
from utils import CheckpointManager, VisualizationUtils, denormalize_image, create_output_directories, save_metrics_to_file

# --- 하이퍼파라미터 설정 ---
EPOCHS = 6
BATCH_SIZE = 4  # 배치 크기 증가 (메모리 허용 시)
LEARNING_RATE = 1e-4
PATIENCE = 3  # Early stopping을 위한 patience

# --- 데이터셋 설정 ---
# 사용할 데이터셋 선택: 'd2sa' 또는 'pix2gestalt'
DATASET_NAME = 'd2sa'  # 'd2sa' 또는 'pix2gestalt'

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
    OUTPUT_DIR = "./outputs/D2SA_aug_amodal_visible_head_250916"
    
elif DATASET_NAME == 'pix2gestalt':
    PIX2GESTALT_ROOT = "/root/datasets/pix2gestalt_occlusions_release"
    OUTPUT_DIR = "./outputs/Pix2Gestalt_amodal_visible_head_250915"
    MAX_SAMPLES = 3000  # 처음 3,000장 사용
    
else:
    raise ValueError(f"지원하지 않는 데이터셋: {DATASET_NAME}")

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
elif DATASET_NAME == 'pix2gestalt':
    train_dataset = Pix2GestaltDataset(
        data_root=PIX2GESTALT_ROOT,
        transform=transform,
        max_samples=MAX_SAMPLES,
        split_ratio=0.8
    )
    train_dataset.set_split('train')
    
    val_dataset = Pix2GestaltDataset(
        data_root=PIX2GESTALT_ROOT,
        transform=transform,
        max_samples=MAX_SAMPLES,
        split_ratio=0.8
    )
    val_dataset.set_split('val')

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"Training set: {len(train_dataset)}개의 샘플 로드 완료.")

# --- Validation Dataset 및 DataLoader 초기화 ---
print("Validation 데이터셋 초기화 중...")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Validation set: {len(val_dataset)}개의 샘플 로드 완료.")

# --- 모델 초기화 ---
print("모델 초기화 중...")
model_baseline = BaselineModel().to(device)


# --- 손실 함수 및 옵티마이저 정의 ---
criterion = nn.BCEWithLogitsLoss()
optimizer_baseline = optim.Adam(
    filter(lambda p: p.requires_grad, model_baseline.parameters()), 
    lr=LEARNING_RATE
)

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

# D2SA 데이터셋의 경우에만 COCO GT 생성
if DATASET_NAME == 'd2sa':
    # Training set용 COCO GT 생성
    train_coco_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    train_coco_data = {
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
        'images': val_dataset.coco['images'],
        'annotations': val_dataset.coco['annotations'], 
        'categories': val_dataset.coco['categories']
    }
    json.dump(val_coco_data, val_coco_file, indent=2)
    val_coco_file.close()

    val_coco_gt = COCO(val_coco_file.name)
    print(f"Validation COCO GT 생성 완료: {len(val_dataset.coco['images'])}개 이미지, {len(val_dataset.coco['annotations'])}개 어노테이션")
else:
    # Pix2Gestalt 데이터셋의 경우 COCO GT 생성하지 않음
    train_coco_gt = None
    val_coco_gt = None
    train_coco_file = None
    val_coco_file = None
    print("Pix2Gestalt 데이터셋 사용: COCO GT 생성 생략")

# --- 학습 기록 ---
train_losses = []
val_metrics_history = []
best_miou = 0.0
patience_counter = 0

# --- 학습 함수 ---
def train_model(model, dataloader, optimizer, criterion, epoch, training_stage='joint'):
    """모델 학습 함수"""
    model.train()
    running_loss = 0.0
    print(f"\n--- 에포크 {epoch} 학습 시작 (단계: {training_stage}) ---")
    print(f"총 배치 수: {len(dataloader)}")
    
    for i, (images, bboxes, texts, amodal_masks, visible_masks, _, _) in enumerate(dataloader):
        if i % 10 == 0:  # 더 자주 출력
            print(f"  배치 {i+1}/{len(dataloader)} 처리 중...")
        
        try:
            images = images.to(device)
            bboxes = bboxes.to(device)
            amodal_masks = amodal_masks.to(device)
            visible_masks = visible_masks.to(device)

            optimizer.zero_grad()

            # 모델 예측
            pred_amodal, _, pred_visible, _ = model(images, bboxes)

            # 기본 손실 계산
            loss_amodal = criterion(pred_amodal, amodal_masks)
            loss_visible = criterion(pred_visible, visible_masks)
            
            # 일관성 손실 계산 (amodal이 visible을 포함해야 함)
            consistency_loss = model.compute_consistency_loss(pred_amodal, pred_visible)
            
            # 단계별 손실 가중치 조정
            if training_stage == 'visible_only':
                total_loss = loss_visible  # visible만 학습
            elif training_stage == 'amodal_only':
                total_loss = loss_amodal   # amodal만 학습
            else:  # joint training
                total_loss = loss_amodal + loss_visible + 0.1 * consistency_loss
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            
            if (i + 1) % 20 == 0:
                if training_stage == 'visible_only':
                    print(f"  Batch {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Visible: {loss_visible.item():.4f})")
                elif training_stage == 'amodal_only':
                    print(f"  Batch {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Amodal: {loss_amodal.item():.4f})")
                else:
                    print(f"  Batch {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Amodal: {loss_amodal.item():.4f}, Visible: {loss_visible.item():.4f}, Consistency: {consistency_loss.item():.4f})")
        
        except Exception as e:
            print(f"  배치 {i+1} 처리 중 오류: {e}")
            continue

    avg_loss = running_loss / len(dataloader)
    print(f"--- 에포크 {epoch} 평균 학습 손실: {avg_loss:.4f} ---")
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
                pred_amodal_logits, iou_amodal, pred_visible_logits, iou_visible = model(images, bboxes)

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

            pred_amodal_logits, iou_amodal, pred_visible_logits, iou_visible = model(images, bboxes)
            pred_amodal_masks = torch.sigmoid(pred_amodal_logits)
            pred_visible_masks = torch.sigmoid(pred_visible_logits)
            
            for j in range(images.size(0)):
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
                ann_dict = {k: v[j].item() for k, v in annotations.items() if torch.is_tensor(v[j])}
                ann_dict['file_name'] = annotations['file_name'][j]

                pred_amodal_np = pred_amodal_masks[j].squeeze().cpu().numpy()
                pred_visible_np = pred_visible_masks[j].squeeze().cpu().numpy()
                gt_amodal_np = amodal_masks[j].squeeze().cpu().numpy()
                gt_visible_np = visible_masks[j].squeeze().cpu().numpy()

                # 시각화 저장
                vis_path = os.path.join(final_vis_dir, f"final_sample_{ann_dict['image_id']}_{ann_dict['id']}.png")
                VisualizationUtils.save_prediction_visualization(
                    denormalize_image(images[j]),
                    gt_amodal_np, gt_visible_np, 
                    pred_amodal_np, pred_visible_np,
                    vis_path, bboxes[j].cpu().numpy(), texts[j]
                )
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    if max_samples is not None:
                        print(f"  시각화 진행: {sample_count}/{max_samples}")
                    else:
                        print(f"  시각화 진행: {sample_count}개 완료")

    print(f"--- 최종 시각화 생성 완료: {sample_count}개 샘플 저장됨 ---")
    print(f"시각화 저장 경로: {final_vis_dir}")

# --- 메인 학습 및 평가 루프 ---
if __name__ == '__main__':
    print("\n=== Enhanced Amodal Instance Segmentation 학습 시작 ===")
    print(f"사용 데이터셋: {DATASET_NAME.upper()}")
    if DATASET_NAME == 'pix2gestalt':
        print(f"최대 샘플 수: {MAX_SAMPLES:,}개")
    print("\n계층적 학습 전략:")
    print("  - 에포크 1-2: Visible 디코더만 학습 (더 쉬운 태스크)")
    print("  - 에포크 3-4: Amodal 디코더만 학습 (visible 고정)")
    print("  - 에포크 5-6: 전체 디코더 미세조정 + 일관성 제약")
    
    # --- 모델, 옵티마이저, 체크포인트 매니저 초기화 ---
    model = BaselineModel().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    checkpoint_manager = CheckpointManager(checkpoints_dir)

    train_losses, val_metrics_history = [], []
    best_amodal_miou = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\n에포크 {epoch}/{EPOCHS}\n{'='*60}")

        # 계층적 학습 전략 적용
        if epoch <= 2:
            # 1-2 에포크: Visible 디코더만 학습
            training_stage = 'visible_only'
            model.set_training_stage('visible_only')
        elif epoch <= 4:
            # 3-4 에포크: Amodal 디코더만 학습 (visible 고정)
            training_stage = 'amodal_only'
            model.set_training_stage('amodal_only')
        else:
            # 5-6 에포크: 전체 디코더 미세조정
            training_stage = 'joint'
            model.set_training_stage('joint')

        train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, training_stage)
        train_losses.append(train_loss)
        
        eval_loss, metrics = evaluate_model(model, val_dataloader, criterion, "BaselineModel", epoch, val_coco_gt)
        val_metrics_history.append(metrics)
        
        current_amodal_miou = metrics.get('amodal_mIoU', 0.0)
        is_best = current_amodal_miou > best_amodal_miou
        if is_best:
            best_amodal_miou = current_amodal_miou
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, metrics, is_best, "baseline")
        
        # curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
        # VisualizationUtils.save_training_curves(train_losses, val_metrics_history, curves_path)
        
        print(f"\n현재 최고 amodal mIoU: {best_amodal_miou:.4f} (에포크 {checkpoint_manager.best_epoch})")
        
        if patience_counter >= PATIENCE:
            print(f"\n{PATIENCE} 에포크 동안 성능 개선이 없어 학습을 조기 종료합니다.")
            break

    print(f"\n=== 학습 완료 ===\n최종 최고 amodal mIoU: {best_amodal_miou:.4f}")
    
    # 최고 성능 모델 로드 및 최종 평가
    best_checkpoint = checkpoint_manager.get_best_checkpoint_path("baseline")
    if best_checkpoint:
        print(f"\n        최고 성능 모델로 최종 평가: {best_checkpoint}")
        checkpoint_manager.load_checkpoint(model, optimizer, best_checkpoint)
        final_loss, final_metrics = evaluate_model(
            model, val_dataloader, criterion, 
            "Baseline Model", "final", val_coco_gt
        )
        
        print(f"\n=== 최종 평가 결과 ===")
        print(f"mIoU: {final_metrics.get('mIoU', 0):.4f}")
        print(f"AP: {final_metrics.get('AP', 0):.4f}")
        print(f"AP50: {final_metrics.get('AP50', 0):.4f}")
        print(f"AP75: {final_metrics.get('AP75', 0):.4f}")
        print(f"AR100: {final_metrics.get('AR@100', 0):.4f}")
        
        # 최종 시각화 생성 (학습 데이터 전체)
        create_final_visualizations(model, train_dataloader, OUTPUT_DIR, max_samples=None)
    
    print(f"\n모든 결과가 {OUTPUT_DIR}에 저장되었습니다.")
    
    # 임시 COCO 파일 정리 (D2SA 데이터셋인 경우에만)
    if DATASET_NAME == 'd2sa':
        try:
            os.unlink(train_coco_file.name)
            os.unlink(val_coco_file.name)
            print("임시 COCO 파일 정리 완료")
        except:
            pass
