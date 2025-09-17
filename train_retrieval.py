'''''
Amodal / Visible decoder 따로 학습
Retrieval 적용
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
from baseline_model_retrieval import BaselineModelRetrieval
from evaluation_metrics import AmodalEvaluationMetrics, compute_occlusion_metrics
from utils import CheckpointManager, VisualizationUtils, denormalize_image, create_output_directories, save_metrics_to_file

# --- 하이퍼파라미터 설정 ---
EPOCHS = 6
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
PATIENCE = 3  # Early stopping을 위한 patience 증가

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
    OUTPUT_DIR = "./outputs/D2SA_aug_amodal_visible_retrieval_250917_backup_test"
    
elif DATASET_NAME == 'pix2gestalt':
    PIX2GESTALT_ROOT = "/root/datasets/pix2gestalt_occlusions_release"
    OUTPUT_DIR = "./outputs/Pix2Gestalt_amodal_visible_retrieval_250917"
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
        max_samples=None  # 전체 데이터 사용
    )
    val_dataset = D2SADataset(
        annotation_file=VAL_ANNOTATION_FILE,  # Validation set
        image_dir=IMAGE_DIR, 
        transform=transform,
        max_samples=None  # 전체 데이터 사용
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
# Retrieval 기능 활성화 (amodal 예측 시에만 사용)
RETRIEVAL_ENABLED = True
model_baseline = BaselineModelRetrieval(retrieval_enabled=RETRIEVAL_ENABLED, output_dir=OUTPUT_DIR).to(device)
print(f"모델이 {device}로 이동되었습니다.")
print(f"Retrieval 기능: {'활성화 (amodal 예측 시에만)' if RETRIEVAL_ENABLED else '비활성화'}")
print(f"모델 파라미터 디바이스 확인: {next(model_baseline.parameters()).device}")


# --- 손실 함수 및 옵티마이저 정의 ---
criterion = nn.BCEWithLogitsLoss()
optimizer_baseline = optim.Adam(
    filter(lambda p: p.requires_grad, model_baseline.parameters()), 
    lr=LEARNING_RATE
)

# --- 통합된 출력 디렉토리 구조 생성 ---
# checkpoints, logs, visualizations, retrieval_debug 폴더를 루트에 생성
checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
logs_dir = os.path.join(OUTPUT_DIR, "logs")
visualizations_dir = os.path.join(OUTPUT_DIR, "visualizations")
retrieval_debug_dir = os.path.join(OUTPUT_DIR, "retrieval_debug")

for dir_path in [checkpoints_dir, logs_dir, visualizations_dir, retrieval_debug_dir]:
    os.makedirs(dir_path, exist_ok=True)

print(f"통합된 출력 디렉토리 구조 생성:")
print(f"  - Checkpoints: {checkpoints_dir}")
print(f"  - Logs: {logs_dir}")
print(f"  - Visualizations: {visualizations_dir}")
print(f"  - Retrieval Debug: {retrieval_debug_dir}")

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
    
    # 디바이스 확인
    print(f"모델 디바이스: {next(model.parameters()).device}")
    print(f"입력 이미지 디바이스: {images.device if 'images' in locals() else 'N/A'}")
    
    for i, (images, bboxes, texts, amodal_masks, visible_masks, _, _) in enumerate(dataloader):
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

    with torch.no_grad():
        for i, (images, bboxes, texts, amodal_masks, visible_masks, _, annotations) in enumerate(dataloader):
            images, bboxes = images.to(device), bboxes.to(device)
            amodal_masks, visible_masks = amodal_masks.to(device), visible_masks.to(device)

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

    avg_loss = running_loss / len(dataloader)
    metrics_amodal = evaluator_amodal.print_metrics(coco_gt, epoch)
    metrics_visible = evaluator_visible.print_metrics(coco_gt, epoch)
    
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
    print("\n계층적 학습 전략 (RAG 사용 최적화):")
    print("  - 에포크 1-3: Visible 디코더만 학습 (Retrieval 비활성화, 빠른 학습)")
    print("  - 에포크 4-6: Amodal 디코더만 학습 (Retrieval 활성화, amodal 성능 향상)")
    print("  - 에포크 7-10: 전체 디코더 미세조정 + Retrieval 활성화 + 일관성 제약")
    print("\nRAG 사용 전략:")
    print("  - Visible mask 예측: Retrieval 미사용 (빠른 학습)")
    print("  - Amodal mask 예측: Retrieval 사용 (성능 향상)")
    
    # --- 모델, 옵티마이저, 체크포인트 매니저 초기화 ---
    model = BaselineModelRetrieval(output_dir=OUTPUT_DIR).to(device)
    print(f"메인 학습 모델이 {device}로 이동되었습니다.")
    print(f"메인 모델 파라미터 디바이스 확인: {next(model.parameters()).device}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    checkpoint_manager = CheckpointManager(os.path.join(OUTPUT_DIR, "checkpoints"))

    train_losses, val_metrics_history = [], []
    best_amodal_miou = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}\n에포크 {epoch}/{EPOCHS}\n{'='*60}")

        # 계층적 학습 전략 적용
        if epoch <= 3:
            # 1-3 에포크: Visible 디코더만 학습 (Retrieval 비활성화)
            training_stage = 'visible_only'
            model.set_training_stage('visible_only')
            # Visible 학습 시에는 retrieval 비활성화 (더 빠른 학습)
            if hasattr(model, 'retrieval_enabled'):
                model.retrieval_enabled = False
                print(f"[INFO] 에포크 {epoch}: Visible 학습 - Retrieval 비활성화")
        elif epoch <= 6:
            # 4-6 에포크: Amodal 디코더만 학습 (visible 고정, Retrieval 활성화)
            training_stage = 'amodal_only'
            model.set_training_stage('amodal_only')
            # Amodal 학습 시에는 retrieval 활성화
            if hasattr(model, 'retrieval_enabled'):
                model.retrieval_enabled = True
                print(f"[INFO] 에포크 {epoch}: Amodal 학습 - Retrieval 활성화")
        else:
            # 7-10 에포크: 전체 디코더 미세조정 (Retrieval 활성화)
            training_stage = 'joint'
            model.set_training_stage('joint')
            # Joint 학습 시에도 retrieval 활성화
            if hasattr(model, 'retrieval_enabled'):
                model.retrieval_enabled = True
                print(f"[INFO] 에포크 {epoch}: Joint 학습 - Retrieval 활성화")

        train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, training_stage)
        train_losses.append(train_loss)
        
        eval_loss, metrics = evaluate_model(model, val_dataloader, criterion, "BaselineModel", epoch, val_coco_gt)
        val_metrics_history.append(metrics)
        
        # Retrieval 디버깅 정보 저장
        if hasattr(model, 'save_retrieval_debug_info'):
            epoch_logs_dir = os.path.join(logs_dir, f"epoch_{epoch}")
            os.makedirs(epoch_logs_dir, exist_ok=True)
            model.save_retrieval_debug_info(epoch_logs_dir)
            
            # Retrieval 통계 출력
            retrieval_stats = model.get_retrieval_stats()
            print(f"\n[RETRIEVAL STATS] 에포크 {epoch}:")
            print(f"  - Amodal retrievals: {retrieval_stats['total_amodal_retrievals']}")
            print(f"  - Visible retrievals: {retrieval_stats['total_visible_retrievals']}")
            print(f"  - Success rate: {retrieval_stats['retrieval_success_count']}/{retrieval_stats['retrieval_success_count'] + retrieval_stats['retrieval_failure_count']}")
            if retrieval_stats['avg_similarity_scores']:
                print(f"  - Avg similarity: {retrieval_stats['avg_similarity']:.4f}")
                print(f"  - Max similarity: {retrieval_stats['max_similarity']:.4f}")
                print(f"  - Min similarity: {retrieval_stats['min_similarity']:.4f}")
        
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
        
        # 최종 Retrieval 디버깅 정보 저장
        if hasattr(model, 'save_retrieval_debug_info'):
            final_logs_dir = os.path.join(logs_dir, "final_evaluation")
            os.makedirs(final_logs_dir, exist_ok=True)
            model.save_retrieval_debug_info(final_logs_dir)
            
            # 최종 Retrieval 통계 출력
            final_retrieval_stats = model.get_retrieval_stats()
            print(f"\n=== 최종 Retrieval 통계 ===")
            print(f"총 Amodal retrievals: {final_retrieval_stats['total_amodal_retrievals']}")
            print(f"총 Visible retrievals: {final_retrieval_stats['total_visible_retrievals']}")
            print(f"Retrieval 성공률: {final_retrieval_stats['retrieval_success_count']}/{final_retrieval_stats['retrieval_success_count'] + final_retrieval_stats['retrieval_failure_count']}")
            if final_retrieval_stats['avg_similarity_scores']:
                print(f"평균 유사도: {final_retrieval_stats['avg_similarity']:.4f}")
                print(f"최대 유사도: {final_retrieval_stats['max_similarity']:.4f}")
                print(f"최소 유사도: {final_retrieval_stats['min_similarity']:.4f}")
        
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