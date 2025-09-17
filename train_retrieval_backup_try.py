'''''
삭제해도 됨
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
    OUTPUT_DIR = "./outputs/Pix2Gestalt_amodal_visible_retrieval_250916-2"
    MAX_SAMPLES = 3000
    
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
        max_samples=None
    )
    val_dataset = D2SADataset(
        annotation_file=VAL_ANNOTATION_FILE,  # Validation set
        image_dir=IMAGE_DIR, 
        transform=transform,
        max_samples=None
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
model = BaselineModelRetrieval(retrieval_enabled=RETRIEVAL_ENABLED, output_dir=OUTPUT_DIR).to(device)
print(f"모델이 {device}로 이동되었습니다.")
print(f"Retrieval 기능: {'활성화' if RETRIEVAL_ENABLED else '비활성화'}")
print(f"모델 파라미터 디바이스 확인: {next(model.parameters()).device}")


# --- 손실 함수 및 옵티마이저 정의 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
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
checkpoint_manager = CheckpointManager(
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
        try:
            images = images.to(device)
            bboxes = bboxes.to(device)
            amodal_masks = amodal_masks.to(device)
            visible_masks = visible_masks.to(device)

            print(f"[DEBUG] 배치 {i}: images.shape={images.shape}, device={images.device}")
            print(f"[DEBUG] bboxes.shape={bboxes.shape}, device={bboxes.device}")

            # Forward pass
            optimizer.zero_grad()

            # 모델 예측
            pred_amodal, _, pred_visible, _ = model(images, bboxes)

            print(f"[DEBUG] pred_amodal.shape={pred_amodal.shape}, pred_visible.shape={pred_visible.shape}")

            # Loss 계산 (train_retrieval.py와 동일)
            if training_stage == 'amodal_only':
                # Amodal만 학습
                loss_amodal = criterion(pred_amodal, amodal_masks)
                total_loss = loss_amodal
                print(f"[DEBUG] Amodal loss: {loss_amodal.item():.4f}")
                
            elif training_stage == 'visible_only':
                # Visible만 학습
                loss_visible = criterion(pred_visible, visible_masks)
                total_loss = loss_visible
                print(f"[DEBUG] Visible loss: {loss_visible.item():.4f}")
                
            else:  # joint
                # Amodal과 Visible 모두 학습
                loss_amodal = criterion(pred_amodal, amodal_masks)
                loss_visible = criterion(pred_visible, visible_masks)
                
                # Consistency loss (amodal과 visible의 일관성)
                consistency_loss = model.compute_consistency_loss(pred_amodal, pred_visible)
                
                # 총 손실 (가중치 조정 가능)
                total_loss = loss_amodal + loss_visible + 0.1 * consistency_loss
                
                print(f"[DEBUG] Amodal loss: {loss_amodal.item():.4f}")
                print(f"[DEBUG] Visible loss: {loss_visible.item():.4f}")
                print(f"[DEBUG] Consistency loss: {consistency_loss.item():.4f}")
                print(f"[DEBUG] Total loss: {total_loss.item():.4f}")

            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            
            # Retrieval 통계 출력 (매 10 배치마다)
            if i % 10 == 0 and hasattr(model, 'get_retrieval_stats'):
                stats = model.get_retrieval_stats()
                print(f"[DEBUG] Retrieval 통계 - Amodal: {stats['total_amodal_retrievals']}, "
                      f"Visible: {stats['total_visible_retrievals']}, "
                      f"성공: {stats['retrieval_success_count']}, "
                      f"실패: {stats['retrieval_failure_count']}")
            
            # 메모리 정리
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[ERROR] 배치 {i} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

    avg_loss = running_loss / len(dataloader)
    print(f"--- 에포크 {epoch} 평균 학습 손실: {avg_loss:.4f} ---")
    return avg_loss

# --- 평가 함수 (train_retrieval.py와 동일하게 단순화) ---
def evaluate_model(model, dataloader, criterion, device, training_stage):
    """모델 평가 함수 - train_retrieval.py와 동일한 구조"""
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    print(f"검증 시작 (단계: {training_stage})...")
    
    with torch.no_grad():
        for batch_idx, (images, bboxes, texts, amodal_masks, visible_masks, _, annotations) in enumerate(dataloader):
            try:
                images = images.to(device)
                bboxes = bboxes.to(device)
                amodal_masks = amodal_masks.to(device)
                visible_masks = visible_masks.to(device)
                
                # 모델 예측
                amodal_pred, amodal_iou, visible_pred, visible_iou = model(images, bboxes)
                
                # Loss 계산 (train_retrieval.py와 동일)
                if training_stage == 'amodal_only':
                    amodal_loss = criterion(amodal_pred, amodal_masks)
                    total_loss = amodal_loss
                elif training_stage == 'visible_only':
                    visible_loss = criterion(visible_pred, visible_masks)
                    total_loss = visible_loss
                else:  # joint
                    amodal_loss = criterion(amodal_pred, amodal_masks)
                    visible_loss = criterion(visible_pred, visible_masks)
                    consistency_loss = model.compute_consistency_loss(amodal_pred, visible_pred)
                    total_loss = amodal_loss + visible_loss + 0.1 * consistency_loss
                
                val_loss += total_loss.item()
                val_batches += 1
                
            except Exception as e:
                print(f"[ERROR] 검증 배치 {batch_idx} 처리 중 오류: {e}")
                continue
    
    avg_val_loss = val_loss / max(val_batches, 1)
    return avg_val_loss

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
    print("  - 에포크 1-2: Visible 디코더만 학습 (Retrieval 비활성화, 빠른 학습)")
    print("  - 에포크 3-4: Amodal 디코더만 학습 (Retrieval 활성화, amodal 성능 향상)")
    print("  - 에포크 5-6: 전체 디코더 미세조정 + Retrieval 활성화 + 일관성 제약")
    print("\nRAG 사용 전략:")
    print("  - Visible mask 예측: Retrieval 미사용 (빠른 학습)")
    print("  - Amodal mask 예측: Retrieval 사용 (성능 향상)")
    
    # --- 옵티마이저 초기화 ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 학습 단계별 실행 (train_retrieval.py와 동일)
    all_train_losses = []
    all_val_losses = []
    
    try:
        # 1단계: Amodal 마스크만 학습
        print(f"\n{'='*50}")
        print("1단계: Amodal 마스크 학습")
        print(f"{'='*50}")
        
        model.set_training_stage('amodal_only')
        for epoch in range(1, AMODAL_EPOCHS + 1):
            print(f"\n--- 에포크 {epoch}/{AMODAL_EPOCHS} 학습 시작 (단계: amodal_only) ---")
            train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, 'amodal_only')
            val_loss = evaluate_model(model, val_dataloader, criterion, device, 'amodal_only')
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            print(f"에포크 {epoch} 완료 - 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
        
        # 2단계: Visible 마스크만 학습
        print(f"\n{'='*50}")
        print("2단계: Visible 마스크 학습")
        print(f"{'='*50}")
        
        model.set_training_stage('visible_only')
        for epoch in range(1, VISIBLE_EPOCHS + 1):
            print(f"\n--- 에포크 {epoch}/{VISIBLE_EPOCHS} 학습 시작 (단계: visible_only) ---")
            train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, 'visible_only')
            val_loss = evaluate_model(model, val_dataloader, criterion, device, 'visible_only')
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            print(f"에포크 {epoch} 완료 - 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
        
        # 3단계: Joint 학습
        print(f"\n{'='*50}")
        print("3단계: Joint 학습 (Amodal + Visible)")
        print(f"{'='*50}")
        
        model.set_training_stage('joint')
        for epoch in range(1, JOINT_EPOCHS + 1):
            print(f"\n--- 에포크 {epoch}/{JOINT_EPOCHS} 학습 시작 (단계: joint) ---")
            train_loss = train_model(model, train_dataloader, optimizer, criterion, epoch, 'joint')
            val_loss = evaluate_model(model, val_dataloader, criterion, device, 'joint')
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            print(f"에포크 {epoch} 완료 - 학습 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': all_train_losses,
            'val_losses': all_val_losses
        }, final_model_path)
        print(f"\n최종 모델 저장: {final_model_path}")
        
        # 학습 결과 요약
        print(f"\n{'='*50}")
        print("학습 완료!")
        print(f"{'='*50}")
        print(f"총 에포크: {len(all_train_losses)}")
        print(f"최종 학습 손실: {all_train_losses[-1]:.4f}")
        if all_val_losses:
            print(f"최종 검증 손실: {all_val_losses[-1]:.4f}")
        
        # Retrieval 통계 출력
        if hasattr(model, 'get_retrieval_stats'):
            stats = model.get_retrieval_stats()
            print(f"\nRetrieval 통계:")
            print(f"  - Amodal retrievals: {stats['total_amodal_retrievals']}")
            print(f"  - Visible retrievals: {stats['total_visible_retrievals']}")
            print(f"  - 성공: {stats['retrieval_success_count']}")
            print(f"  - 실패: {stats['retrieval_failure_count']}")
            if stats['avg_similarity_scores']:
                print(f"  - 평균 유사도: {stats['avg_similarity']:.4f}")
                print(f"  - 최대 유사도: {stats['max_similarity']:.4f}")
                print(f"  - 최소 유사도: {stats['min_similarity']:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시에도 현재까지의 모델 저장
        error_model_path = os.path.join(OUTPUT_DIR, "error_recovery_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'error': str(e)
        }, error_model_path)
        print(f"오류 복구용 모델 저장: {error_model_path}")
    
    finally:
        # 최종 Retrieval 디버깅 정보 저장
        if hasattr(model, 'save_retrieval_debug_info'):
            model.save_retrieval_debug_info(OUTPUT_DIR)
        
        print(f"\n학습 완료. 결과는 {OUTPUT_DIR}에 저장되었습니다.")
    
    # 임시 COCO 파일 정리 (D2SA 데이터셋인 경우에만)
    if DATASET_NAME == 'd2sa':
        try:
            os.unlink(train_coco_file.name)
            os.unlink(val_coco_file.name)
            print("임시 COCO 파일 정리 완료")
        except:
            pass
