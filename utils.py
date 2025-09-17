import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple
import datetime

class CheckpointManager:
    """
    모델 체크포인트 관리 클래스
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.best_metric = 0.0
        self.best_epoch = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 체크포인트 기록 파일
        self.log_file = os.path.join(checkpoint_dir, "checkpoint_log.json")
        self.load_checkpoint_log()
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False, model_name: str = "model"):
        """
        체크포인트를 저장합니다.
        
        Args:
            model: 저장할 모델
            optimizer: 옵티마이저
            epoch: 에포크 번호
            metrics: 평가 메트릭
            is_best: 최고 성능 모델 여부
            model_name: 모델 이름
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{model_name}_epoch_{epoch:03d}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
            torch.save(checkpoint, best_path)
            self.best_metric = metrics.get('mIoU', 0.0)
            self.best_epoch = epoch
            print(f"새로운 최고 성능 모델 저장: {best_path} (mIoU: {self.best_metric:.4f})")
        
        # 최신 체크포인트 저장
        latest_path = os.path.join(self.checkpoint_dir, f"{model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # 체크포인트 로그 업데이트
        self.update_checkpoint_log(epoch, metrics, checkpoint_path, is_best)
        
        # 오래된 체크포인트 정리
        self.cleanup_old_checkpoints(model_name)
        
        print(f"체크포인트 저장됨: {checkpoint_path}")
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       checkpoint_path: str) -> int:
        """
        체크포인트를 로드합니다.
        
        Args:
            model: 모델
            optimizer: 옵티마이저
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            int: 로드된 에포크 번호
        """
        if not os.path.exists(checkpoint_path):
            print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        metrics = checkpoint.get('metrics', {})
        
        print(f"체크포인트 로드됨: {checkpoint_path}")
        print(f"에포크: {epoch}, 메트릭: {metrics}")
        
        return epoch
    
    def get_best_checkpoint_path(self, model_name: str) -> Optional[str]:
        """최고 성능 체크포인트 경로 반환"""
        best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
        return best_path if os.path.exists(best_path) else None
    
    def get_latest_checkpoint_path(self, model_name: str) -> Optional[str]:
        """최신 체크포인트 경로 반환"""
        latest_path = os.path.join(self.checkpoint_dir, f"{model_name}_latest.pth")
        return latest_path if os.path.exists(latest_path) else None
    
    def cleanup_old_checkpoints(self, model_name: str):
        """오래된 체크포인트 파일들을 정리합니다."""
        # 모든 체크포인트 파일 찾기
        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(f"{model_name}_epoch_") and filename.endswith('.pth'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                epoch_num = int(filename.split('_epoch_')[1].split('.')[0])
                checkpoint_files.append((epoch_num, filepath))
        
        # 에포크 번호로 정렬
        checkpoint_files.sort(key=lambda x: x[0])
        
        # 최신 체크포인트들만 유지
        if len(checkpoint_files) > self.max_checkpoints:
            for _, filepath in checkpoint_files[:-self.max_checkpoints]:
                os.remove(filepath)
                print(f"오래된 체크포인트 삭제: {filepath}")
    
    def load_checkpoint_log(self):
        """체크포인트 로그 파일을 로드합니다."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
                    self.best_metric = log_data.get('best_metric', 0.0)
                    self.best_epoch = log_data.get('best_epoch', 0)
                print(f"기존 체크포인트 로그 로드됨: best_metric={self.best_metric:.4f}, best_epoch={self.best_epoch}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"체크포인트 로그 파일이 손상되었습니다: {e}")
                print("새로운 로그 파일로 초기화합니다.")
                # 손상된 파일을 백업하고 새로 시작
                backup_file = self.log_file + ".backup"
                if os.path.exists(self.log_file):
                    os.rename(self.log_file, backup_file)
                    print(f"손상된 파일을 백업했습니다: {backup_file}")
                # 초기값으로 설정
                self.best_metric = 0.0
                self.best_epoch = 0
    
    def update_checkpoint_log(self, epoch: int, metrics: Dict[str, float], 
                             checkpoint_path: str, is_best: bool):
        """체크포인트 로그를 업데이트합니다."""
        # NumPy 타입을 Python 기본 타입으로 변환하는 함수
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        log_data = {
            'best_metric': float(self.best_metric),
            'best_epoch': int(self.best_epoch),
            'last_epoch': int(epoch),
            'last_metrics': convert_numpy_types(metrics),
            'last_checkpoint': checkpoint_path,
            'last_update': datetime.datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

class VisualizationUtils:
    """
    시각화 유틸리티 클래스
    """
    
    @staticmethod
    def save_prediction_visualization(image: np.ndarray, 
                                    gt_amodal_mask: np.ndarray,
                                    gt_visible_mask: np.ndarray,
                                    pred_amodal_mask: np.ndarray,
                                    pred_visible_mask: np.ndarray,
                                    save_path: str,
                                    bbox: Optional[np.ndarray] = None,
                                    text: str = ""):
        """
        예측 결과를 시각화하여 저장합니다.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 원본 이미지
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        if bbox is not None:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                               fill=False, color='red', linewidth=2)
            axes[0, 0].add_patch(rect)
        if text:
            axes[0, 0].text(10, 30, text, color='white', fontsize=12, 
                           bbox=dict(facecolor='black', alpha=0.7))
        
        # GT Amodal & Visible
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(gt_amodal_mask, alpha=0.5, cmap='Reds')
        axes[0, 1].imshow(gt_visible_mask, alpha=0.5, cmap='Blues')
        axes[0, 1].set_title('GT: Amodal (Red), Visible (Blue)')
        
        # Predicted Amodal & Visible
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(pred_amodal_mask, alpha=0.5, cmap='Reds')
        axes[0, 2].imshow(pred_visible_mask, alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('Pred: Amodal (Red), Visible (Blue)')

        # Amodal Comparison
        axes[1, 0].imshow(gt_amodal_mask, cmap='Reds', alpha=0.6)
        axes[1, 0].imshow(pred_amodal_mask, cmap='Greens', alpha=0.6)
        axes[1, 0].set_title('Amodal: GT (Red) vs Pred (Green)')

        # Visible Comparison
        axes[1, 1].imshow(gt_visible_mask, cmap='Reds', alpha=0.6)
        axes[1, 1].imshow(pred_visible_mask, cmap='Greens', alpha=0.6)
        axes[1, 1].set_title('Visible: GT (Red) vs Pred (Green)')

        # Invisible Region Reconstruction
        gt_invisible = np.clip(gt_amodal_mask - gt_visible_mask, 0, 1)
        pred_invisible = np.clip(pred_amodal_mask - gt_visible_mask, 0, 1)
        axes[1, 2].imshow(image)
        axes[1, 2].imshow(gt_invisible, alpha=0.6, cmap='Reds')
        axes[1, 2].imshow(pred_invisible, alpha=0.4, cmap='Greens')
        axes[1, 2].set_title('Invisible Recon: GT (Red) vs Pred (Green)')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_training_curves(train_losses: List[float], 
                           val_metrics: List[Dict[str, float]],
                           save_path: str):
        """
        학습 곡선을 시각화하여 저장합니다.
        
        Args:
            train_losses: 학습 손실 리스트
            val_metrics: 검증 메트릭 리스트
            save_path: 저장 경로
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training Loss
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # mIoU
        if val_metrics:
            miou_values = [m.get('mIoU', 0) for m in val_metrics]
            axes[0, 1].plot(epochs[:len(miou_values)], miou_values, 'r-', label='mIoU')
            axes[0, 1].set_title('mIoU')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mIoU')
            axes[0, 1].grid(True)
        
        # AP@0.5
        if val_metrics:
            ap50_values = [m.get('AP50', 0) for m in val_metrics]
            axes[1, 0].plot(epochs[:len(ap50_values)], ap50_values, 'g-', label='AP@0.5')
            axes[1, 0].set_title('AP@0.5')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AP@0.5')
            axes[1, 0].grid(True)
        
        # AP@0.75
        if val_metrics:
            ap75_values = [m.get('AP75', 0) for m in val_metrics]
            axes[1, 1].plot(epochs[:len(ap75_values)], ap75_values, 'm-', label='AP@0.75')
            axes[1, 1].set_title('AP@0.75')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AP@0.75')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def denormalize_image(tensor_image: torch.Tensor) -> np.ndarray:
    """
    정규화된 텐서 이미지를 원본 이미지로 변환합니다.
    
    Args:
        tensor_image: 정규화된 텐서 이미지 (3, H, W)
        
    Returns:
        np.ndarray: 원본 이미지 (H, W, 3)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 텐서를 numpy로 변환
    if isinstance(tensor_image, torch.Tensor):
        image = tensor_image.cpu().numpy()
    else:
        image = tensor_image
    
    # (3, H, W) -> (H, W, 3)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 정규화 해제
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return (image * 255).astype(np.uint8)

def create_output_directories(base_dir: str, model_name: str, epoch: int) -> Dict[str, str]:
    """
    출력 디렉토리들을 생성합니다.
    
    Args:
        base_dir: 기본 디렉토리
        model_name: 모델 이름
        epoch: 에포크 번호
        
    Returns:
        Dict: 생성된 디렉토리 경로들
    """
    dirs = {
        'masks': os.path.join(base_dir, model_name, f"epoch_{epoch}", "masks"),
        'visualizations': os.path.join(base_dir, model_name, f"epoch_{epoch}", "visualizations"),
        'checkpoints': os.path.join(base_dir, model_name, "checkpoints"),
        'logs': os.path.join(base_dir, model_name, "logs")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_metrics_to_file(metrics: Dict[str, float], filepath: str):
    """메트릭을 JSON 파일로 저장합니다."""
    # NumPy 타입을 Python 기본 타입으로 변환
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 메트릭 변환
    converted_metrics = convert_numpy_types(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(converted_metrics, f, indent=2)

def load_metrics_from_file(filepath: str) -> Dict[str, float]:
    """JSON 파일에서 메트릭을 로드합니다."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

# 사용 예시
if __name__ == '__main__':
    # 체크포인트 매니저 테스트
    checkpoint_manager = CheckpointManager('./test_checkpoints')
    
    # 더미 모델과 옵티마이저
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 더미 메트릭
    metrics = {'mIoU': 0.75, 'AP50': 0.65}
    
    # 체크포인트 저장
    checkpoint_manager.save_checkpoint(model, optimizer, 1, metrics, is_best=True)
    
    print("체크포인트 매니저 테스트 완료!") 