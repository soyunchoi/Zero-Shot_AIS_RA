"""
VLM-SAM 통합 시스템의 전체 파이프라인을 시각화하는 모듈
Attention Map → Point Sampling → SAM Prediction 전체 과정을 시각화합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Dict, Tuple, Optional


class IntegratedVisualizer:
    """VLM-SAM 통합 시스템의 전체 파이프라인 시각화 클래스"""
    
    def __init__(self):
        """IntegratedVisualizer 초기화"""
        print("IntegratedVisualizer 초기화 완료")
    
    def visualize_complete_pipeline(self,
                                  image: np.ndarray,
                                  attention_maps: Dict,
                                  aggregated_attention: np.ndarray,
                                  sam_points: np.ndarray,
                                  sam_labels: np.ndarray,
                                  pred_amodal_mask: np.ndarray,
                                  pred_visible_mask: np.ndarray,
                                  gt_amodal_mask: Optional[np.ndarray] = None,
                                  gt_visible_mask: Optional[np.ndarray] = None,
                                  bbox: Optional[np.ndarray] = None,
                                  save_path: str = None,
                                  title: str = "VLM-SAM Pipeline") -> None:
        """
        VLM-SAM 전체 파이프라인을 시각화합니다.
        
        Args:
            image: 원본 이미지 (H, W, 3)
            attention_maps: Layer별 attention maps
            aggregated_attention: 집계된 attention map (H, W)
            sam_points: SAM prompt points (N, 2)
            sam_labels: SAM prompt labels (N,)
            pred_amodal_mask: 예측된 amodal mask (H, W)
            pred_visible_mask: 예측된 visible mask (H, W)
            gt_amodal_mask: GT amodal mask (optional)
            gt_visible_mask: GT visible mask (optional)
            bbox: 바운딩 박스 [x1, y1, x2, y2] (optional)
            save_path: 저장 경로
            title: 제목
        """
        
        print(f"🎨 통합 파이프라인 시각화 생성: {title}")
        
        # 서브플롯 구성 (2행 4열)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 1. 원본 이미지 + 바운딩 박스
        axes[0, 0].imshow(image)
        if bbox is not None:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                               fill=False, color='red', linewidth=2)
            axes[0, 0].add_patch(rect)
        axes[0, 0].set_title('1. Original Image + BBox')
        axes[0, 0].axis('off')
        
        # 2. Aggregated Attention Map
        im2 = axes[0, 1].imshow(aggregated_attention, cmap='jet')
        axes[0, 1].set_title('2. VLM Attention Map')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # 3. Attention Map + Sampled Points
        axes[0, 2].imshow(image, alpha=0.7)
        axes[0, 2].imshow(aggregated_attention, cmap='jet', alpha=0.5)
        
        # SAM points 시각화
        if len(sam_points) > 0:
            positive_points = sam_points[sam_labels == 1]
            negative_points = sam_points[sam_labels == 0]
            
            if len(positive_points) > 0:
                axes[0, 2].scatter(positive_points[:, 0], positive_points[:, 1], 
                                 c='red', s=100, marker='o', edgecolors='white', linewidth=2,
                                 label=f'Positive ({len(positive_points)})')
            
            if len(negative_points) > 0:
                axes[0, 2].scatter(negative_points[:, 0], negative_points[:, 1], 
                                 c='blue', s=100, marker='x', linewidth=3,
                                 label=f'Negative ({len(negative_points)})')
            
            axes[0, 2].legend()
        
        axes[0, 2].set_title('3. Attention + SAM Points')
        axes[0, 2].axis('off')
        
        # 4. 예측된 Amodal Mask
        axes[0, 3].imshow(image, alpha=0.7)
        im4 = axes[0, 3].imshow(pred_amodal_mask, alpha=0.6, cmap='Reds')
        axes[0, 3].set_title('4. Predicted Amodal Mask')
        axes[0, 3].axis('off')
        
        # 5. 예측된 Visible Mask
        axes[1, 0].imshow(image, alpha=0.7)
        im5 = axes[1, 0].imshow(pred_visible_mask, alpha=0.6, cmap='Blues')
        axes[1, 0].set_title('5. Predicted Visible Mask')
        axes[1, 0].axis('off')
        
        # 6. Amodal vs Visible 비교
        axes[1, 1].imshow(image, alpha=0.7)
        axes[1, 1].imshow(pred_amodal_mask, alpha=0.4, cmap='Reds', label='Amodal')
        axes[1, 1].imshow(pred_visible_mask, alpha=0.4, cmap='Blues', label='Visible')
        axes[1, 1].set_title('6. Amodal (Red) vs Visible (Blue)')
        axes[1, 1].axis('off')
        
        # 7. GT와 비교 (GT가 있는 경우)
        if gt_amodal_mask is not None and gt_visible_mask is not None:
            axes[1, 2].imshow(image, alpha=0.7)
            axes[1, 2].imshow(gt_amodal_mask, alpha=0.3, cmap='Greens', label='GT Amodal')
            axes[1, 2].imshow(pred_amodal_mask, alpha=0.3, cmap='Reds', label='Pred Amodal')
            axes[1, 2].set_title('7. GT (Green) vs Pred (Red)')
            axes[1, 2].axis('off')
        else:
            # GT가 없으면 Invisible 영역 시각화
            invisible_mask = np.clip(pred_amodal_mask - pred_visible_mask, 0, 1)
            axes[1, 2].imshow(image, alpha=0.7)
            axes[1, 2].imshow(invisible_mask, alpha=0.6, cmap='Purples')
            axes[1, 2].set_title('7. Predicted Invisible Region')
            axes[1, 2].axis('off')
        
        # 8. 통계 정보
        axes[1, 3].axis('off')
        
        # 통계 텍스트 생성
        stats_text = []
        stats_text.append(f"VLM-SAM Pipeline Statistics")
        stats_text.append(f"=" * 30)
        stats_text.append(f"Attention Layers: {len(attention_maps)}")
        stats_text.append(f"Attention Mean: {np.mean(aggregated_attention):.3f}")
        stats_text.append(f"Attention Std: {np.std(aggregated_attention):.3f}")
        stats_text.append(f"")
        stats_text.append(f"SAM Prompts:")
        stats_text.append(f"  Total Points: {len(sam_points)}")
        stats_text.append(f"  Positive: {np.sum(sam_labels)}")
        stats_text.append(f"  Negative: {len(sam_points) - np.sum(sam_labels)}")
        stats_text.append(f"")
        stats_text.append(f"Predictions:")
        stats_text.append(f"  Amodal Coverage: {np.mean(pred_amodal_mask):.3f}")
        stats_text.append(f"  Visible Coverage: {np.mean(pred_visible_mask):.3f}")
        
        if gt_amodal_mask is not None:
            # IoU 계산
            amodal_iou = self._calculate_iou(pred_amodal_mask, gt_amodal_mask)
            visible_iou = self._calculate_iou(pred_visible_mask, gt_visible_mask)
            stats_text.append(f"")
            stats_text.append(f"Performance:")
            stats_text.append(f"  Amodal IoU: {amodal_iou:.3f}")
            stats_text.append(f"  Visible IoU: {visible_iou:.3f}")
        
        # 텍스트 표시
        axes[1, 3].text(0.05, 0.95, '\n'.join(stats_text), 
                        transform=axes[1, 3].transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 3].set_title('8. Pipeline Statistics')
        
        # 전체 제목
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 통합 파이프라인 시각화 저장: {save_path}")
        
        plt.close()
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """IoU 계산"""
        # 이진화
        pred_binary = (pred_mask > 0.5).astype(float)
        gt_binary = (gt_mask > 0.5).astype(float)
        
        # IoU 계산
        intersection = np.sum(pred_binary * gt_binary)
        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def create_attention_analysis(self,
                                attention_maps: Dict,
                                aggregated_attention: np.ndarray,
                                save_path: str = None) -> None:
        """
        Attention Map 분석 시각화를 생성합니다.
        
        Args:
            attention_maps: Layer별 attention maps
            aggregated_attention: 집계된 attention map
            save_path: 저장 경로
        """
        
        print("🧠 Attention Map 분석 시각화 생성 중...")
        
        num_layers = len(attention_maps)
        if num_layers == 0:
            print("⚠️ Attention maps가 없습니다.")
            return
        
        # 서브플롯 구성
        cols = min(4, num_layers + 1)
        rows = (num_layers + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Layer별 attention maps
        for i, (layer_name, attention_map) in enumerate(attention_maps.items()):
            row = i // cols
            col = i % cols
            
            im = axes[row, col].imshow(attention_map, cmap='jet')
            axes[row, col].set_title(f'{layer_name}\nMean: {np.mean(attention_map):.3f}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        # 집계된 attention map
        final_row = (num_layers) // cols
        final_col = (num_layers) % cols
        
        im_final = axes[final_row, final_col].imshow(aggregated_attention, cmap='jet')
        axes[final_row, final_col].set_title(f'Aggregated\nMean: {np.mean(aggregated_attention):.3f}')
        axes[final_row, final_col].axis('off')
        plt.colorbar(im_final, ax=axes[final_row, final_col], fraction=0.046)
        
        # 빈 subplot 숨기기
        for i in range(num_layers + 1, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('VLM Attention Maps Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Attention 분석 시각화 저장: {save_path}")
        
        plt.close()


# 사용 예시
if __name__ == '__main__':
    print("=== IntegratedVisualizer 테스트 ===")
    
    # 더미 데이터 생성
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    attention_maps = {
        'layer_26': np.random.rand(224, 224),
        'layer_27': np.random.rand(224, 224),
        'layer_28': np.random.rand(224, 224),
    }
    aggregated_attention = np.mean(list(attention_maps.values()), axis=0)
    sam_points = np.array([[100, 100], [150, 150], [50, 180]])
    sam_labels = np.array([1, 1, 0])
    pred_amodal = np.random.rand(224, 224)
    pred_visible = np.random.rand(224, 224) * 0.8
    bbox = np.array([50, 50, 200, 200])
    
    # 시각화 생성
    visualizer = IntegratedVisualizer()
    
    # 완전한 파이프라인 시각화
    visualizer.visualize_complete_pipeline(
        image, attention_maps, aggregated_attention,
        sam_points, sam_labels, pred_amodal, pred_visible,
        bbox=bbox, save_path="./test_pipeline_visualization.png",
        title="Test VLM-SAM Pipeline"
    )
    
    # Attention 분석 시각화
    visualizer.create_attention_analysis(
        attention_maps, aggregated_attention,
        save_path="./test_attention_analysis.png"
    )
    
    print("✅ IntegratedVisualizer 테스트 완료!")
