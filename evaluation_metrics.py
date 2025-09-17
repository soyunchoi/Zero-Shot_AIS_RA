import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
# sklearn.metrics import removed - using COCO API instead
import cv2
from typing import List, Dict, Tuple

class AmodalEvaluationMetrics:
    """
    Amodal 및 Visible Instance Segmentation을 위한 평가 메트릭 클래스
    """
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.reset()
    
    def reset(self):
        """메트릭 상태 초기화"""
        self.iou_scores = []
        self.predictions = []
        self.ground_truths = []
        self.coco_results = []
        
    def compute_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        두 마스크 간의 IoU를 계산합니다.
        """
        pred_mask = (pred_mask > 0.5).astype(np.bool_)
        gt_mask = (gt_mask > 0.5).astype(np.bool_)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def add_batch(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                  annotations: List[Dict], confidence_scores: List[float]):
        """
        배치 결과를 메트릭에 추가합니다.
        """
        for pred, gt, ann, conf in zip(
            pred_masks, gt_masks, annotations, confidence_scores):
            
            iou = self.compute_iou(pred, gt)
            self.iou_scores.append(iou)
            
            pred_binary = (pred > 0.5).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(pred_binary))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            coco_result = {
                "image_id": ann['image_id'],
                "category_id": ann['category_id'],
                "segmentation": rle,
                "score": conf,
            }
            self.coco_results.append(coco_result)
            
            self.predictions.append({'mask': pred, 'score': conf, 'annotation': ann})
            self.ground_truths.append({'mask': gt, 'annotation': ann})
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        COCO GT 없이 기본 메트릭만 계산합니다.
        """
        miou = np.mean(self.iou_scores) if self.iou_scores else 0.0
        std_iou = np.std(self.iou_scores) if self.iou_scores else 0.0
        
        # IoU 기반 메트릭들
        iou_50 = np.mean([iou for iou in self.iou_scores if iou >= 0.5]) if self.iou_scores else 0.0
        iou_75 = np.mean([iou for iou in self.iou_scores if iou >= 0.75]) if self.iou_scores else 0.0
        
        return {
            f'{self.prefix}mIoU': miou,
            f'{self.prefix}stdIoU': std_iou,
            f'{self.prefix}IoU@50': iou_50,
            f'{self.prefix}IoU@75': iou_75,
            f'{self.prefix}num_samples': len(self.iou_scores)
        }

    def compute_all_metrics(self, coco_gt: COCO) -> Dict[str, float]:
        """
        모든 메트릭을 계산하여 반환합니다.
        """
        miou = np.mean(self.iou_scores) if self.iou_scores else 0.0
        
        # COCO 메트릭만 사용 (참조 표준)
        coco_metrics = self.compute_coco_metrics(coco_gt)
        
        all_metrics = {
            f'{self.prefix}mIoU': miou,
            **{f'{self.prefix}{k}': v for k, v in coco_metrics.items()}
        }
        
        return all_metrics

    def print_metrics(self, coco_gt: COCO, epoch: int = 0):
        """메트릭 결과를 출력합니다."""
        metrics = self.compute_all_metrics(coco_gt)
        
        print(f"\n--- 에포크 {epoch} {self.prefix.capitalize()} 평가 결과 ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 30)
        
        return metrics

    def compute_coco_metrics(self, coco_gt: COCO) -> Dict[str, float]:
        """
        COCO 메트릭 계산
        """
        if not self.coco_results:
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APs': 0.0, 'APm': 0.0, 'APl': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARs': 0.0, 'ARm': 0.0, 'ARl': 0.0
            }
        
        try:
            coco_dt = coco_gt.loadRes(self.coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 결과 추출
            stats = coco_eval.stats
            return {
                'AP': stats[0],      # AP @ IoU=0.50:0.95
                'AP50': stats[1],    # AP @ IoU=0.50
                'AP75': stats[2],    # AP @ IoU=0.75
                'APs': stats[3],     # AP @ IoU=0.50:0.95 (small)
                'APm': stats[4],     # AP @ IoU=0.50:0.95 (medium)
                'APl': stats[5],     # AP @ IoU=0.50:0.95 (large)
                'AR1': stats[6],     # AR @ IoU=0.50:0.95 (max 1 det)
                'AR10': stats[7],    # AR @ IoU=0.50:0.95 (max 10 det)
                'AR100': stats[8],   # AR @ IoU=0.50:0.95 (max 100 det)
                'ARs': stats[9],     # AR @ IoU=0.50:0.95 (small)
                'ARm': stats[10],    # AR @ IoU=0.50:0.95 (medium)
                'ARl': stats[11]     # AR @ IoU=0.50:0.95 (large)
            }
        except Exception as e:
            print(f"COCO 메트릭 계산 중 오류: {e}")
            return {
                'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'APs': 0.0, 'APm': 0.0, 'APl': 0.0,
                'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
                'ARs': 0.0, 'ARm': 0.0, 'ARl': 0.0
            }

def compute_occlusion_metrics(pred_amodal_masks: List[np.ndarray],
                             pred_visible_masks: List[np.ndarray], 
                             gt_amodal_masks: List[np.ndarray],
                             gt_visible_masks: List[np.ndarray]) -> Dict[str, float]:
    """
    Amodal Instance Segmentation에 특화된 메트릭을 계산합니다.
    
    Args:
        pred_amodal_masks: 예측 amodal 마스크 리스트
        pred_visible_masks: 예측 visible 마스크 리스트
        gt_amodal_masks: 실제 amodal 마스크 리스트
        gt_visible_masks: 실제 visible 마스크 리스트
        
    Returns:
        Dict: Amodal segmentation 관련 메트릭
    """
    invisible_reconstruction_ious = []
    occlusion_completion_rates = []
    consistency_scores = []
    
    for pred_amodal, pred_visible, gt_amodal, gt_visible in zip(
        pred_amodal_masks, pred_visible_masks, gt_amodal_masks, gt_visible_masks):
        
        # Invisible 영역 재구성 성능 (Amodal - Visible)
        gt_invisible = gt_amodal - gt_visible
        pred_invisible = pred_amodal - pred_visible
        
        if gt_invisible.sum() > 0:  # invisible 영역이 있는 경우만
            pred_invisible = np.clip(pred_invisible, 0, 1)
            intersection = np.logical_and(pred_invisible > 0.5, gt_invisible > 0.5).sum()
            union = np.logical_or(pred_invisible > 0.5, gt_invisible > 0.5).sum()
            
            if union > 0:
                invisible_iou = intersection / union
                invisible_reconstruction_ious.append(invisible_iou)
        
        # Amodal completion rate
        if gt_amodal.sum() > 0:
            completion_rate = pred_amodal.sum() / gt_amodal.sum()
            occlusion_completion_rates.append(completion_rate)
        
        # Visible-Amodal consistency (비지블 마스크가 아모달 내에 포함되는지)
        if pred_amodal.sum() > 0 and pred_visible.sum() > 0:
            overlap = np.logical_and(pred_amodal > 0.5, pred_visible > 0.5).sum()
            visible_total = (pred_visible > 0.5).sum()
            consistency = overlap / visible_total if visible_total > 0 else 0.0
            consistency_scores.append(consistency)
    
    return {
        'invisible_reconstruction_mIoU': np.mean(invisible_reconstruction_ious) if invisible_reconstruction_ious else 0.0,
        'amodal_completion_rate': np.mean(occlusion_completion_rates) if occlusion_completion_rates else 0.0,
        'visible_amodal_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
        'num_occluded_objects': len(invisible_reconstruction_ious)
    } 