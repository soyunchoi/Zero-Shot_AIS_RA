"""
D2SA 데이터셋을 사용한 VLM-SAM 통합 시스템 종합 실험 스크립트
- VLM occlusion 분석 결과를 JSON으로 저장
- 모든 샘플에 대한 시각화 결과 저장
- 정량적 성능 평가 및 통계 분석
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import traceback
from typing import Dict, List, Tuple, Optional

from vlm_sam_model import VLMSAMModel
from d2sa_dataset import D2SADataset


class D2SAExperiment:
    """D2SA 데이터셋을 사용한 VLM-SAM 실험 클래스"""
    
    def __init__(self, 
                 annotation_file: str,
                 image_dir: str,
                 output_dir: str = "./outputs/d2sa_experiment",
                 max_samples: Optional[int] = None):
        """
        실험 초기화
        
        Args:
            annotation_file: D2SA 어노테이션 파일 경로
            image_dir: 이미지 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            max_samples: 최대 처리할 샘플 수 (None이면 전체)
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.max_samples = max_samples
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 서브 디렉토리 생성
        self.viz_dir = os.path.join(output_dir, "visualizations")
        self.results_dir = os.path.join(output_dir, "results")
        self.analysis_dir = os.path.join(output_dir, "analysis")
        
        for dir_path in [self.viz_dir, self.results_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 실험 메타데이터
        self.experiment_info = {
            "start_time": datetime.now().isoformat(),
            "annotation_file": annotation_file,
            "image_dir": image_dir,
            "max_samples": max_samples,
            "output_dir": output_dir
        }
        
        print(f"=== D2SA 실험 초기화 완료 ===")
        print(f"📁 출력 디렉토리: {output_dir}")
        print(f"📊 최대 샘플 수: {max_samples or '전체'}")
    
    def load_dataset(self):
        """D2SA 데이터셋 로드"""
        print("\n📊 D2SA 데이터셋 로드 중...")
        
        try:
            # 데이터셋에 transform 추가하여 텐서로 변환
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
            
            self.dataset = D2SADataset(
                annotation_file=self.annotation_file,
                image_dir=self.image_dir,
                transform=transform,
                max_samples=self.max_samples
            )
            
            print(f"✅ 데이터셋 로드 성공: {len(self.dataset)}개 샘플")
            
            # 데이터셋 정보 저장
            dataset_info = {
                "total_samples": len(self.dataset),
                "annotation_file": self.annotation_file,
                "image_dir": self.image_dir
            }
            
            with open(os.path.join(self.results_dir, "dataset_info.json"), 'w') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            traceback.print_exc()
            return False
    
    def load_model(self):
        """VLM-SAM 모델 로드"""
        print("\n🤖 VLM-SAM 모델 로드 중...")
        
        try:
            self.model = VLMSAMModel()
            print("✅ 모델 로드 성공")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            traceback.print_exc()
            return False
    
    def get_category_name(self, category_id: int) -> str:
        """카테고리 ID를 카테고리 이름으로 변환"""
        # COCO 카테고리 매핑 (D2SA에서 사용하는 주요 카테고리들)
        category_map = {
            1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
            6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
            11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
            16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
            21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
            27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
            33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
            37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
            41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
            46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
            51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
            56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
            61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
            67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
            75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
            79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
            85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
            90: "toothbrush"
        }
        
        return category_map.get(category_id, f"category_{category_id}")
    
    def run_single_sample(self, idx: int) -> Dict:
        """단일 샘플에 대한 실험 수행"""
        try:
            # 데이터 로드 (D2SADataset은 튜플을 반환)
            data = self.dataset[idx]
            image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info = data
            
            # 배치 차원 추가 및 형변환
            image_tensor = image.unsqueeze(0)  # 배치 차원 추가
            box = bbox.unsqueeze(0)
            
            # PIL 이미지 생성 (원본 이미지에서)
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # 마스크를 numpy로 변환
            amodal_mask = amodal_mask.squeeze().cpu().numpy()
            visible_mask = visible_mask.squeeze().cpu().numpy()
            
            # 어노테이션 정보에서 추출
            image_id = annotation_info['image_id'].item()
            category_id = annotation_info['category_id'].item()
            
            # 카테고리 이름 변환
            target_class = self.get_category_name(category_id)
            
            print(f"  📸 이미지 ID: {image_id}")
            print(f"  🏷️ 카테고리: {target_class} (ID: {category_id})")
            print(f"  📐 이미지 크기: {image.shape}")
            print(f"  📦 바운딩 박스: {box[0].cpu().numpy()}")
            
            # VLM-SAM 파이프라인 실행
            print(f"  🔄 VLM-SAM 파이프라인 실행 중...")
            
            with torch.no_grad():
                results = self.model(
                    image=image_tensor,
                    box=box,
                    image_pil=pil_image,
                    text=target_class
                )
            
            # 결과 언패킹
            pred_amodal, pred_amodal_iou, pred_visible, pred_visible_iou, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = results
            
            # IoU 계산
            amodal_iou = self._calculate_iou(
                torch.sigmoid(pred_amodal[0]).squeeze().cpu().numpy(),
                amodal_mask
            )
            visible_iou = self._calculate_iou(
                torch.sigmoid(pred_visible[0]).squeeze().cpu().numpy(),
                visible_mask
            )
            
            print(f"  ✅ 파이프라인 실행 완료!")
            print(f"    - Attention layers: {len(attention_maps)}개")
            print(f"    - SAM points: {len(sam_points)}개")
            print(f"    - Amodal IoU: {amodal_iou:.3f}")
            print(f"    - Visible IoU: {visible_iou:.3f}")
            
            # 시각화 생성
            viz_path = os.path.join(self.viz_dir, f"sample_{idx:04d}_{image_id}_{target_class}.png")
            
            self.model.create_pipeline_visualization(
                image_pil=pil_image,
                results=results,
                bbox=box[0].cpu().numpy(),
                gt_amodal=amodal_mask,
                gt_visible=visible_mask,
                save_path=viz_path,
                title=f"Sample {idx} - {target_class} (ID: {image_id})"
            )
            
            # 결과 딕셔너리 생성
            sample_result = {
                "sample_idx": idx,
                "image_id": str(image_id),
                "category_id": int(category_id),
                "category_name": target_class,
                "image_shape": image.shape[2:],  # (H, W)
                "bbox": box[0].cpu().numpy().tolist(),
                
                # VLM 분석 결과
                "vlm_analysis": {
                    "visible_objects": occlusion_info.get("visible_objects", []),
                    "occluded_objects": occlusion_info.get("occluded_objects", []),
                    "occlusion_relations": occlusion_info.get("occlusion_relations", [])
                },
                
                # Attention 정보
                "attention_info": {
                    "num_layers": len(attention_maps),
                    "aggregated_stats": {
                        "mean": float(np.mean(aggregated_attention)),
                        "std": float(np.std(aggregated_attention)),
                        "min": float(np.min(aggregated_attention)),
                        "max": float(np.max(aggregated_attention))
                    }
                },
                
                # Point Sampling 결과
                "point_sampling": {
                    "total_points": len(sam_points),
                    "positive_points": int(np.sum(sam_labels)),
                    "negative_points": int(len(sam_points) - np.sum(sam_labels)),
                    "points": sam_points.tolist() if len(sam_points) > 0 else [],
                    "labels": sam_labels.tolist() if len(sam_labels) > 0 else []
                },
                
                # 성능 메트릭
                "performance": {
                    "amodal_iou": float(amodal_iou),
                    "visible_iou": float(visible_iou),
                    "pred_amodal_iou": float(pred_amodal_iou[0].item()),
                    "pred_visible_iou": float(pred_visible_iou[0].item())
                },
                
                # 파일 경로
                "visualization_path": viz_path,
                
                # 처리 상태
                "status": "success",
                "error": None
            }
            
            return sample_result
            
        except Exception as e:
            print(f"  ❌ 샘플 {idx} 처리 실패: {e}")
            traceback.print_exc()
            
            # 오류 결과 반환
            image_id = 'unknown'
            try:
                if 'data' in locals() and len(data) >= 7:
                    annotation_info = data[6]
                    if isinstance(annotation_info, dict) and 'image_id' in annotation_info:
                        image_id = str(annotation_info['image_id'].item())
            except:
                pass
            
            error_result = {
                "sample_idx": idx,
                "image_id": image_id,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            return error_result
    
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
    
    def run_experiment(self):
        """전체 실험 실행"""
        print(f"\n🚀 D2SA 실험 시작")
        print(f"=" * 60)
        
        # 데이터셋 및 모델 로드
        if not self.load_dataset():
            return False
        
        if not self.load_model():
            return False
        
        # 전체 결과 저장용 리스트
        all_results = []
        successful_samples = 0
        failed_samples = 0
        
        # 진행 상황 표시를 위한 tqdm
        total_samples = len(self.dataset)
        
        print(f"\n📊 {total_samples}개 샘플 처리 시작...")
        
        for idx in tqdm(range(total_samples), desc="Processing samples"):
            print(f"\n--- 샘플 {idx+1}/{total_samples} ---")
            
            # 단일 샘플 처리
            result = self.run_single_sample(idx)
            all_results.append(result)
            
            if result["status"] == "success":
                successful_samples += 1
            else:
                failed_samples += 1
            
            # 중간 결과 저장 (10개마다)
            if (idx + 1) % 10 == 0:
                temp_results_path = os.path.join(self.results_dir, f"temp_results_{idx+1}.json")
                with open(temp_results_path, 'w') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 모든 샘플 처리 완료!")
        print(f"  - 성공: {successful_samples}개")
        print(f"  - 실패: {failed_samples}개")
        print(f"  - 성공률: {successful_samples/total_samples*100:.1f}%")
        
        # 최종 결과 저장
        self.save_results(all_results)
        
        # 통계 분석 수행
        self.analyze_results(all_results)
        
        return True
    
    def save_results(self, all_results: List[Dict]):
        """결과를 JSON 파일로 저장"""
        print(f"\n💾 결과 저장 중...")
        
        # 실험 메타데이터 업데이트
        self.experiment_info.update({
            "end_time": datetime.now().isoformat(),
            "total_samples": len(all_results),
            "successful_samples": sum(1 for r in all_results if r["status"] == "success"),
            "failed_samples": sum(1 for r in all_results if r["status"] == "error")
        })
        
        # 전체 결과 저장
        final_results = {
            "experiment_info": self.experiment_info,
            "results": all_results
        }
        
        results_path = os.path.join(self.results_dir, "d2sa_experiment_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 전체 결과 저장: {results_path}")
        
        # 성공한 샘플만 따로 저장
        successful_results = [r for r in all_results if r["status"] == "success"]
        if successful_results:
            success_path = os.path.join(self.results_dir, "successful_results.json")
            with open(success_path, 'w') as f:
                json.dump(successful_results, f, indent=2, ensure_ascii=False)
            print(f"✅ 성공 결과 저장: {success_path}")
        
        # 실패한 샘플 로그 저장
        failed_results = [r for r in all_results if r["status"] == "error"]
        if failed_results:
            error_path = os.path.join(self.results_dir, "error_log.json")
            with open(error_path, 'w') as f:
                json.dump(failed_results, f, indent=2, ensure_ascii=False)
            print(f"⚠️ 오류 로그 저장: {error_path}")
    
    def analyze_results(self, all_results: List[Dict]):
        """결과 통계 분석"""
        print(f"\n📈 결과 분석 중...")
        
        successful_results = [r for r in all_results if r["status"] == "success"]
        
        if not successful_results:
            print("⚠️ 성공한 샘플이 없어 분석을 건너뜁니다.")
            return
        
        # 성능 통계
        amodal_ious = [r["performance"]["amodal_iou"] for r in successful_results]
        visible_ious = [r["performance"]["visible_iou"] for r in successful_results]
        
        # VLM 분석 통계
        vlm_stats = {
            "total_visible_objects": 0,
            "total_occluded_objects": 0,
            "total_occlusion_relations": 0,
            "visible_object_types": {},
            "occluded_object_types": {},
            "category_performance": {}
        }
        
        for result in successful_results:
            vlm = result["vlm_analysis"]
            
            # 객체 수 집계
            vlm_stats["total_visible_objects"] += len(vlm.get("visible_objects", []))
            vlm_stats["total_occluded_objects"] += len(vlm.get("occluded_objects", []))
            vlm_stats["total_occlusion_relations"] += len(vlm.get("occlusion_relations", []))
            
            # 객체 타입별 집계
            for obj in vlm.get("visible_objects", []):
                vlm_stats["visible_object_types"][obj] = vlm_stats["visible_object_types"].get(obj, 0) + 1
            
            for obj in vlm.get("occluded_objects", []):
                vlm_stats["occluded_object_types"][obj] = vlm_stats["occluded_object_types"].get(obj, 0) + 1
            
            # 카테고리별 성능
            category = result["category_name"]
            if category not in vlm_stats["category_performance"]:
                vlm_stats["category_performance"][category] = {
                    "count": 0,
                    "amodal_iou_sum": 0,
                    "visible_iou_sum": 0
                }
            
            vlm_stats["category_performance"][category]["count"] += 1
            vlm_stats["category_performance"][category]["amodal_iou_sum"] += result["performance"]["amodal_iou"]
            vlm_stats["category_performance"][category]["visible_iou_sum"] += result["performance"]["visible_iou"]
        
        # 카테고리별 평균 계산
        for category, stats in vlm_stats["category_performance"].items():
            if stats["count"] > 0:
                stats["avg_amodal_iou"] = stats["amodal_iou_sum"] / stats["count"]
                stats["avg_visible_iou"] = stats["visible_iou_sum"] / stats["count"]
        
        # 전체 통계
        analysis_results = {
            "performance_statistics": {
                "amodal_iou": {
                    "mean": float(np.mean(amodal_ious)),
                    "std": float(np.std(amodal_ious)),
                    "min": float(np.min(amodal_ious)),
                    "max": float(np.max(amodal_ious)),
                    "median": float(np.median(amodal_ious))
                },
                "visible_iou": {
                    "mean": float(np.mean(visible_ious)),
                    "std": float(np.std(visible_ious)),
                    "min": float(np.min(visible_ious)),
                    "max": float(np.max(visible_ious)),
                    "median": float(np.median(visible_ious))
                }
            },
            "vlm_analysis_statistics": vlm_stats,
            "point_sampling_statistics": {
                "avg_total_points": float(np.mean([r["point_sampling"]["total_points"] for r in successful_results])),
                "avg_positive_points": float(np.mean([r["point_sampling"]["positive_points"] for r in successful_results])),
                "avg_negative_points": float(np.mean([r["point_sampling"]["negative_points"] for r in successful_results]))
            }
        }
        
        # 분석 결과 저장
        analysis_path = os.path.join(self.analysis_dir, "experiment_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 분석 결과 저장: {analysis_path}")
        
        # 콘솔에 주요 통계 출력
        print(f"\n📊 주요 통계:")
        print(f"  🎯 평균 Amodal IoU: {analysis_results['performance_statistics']['amodal_iou']['mean']:.3f}")
        print(f"  🎯 평균 Visible IoU: {analysis_results['performance_statistics']['visible_iou']['mean']:.3f}")
        print(f"  👁️ 평균 발견된 보이는 객체: {vlm_stats['total_visible_objects']/len(successful_results):.1f}개")
        print(f"  🫥 평균 발견된 가려진 객체: {vlm_stats['total_occluded_objects']/len(successful_results):.1f}개")
        print(f"  🎯 평균 SAM Points: {analysis_results['point_sampling_statistics']['avg_total_points']:.1f}개")
    
    def create_summary_report(self):
        """실험 요약 보고서 생성"""
        print(f"\n📋 요약 보고서 생성 중...")
        
        # 생성된 파일들 정보 수집
        viz_files = [f for f in os.listdir(self.viz_dir) if f.endswith('.png')]
        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        report_content = f"""# D2SA 실험 결과 보고서

## 실험 정보
- 시작 시간: {self.experiment_info['start_time']}
- 어노테이션 파일: {self.annotation_file}
- 이미지 디렉토리: {self.image_dir}
- 최대 샘플 수: {self.max_samples or '전체'}

## 생성된 결과물
- 시각화 파일: {len(viz_files)}개
- 결과 JSON 파일: {len(result_files)}개
- 총 출력 크기: {self._get_directory_size(self.output_dir):.1f} MB

## 디렉토리 구조
```
{self.output_dir}/
├── visualizations/     # 모든 샘플의 파이프라인 시각화
├── results/           # JSON 결과 파일들
└── analysis/          # 통계 분석 결과
```

## 주요 파일
- `results/d2sa_experiment_results.json`: 전체 실험 결과
- `results/successful_results.json`: 성공한 샘플들만
- `analysis/experiment_analysis.json`: 통계 분석 결과
- `visualizations/`: 각 샘플별 8-panel 파이프라인 시각화

## 사용 방법
실험 결과를 분석하려면:
```python
import json

# 전체 결과 로드
with open('{os.path.join(self.results_dir, "d2sa_experiment_results.json")}', 'r') as f:
    results = json.load(f)

# 분석 결과 로드  
with open('{os.path.join(self.analysis_dir, "experiment_analysis.json")}', 'r') as f:
    analysis = json.load(f)
```
"""
        
        report_path = os.path.join(self.output_dir, "README.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"✅ 요약 보고서 저장: {report_path}")
    
    def _get_directory_size(self, directory: str) -> float:
        """디렉토리 크기 계산 (MB 단위)"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # MB로 변환


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🚀 D2SA VLM-SAM 통합 실험 시작")
    print("=" * 80)
    
    # 실험 설정 (train.py와 동일한 경로 사용)
    D2SA_ROOT = "/root/datasets/D2SA"
    annotation_file = os.path.join(D2SA_ROOT, "D2S_amodal_augmented.json")
    image_dir = os.path.join(D2SA_ROOT, "images")
    output_dir = f"./outputs/d2sa_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    max_samples = 50  # 메모리 문제 방지를 위해 50개로 제한 (None으로 설정하면 전체)
    
    # 실험 객체 생성
    experiment = D2SAExperiment(
        annotation_file=annotation_file,
        image_dir=image_dir,
        output_dir=output_dir,
        max_samples=max_samples
    )
    
    try:
        # 실험 실행
        success = experiment.run_experiment()
        
        if success:
            # 요약 보고서 생성
            experiment.create_summary_report()
            
            print(f"\n🎉 D2SA 실험 완료!")
            print(f"📁 결과 저장 위치: {output_dir}")
            print(f"📊 자세한 내용은 {os.path.join(output_dir, 'README.md')}를 확인하세요.")
        else:
            print(f"\n❌ 실험 실행 실패")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 실험이 중단되었습니다.")
        print(f"📁 지금까지의 결과는 {output_dir}에 저장되어 있습니다.")
    
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
