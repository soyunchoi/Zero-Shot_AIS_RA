"""
Attention Map을 Point Sampling으로 변환하는 모듈
VLM에서 추출된 attention map을 분석하여 SAM prompt용 positive/negative point를 생성합니다.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
# K-means 대신 Grid-based sampling 사용 (메모리 효율적)


class AttentionPointSampler:
    """
    Attention Map에서 positive/negative point를 샘플링하는 클래스
    """
    
    def __init__(self):
        """
        AttentionPointSampler 초기화
        """
        print("AttentionPointSampler 초기화 완료")
    
    def sample_points_from_attention(self, 
                                   attention_map: np.ndarray,
                                   num_positive: int = 10,
                                   num_negative: int = 5,
                                   bbox: Optional[np.ndarray] = None,
                                   min_distance: int = 20,
                                   use_vlsam_method: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Attention map에서 positive/negative point를 샘플링합니다.
        
        Args:
            attention_map (np.ndarray): Attention map (H, W)
            num_positive (int): 추출할 positive point 수
            num_negative (int): 추출할 negative point 수  
            bbox (np.ndarray, optional): 바운딩 박스 [x1, y1, x2, y2]
            min_distance (int): 포인트 간 최소 거리
            use_vlsam_method (bool): VL-SAM 논문 방식 사용 여부
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (positive_points, negative_points, labels)
        """
        print(f"🎯 Point Sampling 시작 ({'VL-SAM 방식' if use_vlsam_method else '기본 방식'}): {num_positive} positive, {num_negative} negative")
        
        if use_vlsam_method:
            return self.vlsam_point_sampling(attention_map, bbox)
        else:
            return self.basic_point_sampling(attention_map, num_positive, num_negative, bbox, min_distance)
    
    def basic_point_sampling(self, attention_map: np.ndarray, num_positive: int, num_negative: int, 
                           bbox: Optional[np.ndarray], min_distance: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """기존 방식의 point sampling"""
        # Attention map 정규화
        attention_normalized = self._normalize_attention_map(attention_map)
        
        # 바운딩 박스 영역 추출 (있는 경우)
        if bbox is not None:
            bbox_mask = self._create_bbox_mask(attention_normalized.shape, bbox)
            print(f"  📦 바운딩 박스 적용: {bbox}")
        else:
            bbox_mask = np.ones_like(attention_normalized, dtype=bool)
        
        # Positive points 추출 (high attention 영역)
        positive_points = self._extract_positive_points(
            attention_normalized, 
            bbox_mask, 
            num_positive, 
            min_distance
        )
        
        # Negative points 추출 (low attention 영역)
        negative_points = self._extract_negative_points(
            attention_normalized, 
            bbox_mask, 
            num_negative, 
            min_distance,
            existing_points=positive_points
        )
        
        # 라벨 생성 (1: positive, 0: negative)
        labels = np.concatenate([
            np.ones(len(positive_points), dtype=int),
            np.zeros(len(negative_points), dtype=int)
        ])
        
        print(f"  ✓ 기본 방식 추출 완료: {len(positive_points)} positive, {len(negative_points)} negative points")
        
        return positive_points, negative_points, labels
    
    def vlsam_point_sampling(self, attention_map: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        VL-SAM 논문 방식의 Point Sampling (Bounding Box 활용 강화)
        1. Bounding Box 영역 내에서 attention 집중
        2. Threshold 필터링으로 약한 활성화 영역 제거
        3. Connected Component Analysis로 최대 연결 영역을 positive area로 선택
        4. Positive area에서 최대값, Negative area에서 최소값 포인트 샘플링
        """
        print("  🧠 VL-SAM 방식 적용: BBox Focus + Threshold + Connected Component + Max/Min Sampling")
        
        # 1. Attention map 정규화
        attention_normalized = self._normalize_attention_map(attention_map)
        
        # 2. Bounding Box 마스크 적용 (있는 경우)
        if bbox is not None:
            bbox_mask = self._create_bbox_mask(attention_normalized.shape, bbox)
            print(f"    📦 Bounding Box 적용: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            
            # Attention을 bbox 영역에 집중시키기
            attention_focused = attention_normalized.copy()
            attention_focused[~bbox_mask] *= 0.1  # bbox 외부 영역은 10%로 감소
            attention_normalized = attention_focused
        else:
            print("    ⚠️ Bounding Box 없음 - 전체 이미지 대상")
        
        # 3. Threshold 필터링 (논문 3.3절)
        positive_area, negative_area = self._vlsam_threshold_filtering(attention_normalized)
        
        # 4. Connected Component Analysis로 최대 연결 영역 찾기
        main_positive_area = self._find_maximum_connected_component(positive_area)
        
        # 5. VL-SAM 방식으로 포인트 샘플링
        positive_points, negative_points = self._vlsam_sample_points(
            attention_normalized, main_positive_area, negative_area, bbox
        )
        
        # 5. 라벨 생성
        labels = np.concatenate([
            np.ones(len(positive_points), dtype=int),
            np.zeros(len(negative_points), dtype=int)
        ])
        
        print(f"  ✓ VL-SAM 방식 완료: {len(positive_points)} positive, {len(negative_points)} negative points")
        
        return positive_points, negative_points, labels
    
    def _vlsam_threshold_filtering(self, attention_map: np.ndarray, threshold_percentile: float = 70) -> Tuple[np.ndarray, np.ndarray]:
        """
        VL-SAM 논문의 Threshold 필터링
        약한 활성화 영역을 필터링하여 positive/negative 영역 분리
        """
        # Threshold 계산 (상위 30% 영역을 positive로 간주)
        threshold = np.percentile(attention_map, threshold_percentile)
        
        # Positive area: threshold 이상
        positive_area = attention_map >= threshold
        
        # Negative area: threshold 미만
        negative_area = attention_map < threshold
        
        positive_ratio = np.sum(positive_area) / attention_map.size
        negative_ratio = np.sum(negative_area) / attention_map.size
        
        print(f"    📊 Threshold 필터링 완료:")
        print(f"      - Threshold: {threshold:.3f} ({threshold_percentile}th percentile)")
        print(f"      - Positive area: {positive_ratio:.1%}")
        print(f"      - Negative area: {negative_ratio:.1%}")
        
        return positive_area, negative_area
    
    def _find_maximum_connected_component(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Connected Component Analysis로 최대 연결 영역 찾기
        VL-SAM 논문: "find the maximum connectivity area as the positive area"
        """
        from scipy import ndimage
        
        # Connected components 라벨링
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            print("    ⚠️ Connected component가 없습니다.")
            return binary_mask
        
        # 각 component의 크기 계산
        component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        
        # 가장 큰 component 찾기
        largest_component_label = np.argmax(component_sizes) + 1
        main_positive_area = labeled_array == largest_component_label
        
        largest_size = component_sizes[largest_component_label - 1]
        total_positive = np.sum(binary_mask)
        
        print(f"    🔗 Connected Component 분석:")
        print(f"      - 전체 components: {num_features}개")
        print(f"      - 최대 component 크기: {largest_size} ({largest_size/total_positive:.1%})")
        
        return main_positive_area
    
    def _vlsam_sample_points(self, attention_map: np.ndarray, positive_area: np.ndarray, 
                           negative_area: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        VL-SAM 논문 방식의 포인트 샘플링
        - Positive area에서 최대 활성화 값을 가진 포인트 샘플링
        - Negative area에서 최소 활성화 값을 가진 포인트 샘플링
        """
        positive_points = []
        negative_points = []
        
        # 1. Positive point 샘플링 (최대 활성화 값)
        if np.sum(positive_area) > 0:
            # Positive area 내에서 attention 값들
            positive_attention_values = attention_map[positive_area]
            max_attention_value = np.max(positive_attention_values)
            
            # 최대값을 가진 모든 위치 찾기
            max_positions = np.where((attention_map == max_attention_value) & positive_area)
            
            if len(max_positions[0]) > 0:
                # Bounding Box 내부의 포인트 우선 선택
                if bbox is not None:
                    bbox_candidates = []
                    for i in range(len(max_positions[0])):
                        y, x = max_positions[0][i], max_positions[1][i]
                        # bbox 내부에 있는지 확인
                        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                            bbox_candidates.append((x, y))
                    
                    if bbox_candidates:
                        # bbox 내부 포인트 중 첫 번째 선택
                        max_x, max_y = bbox_candidates[0]
                        print(f"    ✅ Positive point (bbox 내부): ({max_x}, {max_y}), attention: {max_attention_value:.3f}")
                    else:
                        # bbox 내부에 없으면 일반적인 최대값 포인트 선택
                        max_y, max_x = max_positions[0][0], max_positions[1][0]
                        print(f"    ✅ Positive point (bbox 외부): ({max_x}, {max_y}), attention: {max_attention_value:.3f}")
                else:
                    # bbox가 없으면 첫 번째 최대값 포인트 선택
                    max_y, max_x = max_positions[0][0], max_positions[1][0]
                    print(f"    ✅ Positive point: ({max_x}, {max_y}), attention: {max_attention_value:.3f}")
                
                positive_points.append((max_x, max_y))
        
        # 2. Negative point 샘플링 (최소 활성화 값)
        if np.sum(negative_area) > 0:
            # Negative area 내에서 attention 값들
            negative_attention_values = attention_map[negative_area]
            min_attention_value = np.min(negative_attention_values)
            
            # 최소값을 가진 모든 위치 찾기
            min_positions = np.where((attention_map == min_attention_value) & negative_area)
            
            if len(min_positions[0]) > 0:
                # Bounding Box 외부의 포인트 우선 선택 (negative는 객체 외부가 좋음)
                if bbox is not None:
                    bbox_outside_candidates = []
                    bbox_inside_candidates = []
                    
                    for i in range(len(min_positions[0])):
                        y, x = min_positions[0][i], min_positions[1][i]
                        # bbox 외부에 있는지 확인
                        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                            bbox_outside_candidates.append((x, y))
                        else:
                            bbox_inside_candidates.append((x, y))
                    
                    if bbox_outside_candidates:
                        # bbox 외부 포인트 중 첫 번째 선택 (negative에게 더 적절)
                        min_x, min_y = bbox_outside_candidates[0]
                        print(f"    ❌ Negative point (bbox 외부): ({min_x}, {min_y}), attention: {min_attention_value:.3f}")
                    elif bbox_inside_candidates:
                        # bbox 외부에 없으면 내부 포인트 선택
                        min_x, min_y = bbox_inside_candidates[0]
                        print(f"    ❌ Negative point (bbox 내부): ({min_x}, {min_y}), attention: {min_attention_value:.3f}")
                    else:
                        # 후보가 없으면 일반적인 최소값 포인트 선택
                        min_y, min_x = min_positions[0][0], min_positions[1][0]
                        print(f"    ❌ Negative point (일반): ({min_x}, {min_y}), attention: {min_attention_value:.3f}")
                else:
                    # bbox가 없으면 첫 번째 최소값 포인트 선택
                    min_y, min_x = min_positions[0][0], min_positions[1][0]
                    print(f"    ❌ Negative point: ({min_x}, {min_y}), attention: {min_attention_value:.3f}")
                
                negative_points.append((min_x, min_y))
        
        # 3. Fallback: 포인트가 없는 경우
        if len(positive_points) == 0:
            print("    ⚠️ Positive point를 찾지 못함, 중앙점 사용")
            h, w = attention_map.shape
            positive_points.append((w//2, h//2))
        
        if len(negative_points) == 0:
            print("    ⚠️ Negative point를 찾지 못함, 모서리점 사용")
            negative_points.append((0, 0))
        
        return positive_points, negative_points
    
    def _normalize_attention_map(self, attention_map: np.ndarray) -> np.ndarray:
        """Attention map을 0-1 범위로 정규화"""
        if attention_map.max() > attention_map.min():
            normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        else:
            normalized = attention_map
        return normalized
    
    def _create_bbox_mask(self, shape: Tuple[int, int], bbox: np.ndarray) -> np.ndarray:
        """바운딩 박스 영역의 마스크 생성"""
        mask = np.zeros(shape, dtype=bool)
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 이미지 경계 내로 제한
        x1 = max(0, min(x1, shape[1]-1))
        y1 = max(0, min(y1, shape[0]-1))
        x2 = max(0, min(x2, shape[1]-1))
        y2 = max(0, min(y2, shape[0]-1))
        
        mask[y1:y2, x1:x2] = True
        return mask
    
    def _extract_positive_points(self, 
                                attention_map: np.ndarray, 
                                mask: np.ndarray, 
                                num_points: int, 
                                min_distance: int) -> np.ndarray:
        """High attention 영역에서 positive point 추출"""
        
        # 마스크 적용
        masked_attention = attention_map * mask
        
        # 방법 1: Peak detection을 통한 local maxima 찾기
        local_maxima = self._find_local_maxima(masked_attention, min_distance)
        
        if len(local_maxima) >= num_points:
            # Attention 값에 따라 정렬하고 상위 N개 선택
            attention_values = [masked_attention[y, x] for x, y in local_maxima]
            sorted_indices = np.argsort(attention_values)[::-1]
            selected_points = [local_maxima[i] for i in sorted_indices[:num_points]]
        else:
            # Local maxima가 부족한 경우 추가 샘플링
            selected_points = local_maxima.copy()
            
            # K-means clustering을 통한 추가 포인트 생성
            if len(selected_points) < num_points:
                additional_points = self._cluster_based_sampling(
                    masked_attention, 
                    mask, 
                    num_points - len(selected_points),
                    min_distance,
                    existing_points=selected_points,
                    mode='high'
                )
                selected_points.extend(additional_points)
        
        return np.array(selected_points)
    
    def _extract_negative_points(self, 
                                attention_map: np.ndarray, 
                                mask: np.ndarray, 
                                num_points: int, 
                                min_distance: int,
                                existing_points: np.ndarray) -> np.ndarray:
        """Low attention 영역에서 negative point 추출"""
        
        # 기존 positive points 주변은 제외
        exclusion_mask = self._create_exclusion_mask(
            attention_map.shape, 
            existing_points, 
            exclusion_radius=min_distance
        )
        
        # 마스크 적용 (bbox 내부이면서 exclusion 영역 제외)
        valid_mask = mask & (~exclusion_mask)
        
        # Low attention 영역 추출 (threshold 이하)
        low_attention_threshold = np.percentile(attention_map[valid_mask], 30)
        low_attention_mask = (attention_map <= low_attention_threshold) & valid_mask
        
        if np.sum(low_attention_mask) == 0:
            print("  ⚠️ Low attention 영역이 없어 random sampling 사용")
            return self._random_sampling(valid_mask, num_points)
        
        # K-means clustering을 통한 포인트 추출
        negative_points = self._cluster_based_sampling(
            attention_map, 
            low_attention_mask, 
            num_points,
            min_distance,
            existing_points=existing_points,
            mode='low'
        )
        
        return np.array(negative_points)
    
    def _find_local_maxima(self, attention_map: np.ndarray, min_distance: int) -> List[Tuple[int, int]]:
        """Local maxima 지점 찾기"""
        # Gaussian smoothing 적용
        smoothed = cv2.GaussianBlur(attention_map, (5, 5), 1.0)
        
        # Local maxima 검출
        from scipy.ndimage import maximum_filter
        
        local_maxima = maximum_filter(smoothed, size=min_distance) == smoothed
        
        # Threshold 적용 (상위 70% 이상만)
        threshold = np.percentile(smoothed, 70)
        local_maxima = local_maxima & (smoothed > threshold)
        
        # 좌표 추출
        y_coords, x_coords = np.where(local_maxima)
        points = [(x, y) for x, y in zip(x_coords, y_coords)]
        
        return points
    
    def _cluster_based_sampling(self, 
                               attention_map: np.ndarray, 
                               mask: np.ndarray, 
                               num_points: int,
                               min_distance: int,
                               existing_points: List,
                               mode: str = 'high') -> List[Tuple[int, int]]:
        """Grid-based point sampling (K-means 대신 메모리 효율적인 방법)"""
        
        # 마스크 영역의 좌표 추출
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            return []
        
        # 좌표를 feature로 사용
        coordinates = np.column_stack([x_coords, y_coords])
        
        if len(coordinates) < num_points:
            # 포인트가 부족한 경우 모든 포인트 사용
            sampled_points = [(x, y) for x, y in coordinates[::max(1, len(coordinates)//num_points)]]
        else:
            # Grid-based sampling (K-means 대신 메모리 효율적)
            try:
                h, w = attention_map.shape
                sampled_points = []
                
                # 격자 크기 계산 (좌표 밀도에 따라 조정)
                grid_size = max(3, int(np.sqrt(len(coordinates) / num_points)))
                
                # 격자 기반으로 분산된 포인트 선택
                for i in range(0, h, grid_size):
                    for j in range(0, w, grid_size):
                        # 현재 격자 영역에서 유효한 좌표 찾기
                        grid_coords = []
                        for x, y in coordinates:
                            if i <= y < i + grid_size and j <= x < j + grid_size:
                                grid_coords.append((x, y))
                        
                        if len(grid_coords) > 0:
                            # 격자 내에서 attention 값에 따라 최적 포인트 선택
                            if mode == 'high':
                                # 가장 높은 attention 값의 포인트 선택
                                best_coord = max(grid_coords, key=lambda p: attention_map[p[1], p[0]])
                            else:
                                # 가장 낮은 attention 값의 포인트 선택
                                best_coord = min(grid_coords, key=lambda p: attention_map[p[1], p[0]])
                            
                            # 거리 조건 확인
                            if self._is_valid_point(best_coord, existing_points + sampled_points, min_distance):
                                sampled_points.append(best_coord)
                                
                                if len(sampled_points) >= num_points:
                                    break
                    
                    if len(sampled_points) >= num_points:
                        break
                
                # 포인트가 부족한 경우 간격을 둔 샘플링으로 보완
                if len(sampled_points) < num_points:
                    step = max(1, len(coordinates) // (num_points - len(sampled_points)))
                    for i in range(0, len(coordinates), step):
                        coord = (coordinates[i, 0], coordinates[i, 1])
                        if self._is_valid_point(coord, existing_points + sampled_points, min_distance):
                            sampled_points.append(coord)
                            if len(sampled_points) >= num_points:
                                break
                        
            except Exception as e:
                print(f"  ⚠️ Grid-based sampling 실패: {e}, random sampling 사용")
                sampled_points = self._random_sampling(mask, num_points)
        
        return sampled_points[:num_points]
    
    def _create_exclusion_mask(self, 
                              shape: Tuple[int, int], 
                              points: np.ndarray, 
                              exclusion_radius: int) -> np.ndarray:
        """기존 포인트 주변의 exclusion mask 생성"""
        mask = np.zeros(shape, dtype=bool)
        
        if len(points) == 0:
            return mask
        
        for point in points:
            x, y = point
            # 원형 exclusion 영역 생성
            y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]
            distances = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            mask |= (distances <= exclusion_radius)
        
        return mask
    
    def _random_sampling(self, mask: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
        """Random sampling fallback"""
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            return []
        
        if len(x_coords) <= num_points:
            return [(x, y) for x, y in zip(x_coords, y_coords)]
        
        # Random sampling
        indices = np.random.choice(len(x_coords), size=num_points, replace=False)
        return [(x_coords[i], y_coords[i]) for i in indices]
    
    def _is_valid_point(self, 
                       point: Tuple[int, int], 
                       existing_points: List[Tuple[int, int]], 
                       min_distance: int) -> bool:
        """포인트가 기존 포인트들과 충분한 거리를 가지는지 확인"""
        x, y = point
        
        for ex_x, ex_y in existing_points:
            distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if distance < min_distance:
                return False
        
        return True
    
    def adaptive_point_sampling(self, 
                               attention_map: np.ndarray,
                               bbox: Optional[np.ndarray] = None,
                               target_object_class: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        객체 클래스와 attention 분포에 따른 적응적 point sampling
        
        Args:
            attention_map: Attention map
            bbox: 바운딩 박스
            target_object_class: 타겟 객체 클래스명
            
        Returns:
            Tuple: (positive_points, negative_points, labels)
        """
        print(f"🧠 적응적 Point Sampling (객체: {target_object_class})")
        
        # 객체 클래스에 따른 파라미터 조정
        if target_object_class:
            params = self._get_class_specific_params(target_object_class)
        else:
            params = self._get_default_params()
        
        # Attention 분포 분석
        attention_stats = self._analyze_attention_distribution(attention_map)
        
        # 분포에 따른 파라미터 조정
        adjusted_params = self._adjust_params_by_distribution(params, attention_stats)
        
        print(f"  📊 Attention 분포: mean={attention_stats['mean']:.3f}, std={attention_stats['std']:.3f}")
        print(f"  ⚙️ 사용 파라미터: pos={adjusted_params['num_positive']}, neg={adjusted_params['num_negative']}")
        
        # Point sampling 실행
        return self.sample_points_from_attention(
            attention_map,
            num_positive=adjusted_params['num_positive'],
            num_negative=adjusted_params['num_negative'],
            bbox=bbox,
            min_distance=adjusted_params['min_distance']
        )
    
    def _get_class_specific_params(self, object_class: str) -> Dict:
        """객체 클래스별 파라미터 설정"""
        # 일반적인 객체 클래스별 설정
        class_params = {
            # 작은 객체들
            'person': {'num_positive': 8, 'num_negative': 4, 'min_distance': 15},
            'bottle': {'num_positive': 6, 'num_negative': 3, 'min_distance': 20},
            'cup': {'num_positive': 5, 'num_negative': 3, 'min_distance': 15},
            
            # 중간 크기 객체들
            'car': {'num_positive': 12, 'num_negative': 6, 'min_distance': 25},
            'bike': {'num_positive': 10, 'num_negative': 5, 'min_distance': 20},
            
            # 큰 객체들
            'building': {'num_positive': 15, 'num_negative': 8, 'min_distance': 30},
            'road': {'num_positive': 20, 'num_negative': 10, 'min_distance': 35},
        }
        
        return class_params.get(object_class, self._get_default_params())
    
    def _get_default_params(self) -> Dict:
        """기본 파라미터"""
        return {'num_positive': 10, 'num_negative': 5, 'min_distance': 20}
    
    def _analyze_attention_distribution(self, attention_map: np.ndarray) -> Dict:
        """Attention map의 분포 분석"""
        return {
            'mean': np.mean(attention_map),
            'std': np.std(attention_map),
            'max': np.max(attention_map),
            'min': np.min(attention_map),
            'q75': np.percentile(attention_map, 75),
            'q25': np.percentile(attention_map, 25)
        }
    
    def _adjust_params_by_distribution(self, base_params: Dict, stats: Dict) -> Dict:
        """분포 통계에 따른 파라미터 조정"""
        adjusted = base_params.copy()
        
        # High variance -> 더 많은 포인트 필요
        if stats['std'] > 0.3:
            adjusted['num_positive'] = int(adjusted['num_positive'] * 1.2)
            adjusted['num_negative'] = int(adjusted['num_negative'] * 1.1)
        
        # Low variance -> 포인트 수 감소
        elif stats['std'] < 0.1:
            adjusted['num_positive'] = max(5, int(adjusted['num_positive'] * 0.8))
            adjusted['num_negative'] = max(3, int(adjusted['num_negative'] * 0.8))
        
        return adjusted
    
    def visualize_sampled_points(self, 
                                image: np.ndarray,
                                attention_map: np.ndarray,
                                positive_points: np.ndarray,
                                negative_points: np.ndarray,
                                bbox: Optional[np.ndarray] = None,
                                save_path: str = None) -> None:
        """샘플링된 포인트들을 시각화"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 원본 이미지 + 포인트들
        axes[0].imshow(image)
        
        # Positive points (빨간색)
        if len(positive_points) > 0:
            axes[0].scatter(positive_points[:, 0], positive_points[:, 1], 
                          c='red', s=100, marker='o', alpha=0.8, label='Positive')
        
        # Negative points (파란색)  
        if len(negative_points) > 0:
            axes[0].scatter(negative_points[:, 0], negative_points[:, 1], 
                          c='blue', s=100, marker='x', alpha=0.8, label='Negative')
        
        # 바운딩 박스
        if bbox is not None:
            from matplotlib.patches import Rectangle
            rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=2, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)
        
        axes[0].set_title('Sampled Points on Image')
        axes[0].legend()
        axes[0].axis('off')
        
        # 2. Attention map + 포인트들
        im = axes[1].imshow(attention_map, cmap='jet', alpha=0.8)
        
        if len(positive_points) > 0:
            axes[1].scatter(positive_points[:, 0], positive_points[:, 1], 
                          c='white', s=100, marker='o', edgecolors='black', linewidth=2)
        
        if len(negative_points) > 0:
            axes[1].scatter(negative_points[:, 0], negative_points[:, 1], 
                          c='black', s=100, marker='x', linewidth=3)
        
        axes[1].set_title('Attention Map + Points')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 3. 이미지 + Attention overlay + 포인트들
        axes[2].imshow(image, alpha=0.7)
        axes[2].imshow(attention_map, cmap='jet', alpha=0.4)
        
        if len(positive_points) > 0:
            axes[2].scatter(positive_points[:, 0], positive_points[:, 1], 
                          c='red', s=100, marker='o', alpha=0.9, edgecolors='white', linewidth=2)
        
        if len(negative_points) > 0:
            axes[2].scatter(negative_points[:, 0], negative_points[:, 1], 
                          c='blue', s=100, marker='x', alpha=0.9, linewidth=3)
        
        axes[2].set_title('Combined View')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Point sampling 시각화 저장: {save_path}")
        
        plt.close()


# 사용 예시
if __name__ == '__main__':
    print("=== AttentionPointSampler 테스트 ===")
    
    # 더미 attention map 생성
    attention_map = np.random.rand(224, 224)
    
    # 가우시안 blob 추가 (high attention 영역)
    y, x = np.ogrid[:224, :224]
    center1 = (80, 60)
    center2 = (140, 160)
    
    blob1 = np.exp(-((x - center1[0])**2 + (y - center1[1])**2) / (2 * 20**2))
    blob2 = np.exp(-((x - center2[0])**2 + (y - center2[1])**2) / (2 * 15**2))
    
    attention_map = 0.3 * attention_map + 0.4 * blob1 + 0.3 * blob2
    attention_map = np.clip(attention_map, 0, 1)
    
    # 더미 바운딩 박스
    bbox = np.array([50, 30, 180, 200])
    
    # 더미 이미지
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Point sampler 초기화
    sampler = AttentionPointSampler()
    
    # 기본 point sampling
    print("\n1. 기본 Point Sampling:")
    pos_points, neg_points, labels = sampler.sample_points_from_attention(
        attention_map, 
        num_positive=8, 
        num_negative=4,
        bbox=bbox
    )
    
    print(f"Positive points: {pos_points}")
    print(f"Negative points: {neg_points}")
    print(f"Labels: {labels}")
    
    # 적응적 point sampling
    print("\n2. 적응적 Point Sampling:")
    pos_points_adaptive, neg_points_adaptive, labels_adaptive = sampler.adaptive_point_sampling(
        attention_map,
        bbox=bbox,
        target_object_class='person'
    )
    
    # 시각화
    sampler.visualize_sampled_points(
        dummy_image,
        attention_map,
        pos_points,
        neg_points,
        bbox=bbox,
        save_path="./test_point_sampling.png"
    )
    
    print("\n✅ AttentionPointSampler 테스트 완료!")
