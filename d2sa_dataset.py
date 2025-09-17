import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np
from pycocotools import mask as maskUtils
import cv2

class D2SADataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None, max_samples=None):
        """
        D2SA 데이터셋을 로드하는 PyTorch Dataset 클래스.

        Args:
            annotation_file (str or list): 어노테이션 파일 경로 또는 경로 리스트.
            image_dir (str): 이미지 파일들이 저장된 디렉토리의 절대 경로.
            transform (callable, optional): 이미지에 적용할 변환.
            max_samples (int, optional): 최대 로드할 샘플 수 (디버깅용).
        """
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transform = transform
        self.max_samples = max_samples

        # 단일 파일 또는 다중 파일 처리
        if isinstance(annotation_file, str):
            annotation_files = [annotation_file]
        else:
            annotation_files = annotation_file

        # 모든 어노테이션 파일 로드 및 병합
        all_images = []
        all_annotations = []
        all_categories = []
        
        for i, ann_file in enumerate(annotation_files):
            print(f"어노테이션 파일 로드 중 ({i+1}/{len(annotation_files)}): {ann_file}")
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # 첫 번째 파일에서 categories와 images 기본 구조 가져오기
            if i == 0:
                all_categories = coco_data['categories']
                
            # 이미지와 어노테이션 추가 (ID 충돌 방지)
            image_id_offset = max([img['id'] for img in all_images], default=0)
            ann_id_offset = max([ann['id'] for ann in all_annotations], default=0)
            
            # 이미지 ID 업데이트
            for img in coco_data['images']:
                if i > 0:  # 첫 번째 파일이 아닌 경우 ID 오프셋 적용
                    img['id'] += image_id_offset
                if img not in all_images:  # 중복 방지
                    all_images.append(img)
            
            # 어노테이션 ID 및 이미지 ID 업데이트
            for ann in coco_data['annotations']:
                if i > 0:  # 첫 번째 파일이 아닌 경우 ID 오프셋 적용
                    ann['id'] += ann_id_offset
                    ann['image_id'] += image_id_offset
                all_annotations.append(ann)
        
        # 병합된 COCO 데이터 생성
        self.coco = {
            'images': all_images,
            'annotations': all_annotations,
            'categories': all_categories
        }
        
        print(f"총 {len(annotation_files)}개 파일 병합 완료:")
        print(f"  - 이미지: {len(all_images)}개")
        print(f"  - 어노테이션: {len(all_annotations)}개")

        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.categories = self.coco['categories']

        # 이미지 ID를 이미지 정보에 매핑
        self.img_id_to_info = {img['id']: img for img in self.images}
        # 카테고리 ID를 카테고리 이름에 매핑
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

        # 어노테이션 필터링 (amodal segmentation이 있는 것만)
        self.valid_annotations = []
        for ann in self.annotations:
            # D2SA 형식 확인: segmentation (amodal)과 visible_mask가 있어야 함
            if 'segmentation' in ann and 'visible_mask' in ann:
                if ann['image_id'] in self.img_id_to_info:
                    self.valid_annotations.append(ann)
        
        # 최대 샘플 수 제한 (디버깅용)
        if max_samples and max_samples < len(self.valid_annotations):
            print(f"샘플 수를 {len(self.valid_annotations)}개에서 {max_samples}개로 제한합니다.")
            # 랜덤하게 샘플링하여 다양한 카테고리를 포함하도록 함
            import random
            random.seed(42)  # 재현성을 위한 시드 설정
            self.valid_annotations = random.sample(self.valid_annotations, max_samples)
            
        print(f"최종적으로 {len(self.valid_annotations)}개의 유효한 amodal 어노테이션을 사용합니다.")
        
        # 카테고리 분포 확인
        category_counts = {}
        for ann in self.valid_annotations:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        print(f"사용된 카테고리 수: {len(category_counts)}개 (전체 60개 중)")

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 반환합니다.
        
        Returns:
            tuple: (image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info)
        """
        ann = self.valid_annotations[idx]
        img_info = self.img_id_to_info[ann['image_id']]
        
        # 이미지 로드
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 에러: {e}")
            # 더미 이미지 생성
            image = Image.new('RGB', (640, 480), color='black')
        
        original_size = image.size  # (width, height)
        
        # 바운딩 박스 추출 (COCO 형식: [x, y, width, height] -> [x1, y1, x2, y2])
        bbox_xywh = ann['bbox']
        bbox = torch.tensor([
            bbox_xywh[0],  # x1
            bbox_xywh[1],  # y1
            bbox_xywh[0] + bbox_xywh[2],  # x2
            bbox_xywh[1] + bbox_xywh[3]   # y2
        ], dtype=torch.float32)
        
        # 카테고리 이름으로 텍스트 생성
        category_name = self.cat_id_to_name.get(ann['category_id'], "unknown")
        text = f"a {category_name}"
        
        # Amodal mask 처리 (전체 객체 영역)
        amodal_mask = self._decode_mask(ann['segmentation'], img_info['height'], img_info['width'])
        
        # Visible mask 처리 (보이는 영역)
        visible_mask = self._decode_mask(ann['visible_mask'], img_info['height'], img_info['width'])
        
        # Invisible mask 처리 (가려진 영역, 선택적)
        if 'invisible_mask' in ann and ann['invisible_mask']:
            invisible_mask = self._decode_mask(ann['invisible_mask'], img_info['height'], img_info['width'])
        else:
            invisible_mask = np.zeros_like(amodal_mask)
        
        # 이미지 변환 적용
        if self.transform:
            image = self.transform(image)
            # 마스크도 같은 크기로 리사이즈
            target_size = (1024, 1024)  # EfficientSAM 입력 크기
            amodal_mask = cv2.resize(amodal_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            visible_mask = cv2.resize(visible_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            invisible_mask = cv2.resize(invisible_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            
            # 바운딩 박스도 스케일링
            scale_x = target_size[0] / original_size[0]
            scale_y = target_size[1] / original_size[1]
            bbox = bbox * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        
        # 텐서로 변환
        amodal_mask = torch.from_numpy(amodal_mask).float().unsqueeze(0)  # (1, H, W)
        visible_mask = torch.from_numpy(visible_mask).float().unsqueeze(0)
        invisible_mask = torch.from_numpy(invisible_mask).float().unsqueeze(0)
        
        # 어노테이션 정보 (텐서로 변환하여 배치 처리 지원)
        annotation_info = {
            'id': torch.tensor(ann['id'], dtype=torch.long),
            'image_id': torch.tensor(ann['image_id'], dtype=torch.long),
            'category_id': torch.tensor(ann['category_id'], dtype=torch.long),
            'area': torch.tensor(ann['area'], dtype=torch.float32),
            'occlude_rate': torch.tensor(ann.get('occlude_rate', 0.0), dtype=torch.float32),
            'occl_depth': torch.tensor(ann.get('occl_depth', 0), dtype=torch.long),
            'file_name': img_info['file_name']
        }
        
        return image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info

    def _decode_mask(self, segmentation, height, width):
        """
        COCO RLE 형식의 segmentation을 binary mask로 디코딩합니다.
        
        Args:
            segmentation: RLE 형식의 segmentation 또는 polygon
            height: 이미지 높이
            width: 이미지 너비
            
        Returns:
            numpy.ndarray: Binary mask (H, W)
        """
        if isinstance(segmentation, dict):
            # RLE 형식
            if isinstance(segmentation['counts'], str):
                # 문자열 형태의 RLE
                rle = segmentation
            else:
                # 리스트 형태의 RLE를 문자열로 변환
                rle = maskUtils.frPyObjects(segmentation, height, width)
            mask = maskUtils.decode(rle)
        elif isinstance(segmentation, list):
            if len(segmentation) == 0:
                return np.zeros((height, width), dtype=np.uint8)
            # Polygon 형식
            rle = maskUtils.frPyObjects(segmentation, height, width)
            mask = maskUtils.decode(rle)
        else:
            # 빈 마스크
            mask = np.zeros((height, width), dtype=np.uint8)
        
        return mask.astype(np.bool_).astype(np.float32)

    def get_category_names(self):
        """카테고리 이름 리스트 반환"""
        return list(self.cat_id_to_name.values())

    def get_annotation_info(self, idx):
        """특정 인덱스의 어노테이션 정보 반환"""
        return self.valid_annotations[idx]

# 사용 예시 및 테스트
if __name__ == '__main__':
    from torchvision import transforms
    
    # 테스트용 변환
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 데이터셋 초기화 (작은 샘플로 테스트)
    dataset = D2SADataset(
        annotation_file="/root/datasets/D2SA/D2S_amodal_augmented.json",
        image_dir="/root/datasets/D2SA/images",
        transform=transform,
        max_samples=5
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"카테고리: {dataset.get_category_names()[:5]}")  # 처음 5개만 출력
    
    # 첫 번째 샘플 테스트
    if len(dataset) > 0:
        image, bbox, text, amodal_mask, visible_mask, invisible_mask, ann_info = dataset[0]
        print(f"이미지 shape: {image.shape}")
        print(f"bbox: {bbox}")
        print(f"text: {text}")
        print(f"amodal_mask shape: {amodal_mask.shape}")
        print(f"visible_mask shape: {visible_mask.shape}")
        print(f"invisible_mask shape: {invisible_mask.shape}")
        print(f"annotation info: {ann_info}")