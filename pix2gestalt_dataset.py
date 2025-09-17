import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import glob
import random

class Pix2GestaltDataset(Dataset):
    def __init__(self, data_root, transform=None, max_samples=None, split_ratio=0.8):
        """
        Pix2Gestalt 데이터셋을 로드하는 PyTorch Dataset 클래스.

        Args:
            data_root (str): pix2gestalt 데이터셋 루트 디렉토리 경로.
            transform (callable, optional): 이미지에 적용할 변환.
            max_samples (int, optional): 최대 로드할 샘플 수 (디버깅용).
            split_ratio (float): train/val 분할 비율 (기본값: 0.8).
        """
        self.data_root = data_root
        self.transform = transform
        self.max_samples = max_samples
        self.split_ratio = split_ratio

        # 디렉토리 경로 설정
        self.occlusion_dir = os.path.join(data_root, "occlusion")
        self.whole_dir = os.path.join(data_root, "whole")
        self.whole_mask_dir = os.path.join(data_root, "whole_mask")
        self.visible_mask_dir = os.path.join(data_root, "visible_object_mask")

        # 모든 파일 목록 가져오기
        self._load_file_list()
        
        # 최대 샘플 수 제한 (분할 전에 적용)
        if max_samples and max_samples < len(self.file_list):
            print(f"샘플 수를 {len(self.file_list)}개에서 {max_samples}개로 제한합니다.")
            random.seed(42)  # 재현성을 위한 시드 설정
            self.file_list = random.sample(self.file_list, max_samples)
        
        # train/val 분할
        self._split_dataset()
            
        print(f"최종적으로 {len(self.file_list)}개의 샘플을 사용합니다.")

    def _load_file_list(self):
        """파일 목록을 로드합니다."""
        print("파일 목록 로드 중...")
        
        # occlusion 디렉토리에서 파일 목록 가져오기
        occlusion_files = glob.glob(os.path.join(self.occlusion_dir, "*.png"))
        
        # 파일명에서 ID 추출하여 정렬
        file_data = []
        for file_path in occlusion_files:
            filename = os.path.basename(file_path)
            # 파일명에서 ID 추출 (예: 10000015_occlusion.png -> 10000015)
            file_id = filename.split('_')[0]
            file_data.append({
                'id': file_id,
                'occlusion_path': file_path,
                'whole_path': os.path.join(self.whole_dir, f"{file_id}_whole.png"),
                'whole_mask_path': os.path.join(self.whole_mask_dir, f"{file_id}_whole_mask.png"),
                'visible_mask_path': os.path.join(self.visible_mask_dir, f"{file_id}_visible_mask.png")
            })
        
        # ID로 정렬
        file_data.sort(key=lambda x: int(x['id']))
        self.file_list = file_data
        
        print(f"총 {len(self.file_list)}개의 파일을 찾았습니다.")

    def _split_dataset(self):
        """데이터셋을 train/val로 분할합니다."""
        total_files = len(self.file_list)
        train_size = int(total_files * self.split_ratio)
        
        # 파일을 ID 순서대로 정렬되어 있으므로 앞의 80%를 train으로 사용
        self.train_files = self.file_list[:train_size]
        self.val_files = self.file_list[train_size:]
        
        print(f"Train: {len(self.train_files)}개, Val: {len(self.val_files)}개")

    def set_split(self, split='train'):
        """사용할 데이터 분할을 설정합니다."""
        if split == 'train':
            self.file_list = self.train_files
        elif split == 'val':
            self.file_list = self.val_files
        else:
            raise ValueError("split은 'train' 또는 'val'이어야 합니다.")
        
        print(f"{split} 분할로 설정: {len(self.file_list)}개 샘플")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        데이터셋에서 하나의 샘플을 반환합니다.
        
        Returns:
            tuple: (image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info)
        """
        file_data = self.file_list[idx]
        
        # 이미지 로드 (occlusion 이미지를 메인 이미지로 사용)
        try:
            image = Image.open(file_data['occlusion_path']).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {file_data['occlusion_path']}, 에러: {e}")
            # 더미 이미지 생성
            image = Image.new('RGB', (640, 480), color='black')
        
        original_size = image.size  # (width, height)
        
        # 바운딩 박스 생성 (이미지 전체 영역)
        bbox = torch.tensor([
            0,  # x1
            0,  # y1
            original_size[0],  # x2
            original_size[1]   # y2
        ], dtype=torch.float32)
        
        # 텍스트 생성 (pix2gestalt는 객체 카테고리 정보가 없으므로 일반적인 텍스트 사용)
        text = "a object"
        
        # Amodal mask 로드 (whole_mask)
        amodal_mask = self._load_mask(file_data['whole_mask_path'], original_size)
        
        # Visible mask 로드 (visible_mask)
        visible_mask = self._load_mask(file_data['visible_mask_path'], original_size)
        
        # Invisible mask 계산 (amodal - visible)
        invisible_mask = np.maximum(0, amodal_mask - visible_mask)
        
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
            'id': torch.tensor(int(file_data['id']), dtype=torch.long),
            'image_id': torch.tensor(int(file_data['id']), dtype=torch.long),
            'category_id': torch.tensor(1, dtype=torch.long),  # pix2gestalt는 카테고리 정보가 없으므로 1로 설정
            'area': torch.tensor(np.sum(amodal_mask.numpy()), dtype=torch.float32),
            'occlude_rate': torch.tensor(1.0 - np.sum(visible_mask.numpy()) / (np.sum(amodal_mask.numpy()) + 1e-8), dtype=torch.float32),
            'occl_depth': torch.tensor(0, dtype=torch.long),  # pix2gestalt는 depth 정보가 없음
            'file_name': f"{file_data['id']}_occlusion.png"
        }
        
        return image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info

    def _load_mask(self, mask_path, original_size):
        """
        마스크 이미지를 로드하고 binary mask로 변환합니다.
        
        Args:
            mask_path: 마스크 파일 경로
            original_size: 원본 이미지 크기 (width, height)
            
        Returns:
            numpy.ndarray: Binary mask (H, W)
        """
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            # 0이 아닌 값들을 1로 변환
            mask = (mask > 0).astype(np.float32)
        except Exception as e:
            print(f"마스크 로드 실패: {mask_path}, 에러: {e}")
            # 빈 마스크 생성
            mask = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
        
        return mask

    def get_category_names(self):
        """카테고리 이름 리스트 반환 (pix2gestalt는 카테고리 정보가 없으므로 기본값)"""
        return ["object"]

    def get_annotation_info(self, idx):
        """특정 인덱스의 어노테이션 정보 반환"""
        return self.file_list[idx]

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
    dataset = Pix2GestaltDataset(
        data_root="/root/datasets/pix2gestalt_occlusions_release",
        transform=transform,
        max_samples=5
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"카테고리: {dataset.get_category_names()}")
    
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
