'''''
Amodal / Visible decoder 따로 학습
Retrieval 적용
'''''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import shutil
from datetime import datetime

# EfficientSAM 모듈 임포트
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

# CLIP 모듈 임포트
try:
    import open_clip
    CLIP_AVAILABLE = True
    print("OpenCLIP 모듈 임포트 성공")
except ImportError:
    CLIP_AVAILABLE = False
    print("OpenCLIP 모듈을 찾을 수 없습니다. pip install open_clip_torch로 설치하세요.")

# 벡터 데이터베이스 임포트
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS 모듈 임포트 성공")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS 모듈을 찾을 수 없습니다. pip install faiss-cpu 또는 faiss-gpu로 설치하세요.")

class RetrievalDatabase:
    """Retrieval 데이터베이스 클래스 - CLIP 기반 임베딩 사용"""
    def __init__(self, retrieval_root="/root/datasets/UOAIS/UOAIS-Sim/retrieval", save_retrieval_results=True, output_dir="./outputs"):
        self.retrieval_root = retrieval_root
        self.train_masks_dir = os.path.join(retrieval_root, "train_masks")
        self.class_info_file = os.path.join(retrieval_root, "train_image_class_info.json")
        
        # Retrieval 결과 저장 설정
        self.save_retrieval_results = save_retrieval_results
        self.output_dir = output_dir
        self.retrieval_debug_info = []
        
        # CLIP 모델 초기화
        self.clip_model, self.clip_preprocess = self._initialize_clip()
        
        # 데이터베이스 로드
        self.load_database()
        
        # CLIP 기반 이미지 임베딩 사전 계산
        self._precompute_clip_embeddings()
        
        # clip_embeddings 속성이 없을 경우 빈 딕셔너리로 초기화
        if not hasattr(self, 'clip_embeddings'):
            self.clip_embeddings = {}
            print("CLIP 임베딩이 없어 빈 딕셔너리로 초기화합니다.")
        
        print(f"Retrieval 데이터베이스 로드 완료: {len(self.image_features)}개 이미지")
        print(f"CLIP 모델: {'사용 가능' if self.clip_model else '사용 불가'}")
        print(f"CLIP 임베딩: {len(self.clip_embeddings)}개")
        print(f"Retrieval 결과 저장: {'활성화' if self.save_retrieval_results else '비활성화'}")
    
    def _initialize_clip(self):
        """OpenCLIP 모델 초기화"""
        if not CLIP_AVAILABLE:
            print("OpenCLIP을 사용할 수 없습니다. raw pixel 기반으로 fallback합니다.")
            return None, None
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='openai',
                device=device
            )
            model.eval()  # 평가 모드로 설정
            print(f"OpenCLIP ViT-B-32 모델 로드 완료 (디바이스: {device})")
            return model, preprocess
        except Exception as e:
            print(f"OpenCLIP 모델 로드 실패: {e}")
            return None, None
    
    def load_database(self):
        """데이터베이스 로드 - RGB 이미지 경로와 amodal mask 정보 저장"""
        with open(self.class_info_file, 'r') as f:
            self.class_info = json.load(f)
        
        self.image_features = {}
        self.rgb_paths = {}
        
        # 이미지 경로와 기본 정보 저장
        for img_name, info in self.class_info.items():
            if 'image_info' in info:
                # 상대 경로를 절대 경로로 변환
                relative_path = info['image_info']['file_name']
                # bin/color/2.png -> train/bin/color/2.png로 변환
                if relative_path.startswith('bin/color/'):
                    rgb_path = os.path.join(self.retrieval_root, '..', 'train', relative_path)
                    rgb_path = os.path.abspath(rgb_path)
                else:
                    rgb_path = os.path.join(self.retrieval_root, relative_path)
                
                # RGB 이미지 파일이 존재하는지 확인
                if os.path.exists(rgb_path):
                    self.rgb_paths[img_name] = rgb_path
                    # 기본 feature 정보 저장
                    self.image_features[img_name] = {
                        'surface': info.get('surface', 'object'),
                        'objects': info.get('objects', []),
                        'rgb_path': rgb_path
                    }
                else:
                    print(f"RGB 이미지 파일을 찾을 수 없음: {rgb_path}")
        
    def _precompute_clip_embeddings(self):
        """CLIP 기반 mask 영역 임베딩을 사전 계산하여 저장"""
        if not self.clip_model:
            print("CLIP 모델이 없어 임베딩 사전 계산을 건너뜁니다.")
            return
        
        print("CLIP 기반 mask 영역 임베딩 사전 계산 시작...")
        device = next(self.clip_model.parameters()).device
        
        # 임베딩 저장용 딕셔너리
        self.clip_embeddings = {}
        
        # 처리할 이미지 수 제한 (메모리 효율성을 위해)
        max_images = min(1000, len(self.image_features))
        print(f"임베딩 계산 대상: {max_images}개 이미지")
        
        processed_count = 0
        for img_name, info in list(self.image_features.items())[:max_images]:
            try:
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"임베딩 계산 진행률: {processed_count}/{max_images}")
                
                # RGB 이미지 로드
                rgb_image = Image.open(info['rgb_path']).convert('RGB')
                
                # 해당 이미지의 amodal mask 찾기
                base_name = img_name.replace('.png', '')
                mask_files = [f for f in os.listdir(self.train_masks_dir) 
                            if f.startswith(base_name) and f.endswith('_amodal_mask.png')]
                
                if not mask_files:
                    print(f"[WARNING] {base_name}에 대한 mask 파일을 찾을 수 없음")
                    continue
                
                # 첫 번째 amodal mask 사용
                amodal_mask_path = os.path.join(self.train_masks_dir, mask_files[0])
                mask_image = Image.open(amodal_mask_path).convert('L')
                mask_array = np.array(mask_image)
                
                # Mask 영역의 bbox 계산
                mask_coords = np.where(mask_array > 0)
                if len(mask_coords[0]) == 0:
                    print(f"[WARNING] {base_name}의 mask가 비어있음")
                    continue
                
                y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
                x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
                
                # Mask 영역만 crop
                mask_cropped = rgb_image.crop((x_min, y_min, x_max, y_max))
                
                # CLIP 전처리 및 임베딩 계산
                image_input = self.clip_preprocess(mask_cropped).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    # 정규화
                    image_features = F.normalize(image_features, p=2, dim=1)
                
                # CPU로 이동하여 저장 (메모리 효율성)
                self.clip_embeddings[img_name] = image_features.cpu()
                
            except Exception as e:
                print(f"이미지 {img_name} mask 영역 임베딩 계산 실패: {e}")
                continue
        
        print(f"CLIP mask 영역 임베딩 사전 계산 완료: {len(self.clip_embeddings)}개 이미지")
    
    def get_retrieval_features(self, query_image, query_bbox, top_k=5):
        """쿼리 이미지에 대한 retrieval CLIP feature embeddings 반환"""
        print(f"[DEBUG] get_retrieval_features 시작: top_k={top_k}")
        
        # CLIP 기반 유사도 계산
        top_similar = self.compute_image_similarity(query_image, query_bbox, top_k)
        print(f"[DEBUG] 유사도 계산 결과: {len(top_similar)}개 이미지")
        
        # 디바이스 정보 가져오기
        device = query_image.device
        print(f"[DEBUG] 대상 디바이스: {device}")
        
        retrieval_features = []
        for i, (img_name, similarity, info) in enumerate(top_similar):
            try:
                print(f"[DEBUG] 처리 중: {i+1}/{len(top_similar)} - {img_name}")
                
                # 사전 계산된 CLIP feature embedding 가져오기
                if img_name in self.clip_embeddings:
                    clip_feature = self.clip_embeddings[img_name].to(device)
                    print(f"[DEBUG] CLIP feature shape: {clip_feature.shape}, device: {clip_feature.device}")
                    
                    retrieval_features.append({
                        'image_name': img_name,
                        'similarity': similarity,
                        'clip_feature': clip_feature,  # CLIP feature embedding 직접 사용
                        'surface': info['surface'],
                        'objects': info['objects']
                    })
                    print(f"[DEBUG] {img_name} CLIP feature 처리 완료")
                else:
                    print(f"[WARNING] {img_name}에 대한 CLIP embedding을 찾을 수 없음")
                    
            except Exception as e:
                print(f"[ERROR] CLIP feature 로드 중 오류: {e}")
                print(f"[ERROR] 이미지: {img_name}")
                continue
        
        print(f"[DEBUG] 최종 retrieval features: {len(retrieval_features)}개")
        
        # Retrieval 결과 저장 (디버깅용)
        if self.save_retrieval_results and retrieval_features:
            self._save_retrieval_results(query_image, query_bbox, retrieval_features)
        
        return retrieval_features
    
    def _save_retrieval_results(self, query_image, query_bbox, retrieval_features):
        """Retrieval 결과를 이미지와 JSON으로 저장"""
        try:
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            
            # 저장 디렉토리 생성 (통합된 구조 사용)
            retrieval_debug_dir = os.path.join(self.output_dir, "retrieval_debug")
            save_dir = os.path.join(retrieval_debug_dir, timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            # Query 이미지 저장
            query_pil = self._tensor_to_pil(query_image[0] if query_image.dim() == 4 else query_image)
            query_path = os.path.join(save_dir, "query_image.png")
            query_pil.save(query_path)
            
            # Query bbox 정보 저장
            if query_bbox.dim() == 1:
                x1, y1, x2, y2 = query_bbox.int().tolist()
            else:
                x1, y1, x2, y2 = query_bbox[0].int().tolist()
            
            # Retrieval 결과 이미지들 저장
            retrieval_info = {
                "timestamp": timestamp,
                "query_bbox": [x1, y1, x2, y2],
                "num_retrieved": len(retrieval_features),
                "retrieval_results": []
            }
            
            for i, ret_feat in enumerate(retrieval_features):
                img_name = ret_feat['image_name']
                similarity = ret_feat['similarity']
                surface = ret_feat['surface']
                objects = ret_feat['objects']
                
                # 원본 이미지 복사
                if img_name in self.rgb_paths:
                    src_path = self.rgb_paths[img_name]
                    dst_path = os.path.join(save_dir, f"retrieved_{i+1}_{img_name}")
                    shutil.copy2(src_path, dst_path)
                    
                    # Amodal mask도 복사 (있는 경우)
                    base_name = img_name.replace('.png', '')
                    mask_files = [f for f in os.listdir(self.train_masks_dir) 
                                if f.startswith(base_name) and f.endswith('_amodal_mask.png')]
                    if mask_files:
                        mask_src = os.path.join(self.train_masks_dir, mask_files[0])
                        mask_dst = os.path.join(save_dir, f"retrieved_{i+1}_{base_name}_amodal_mask.png")
                        shutil.copy2(mask_src, mask_dst)
                
                retrieval_info["retrieval_results"].append({
                    "rank": i + 1,
                    "image_name": img_name,
                    "similarity": float(similarity),
                    "surface": surface,
                    "objects": objects,
                    "rgb_path": self.rgb_paths.get(img_name, "N/A")
                })
            
            # JSON 파일로 저장
            json_path = os.path.join(save_dir, "retrieval_info.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(retrieval_info, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] Retrieval 결과 저장 완료: {save_dir}")
            
            # 디버깅 정보에 추가
            self.retrieval_debug_info.append({
                "timestamp": timestamp,
                "save_dir": save_dir,
                "num_retrieved": len(retrieval_features),
                "top_similarity": retrieval_features[0]['similarity'] if retrieval_features else 0.0
            })
            
        except Exception as e:
            print(f"[ERROR] Retrieval 결과 저장 실패: {e}")
    
    def get_retrieval_debug_info(self):
        """저장된 retrieval 디버깅 정보 반환"""
        return self.retrieval_debug_info
    
    def save_retrieval_summary(self, output_dir):
        """전체 retrieval 요약 정보를 JSON으로 저장"""
        try:
            summary_path = os.path.join(output_dir, "retrieval_summary.json")
            summary_info = {
                "total_retrievals": len(self.retrieval_debug_info),
                "retrieval_sessions": self.retrieval_debug_info,
                "database_info": {
                    "total_images": len(self.image_features),
                    "clip_embeddings_available": len(self.clip_embeddings),
                    "retrieval_root": self.retrieval_root
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_info, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Retrieval 요약 정보 저장 완료: {summary_path}")
            
        except Exception as e:
            print(f"[ERROR] Retrieval 요약 정보 저장 실패: {e}")
    
    def compute_image_similarity(self, query_image, query_bbox, top_k=5):
        """CLIP 기반 임베딩을 사용하여 유사도 계산 - Cosine Similarity 사용"""
        similarities = []
        
        print(f"[DEBUG] compute_image_similarity 시작: query_image.shape={query_image.shape}, device={query_image.device}")
        print(f"[DEBUG] query_bbox.shape={query_bbox.shape}, device={query_image.device}")
        
        # CLIP 모델이 없는 경우 fallback
        if not self.clip_model:
            print("[WARNING] CLIP 모델이 없어 raw pixel 기반으로 fallback합니다.")
            return self._compute_raw_pixel_similarity(query_image, query_bbox, top_k)
        
        # 이미지 차원 처리: (C, H, W) -> (1, C, H, W)
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)  # (1, C, H, W)
            print(f"[DEBUG] 이미지 차원 확장 후: {query_image.shape}")
        
        # 디바이스 정보 가져오기
        device = next(self.clip_model.parameters()).device
        print(f"[DEBUG] 사용할 디바이스: {device}")
        
        # GT bbox 영역만 crop
        if query_bbox.dim() == 1:  # (4,) 형태
            x1, y1, x2, y2 = query_bbox.int()
        else:  # (B, 4) 형태
            x1, y1, x2, y2 = query_bbox[0].int()
        
        print(f"[DEBUG] Bbox 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # bbox 유효성 검사
        if x1 >= x2 or y1 >= y2:
            print(f"[WARNING] 잘못된 bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return []
        
        # Query 이미지의 bbox 영역을 CLIP 입력으로 변환
        query_cropped = query_image[:, :, y1:y2, x1:x2]  # bbox 영역만
        print(f"[DEBUG] Query cropped shape: {query_cropped.shape}")
        
        # CLIP 전처리를 위해 PIL Image로 변환
        query_cropped_pil = self._tensor_to_pil(query_cropped[0])  # 첫 번째 배치만 사용
        query_clip_input = self.clip_preprocess(query_cropped_pil).unsqueeze(0).to(device)
        print(f"[DEBUG] Query CLIP input device: {query_clip_input.device}, CLIP model device: {device}")
        
        # Query 이미지의 CLIP 임베딩 계산
        with torch.no_grad():
            query_features = self.clip_model.encode_image(query_clip_input)
            query_features = F.normalize(query_features, p=2, dim=1)
        
        print(f"[DEBUG] Query CLIP 임베딩 shape: {query_features.shape}")
        
        # 사전 계산된 CLIP 임베딩과 비교
        max_images_to_process = min(len(self.clip_embeddings), 500)  # CLIP 임베딩이 있는 이미지만 처리
        print(f"[DEBUG] CLIP 임베딩 기반 처리: {max_images_to_process}개 이미지")
        
        processed_count = 0
        for img_name in list(self.clip_embeddings.keys())[:max_images_to_process]:
            try:
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"[DEBUG] CLIP 유사도 계산 진행률: {processed_count}/{max_images_to_process}")
                
                # 사전 계산된 CLIP 임베딩 가져오기
                stored_features = self.clip_embeddings[img_name].to(device)
                
                # Cosine Similarity 계산
                similarity = F.cosine_similarity(query_features, stored_features, dim=1)[0].item()
                
                # 이미지 정보 가져오기
                info = self.image_features.get(img_name, {})
                similarities.append((img_name, similarity, info))
                
            except Exception as e:
                print(f"[ERROR] CLIP 유사도 계산 중 오류: {e}")
                continue
        
        print(f"[DEBUG] CLIP 기반 유사도 계산 완료: {len(similarities)}개 이미지 처리됨")
        
        # 유사도 기준으로 정렬하여 Top-K 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"[DEBUG] Top-{top_k} 결과 (CLIP 기반):")
        for i, (img_name, sim, info) in enumerate(top_results):
            print(f"  {i+1}. {img_name}: similarity={sim:.4f}, surface={info.get('surface', 'N/A')}")
        
        return top_results
    
    def _compute_raw_pixel_similarity(self, query_image, query_bbox, top_k=5):
        """Raw pixel 기반 유사도 계산 (CLIP이 없을 때 fallback)"""
        similarities = []
        
        print(f"[DEBUG] Raw pixel 기반 유사도 계산 시작")
        
        # 이미지 차원 처리: (C, H, W) -> (1, C, H, W)
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)
        
        # 디바이스 정보 가져오기
        device = query_image.device
        
        # GT bbox 영역만 crop
        if query_bbox.dim() == 1:
            x1, y1, x2, y2 = query_bbox.int()
        else:
            x1, y1, x2, y2 = query_bbox[0].int()
        
        if x1 >= x2 or y1 >= y2:
            print(f"[WARNING] 잘못된 bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return []
        
        query_cropped = query_image[:, :, y1:y2, x1:x2]
        
        # Raw pixel 기반 유사도 계산 (기존 방식)
        max_images_to_process = min(100, len(self.image_features))
        print(f"[DEBUG] Raw pixel 처리: {max_images_to_process}개 이미지")
        
        # 기존 transform 사용
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for img_name, info in list(self.image_features.items())[:max_images_to_process]:
            try:
                rgb_image = Image.open(info['rgb_path']).convert('RGB')
                rgb_tensor = transform(rgb_image).unsqueeze(0).to(device)
                rgb_cropped = rgb_tensor[:, :, y1:y2, x1:x2]
                
                query_flat = query_cropped.reshape(query_cropped.size(0), -1)
                rgb_flat = rgb_cropped.reshape(rgb_cropped.size(0), -1)
                
                similarity = F.cosine_similarity(query_flat, rgb_flat, dim=1)[0].item()
                similarities.append((img_name, similarity, info))
                
            except Exception as e:
                print(f"[ERROR] Raw pixel 처리 중 오류: {e}")
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _tensor_to_pil(self, tensor):
        """텐서를 PIL Image로 변환"""
        # 텐서를 CPU로 이동하고 정규화 해제
        if tensor.device != 'cpu':
            tensor = tensor.cpu()
        
        # 정규화 해제 (ImageNet 정규화)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        
        # [0, 1] 범위로 클리핑
        tensor = torch.clamp(tensor, 0, 1)
        
        # PIL Image로 변환
        tensor = (tensor * 255).byte()
        return transforms.ToPILImage()(tensor)
    
    def compute_image_similarity(self, query_image, query_bbox, top_k=5):
        """CLIP 기반 임베딩을 사용하여 유사도 계산 - Cosine Similarity 사용"""
        similarities = []
        
        print(f"[DEBUG] compute_image_similarity 시작: query_image.shape={query_image.shape}, device={query_image.device}")
        print(f"[DEBUG] query_bbox.shape={query_bbox.shape}, device={query_bbox.device}")
        
        # CLIP 모델이 없는 경우 fallback
        if not self.clip_model:
            print("[WARNING] CLIP 모델이 없어 raw pixel 기반으로 fallback합니다.")
            return self._compute_raw_pixel_similarity(query_image, query_bbox, top_k)
        
        # 이미지 차원 처리: (C, H, W) -> (1, C, H, W)
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)  # (1, C, H, W)
            print(f"[DEBUG] 이미지 차원 확장 후: {query_image.shape}")
        
        # 디바이스 정보 가져오기
        device = query_image.device
        print(f"[DEBUG] 사용할 디바이스: {device}")
        
        # GT bbox 영역만 crop
        if query_bbox.dim() == 1:  # (4,) 형태
            x1, y1, x2, y2 = query_bbox.int()
        else:  # (B, 4) 형태
            x1, y1, x2, y2 = query_bbox[0].int()
        
        print(f"[DEBUG] Bbox 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # bbox 유효성 검사
        if x1 >= x2 or y1 >= y2:
            print(f"[WARNING] 잘못된 bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return []
        
        # Query 이미지의 bbox 영역을 CLIP 입력으로 변환
        query_cropped = query_image[:, :, y1:y2, x1:x2]  # bbox 영역만
        print(f"[DEBUG] Query cropped shape: {query_cropped.shape}")
        
        # CLIP 전처리를 위해 PIL Image로 변환
        query_cropped_pil = self._tensor_to_pil(query_cropped[0])  # 첫 번째 배치만 사용
        query_clip_input = self.clip_preprocess(query_cropped_pil).unsqueeze(0).to(device)
        print(f"[DEBUG] Query CLIP input device: {query_clip_input.device}, CLIP model device: {device}")
        
        # Query 이미지의 CLIP 임베딩 계산
        with torch.no_grad():
            query_features = self.clip_model.encode_image(query_clip_input)
            query_features = F.normalize(query_features, p=2, dim=1)
        
        print(f"[DEBUG] Query CLIP 임베딩 shape: {query_features.shape}")
        
        # 사전 계산된 CLIP 임베딩과 비교
        max_images_to_process = min(len(self.clip_embeddings), 500)  # CLIP 임베딩이 있는 이미지만 처리
        print(f"[DEBUG] CLIP 임베딩 기반 처리: {max_images_to_process}개 이미지")
        
        processed_count = 0
        for img_name in list(self.clip_embeddings.keys())[:max_images_to_process]:
            try:
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"[DEBUG] CLIP 유사도 계산 진행률: {processed_count}/{max_images_to_process}")
                
                # 사전 계산된 CLIP 임베딩 가져오기
                stored_features = self.clip_embeddings[img_name].to(device)
                
                # Cosine Similarity 계산
                similarity = F.cosine_similarity(query_features, stored_features, dim=1)[0].item()
                
                # 이미지 정보 가져오기
                info = self.image_features.get(img_name, {})
                similarities.append((img_name, similarity, info))
                
            except Exception as e:
                print(f"[ERROR] CLIP 유사도 계산 중 오류: {e}")
                continue
        
        print(f"[DEBUG] CLIP 기반 유사도 계산 완료: {len(similarities)}개 이미지 처리됨")
        
        # 유사도 기준으로 정렬하여 Top-K 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"[DEBUG] Top-{top_k} 결과 (CLIP 기반):")
        for i, (img_name, sim, info) in enumerate(top_results):
            print(f"  {i+1}. {img_name}: similarity={sim:.4f}, surface={info.get('surface', 'N/A')}")
        
        return top_results
    
    def get_retrieval_features(self, query_image, query_bbox, top_k=5):
        """쿼리 이미지에 대한 retrieval amodal mask features 반환"""
        print(f"[DEBUG] get_retrieval_features 시작: top_k={top_k}")
        
        top_similar = self.compute_image_similarity(query_image, query_bbox, top_k)
        print(f"[DEBUG] 유사도 계산 결과: {len(top_similar)}개 이미지")
        
        # 디바이스 정보 가져오기
        device = query_image.device
        print(f"[DEBUG] 대상 디바이스: {device}")
        
        retrieval_features = []
        for i, (img_name, similarity, info) in enumerate(top_similar):
            try:
                print(f"[DEBUG] 처리 중: {i+1}/{len(top_similar)} - {img_name}")
                
                # 사전 계산된 CLIP feature embedding 가져오기
                if img_name in self.clip_embeddings:
                    clip_feature = self.clip_embeddings[img_name].to(device)
                    print(f"[DEBUG] CLIP feature shape: {clip_feature.shape}, device: {clip_feature.device}")
                    
                    retrieval_features.append({
                        'image_name': img_name,
                        'similarity': similarity,
                        'clip_feature': clip_feature,  # CLIP feature embedding 직접 사용
                        'surface': info['surface'],
                        'objects': info['objects']
                    })
                    print(f"[DEBUG] {img_name} CLIP feature 처리 완료")
                else:
                    print(f"[WARNING] {img_name}에 대한 CLIP embedding을 찾을 수 없음")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] CLIP feature 로드 중 오류: {e}")
                print(f"[ERROR] 이미지: {img_name}")
                continue
        
        print(f"[DEBUG] 최종 retrieval features: {len(retrieval_features)}개")
        return retrieval_features

class BaselineModelRetrieval(nn.Module):
    """
    모델 A (Baseline) + Retrieval Augmented: Amodal과 Visible 마스크를 별도로 예측하는 모델.
    Retrieval 데이터베이스를 활용하여 이미지 feature를 보강합니다.
    """
    def __init__(self, training_stage='joint', retrieval_enabled=True, output_dir="./outputs"):
        super().__init__()
        self.output_dir = output_dir
        # EfficientSAM 모델 로드
        self.efficient_sam = build_efficient_sam_vitt().eval()
        print("EfficientSAM (ViT-Tiny) 모델 로드 완료.")

        # --- 인코더와 디코더 분리 ---
        self.image_encoder = self.efficient_sam.image_encoder
        self.prompt_encoder = self.efficient_sam.prompt_encoder
        
        # 태스크별 특화된 디코더 생성
        self.amodal_mask_decoder = self._create_amodal_decoder()
        self.visible_mask_decoder = self._create_visible_decoder()
        print("Amodal 및 Visible 마스크를 위한 태스크별 특화 디코더 생성 완료.")

        # Retrieval 데이터베이스 초기화 (UOAIS-Sim만 사용)
        self.retrieval_enabled = retrieval_enabled
        if retrieval_enabled:
            try:
                self.retrieval_db = RetrievalDatabase(save_retrieval_results=True, output_dir=self.output_dir)
                print("UOAIS-Sim Retrieval 데이터베이스 초기화 완료.")
            except Exception as e:
                print(f"UOAIS-Sim Retrieval 데이터베이스 초기화 실패: {e}")
                print("Retrieval 기능을 비활성화합니다.")
                self.retrieval_db = None
                self.retrieval_enabled = False
        else:
            self.retrieval_db = None
        
        # Retrieval 디버깅 정보 수집
        self.retrieval_stats = {
            "total_amodal_retrievals": 0,
            "total_visible_retrievals": 0,
            "retrieval_success_count": 0,
            "retrieval_failure_count": 0,
            "avg_similarity_scores": []
        }

        # 학습 단계 설정
        self.training_stage = training_stage
        self._setup_training_stage(training_stage)
        
        print("태스크별 특화 학습 설정 완료.")

    def _create_amodal_decoder(self):
        """Amodal 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Amodal 특화: 더 넓은 영역을 예측하도록 초기화
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                param.data *= 1.1  # 10% 확장
                print(f"Amodal 디코더 {name} 가중치 확장 적용")
        
        return decoder
    
    def _create_visible_decoder(self):
        """Visible 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Visible 특화: 더 정확한 경계를 예측하도록 초기화
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                param.data *= 0.9  # 10% 축소
                print(f"Visible 디코더 {name} 가중치 축소 적용")
        
        return decoder

    def _setup_training_stage(self, stage):
        """학습 단계별 파라미터 설정"""
        if stage == 'visible_only':
            self._freeze_encoder()
            self._freeze_decoder(self.amodal_mask_decoder)
            self._unfreeze_decoder(self.visible_mask_decoder)
            print("1단계: Visible 디코더만 학습 (Amodal 고정)")
            
        elif stage == 'amodal_only':
            self._freeze_encoder()
            self._freeze_decoder(self.visible_mask_decoder)
            self._unfreeze_decoder(self.amodal_mask_decoder)
            print("2단계: Amodal 디코더만 학습 (Visible 고정)")
            
        elif stage == 'joint':
            self._freeze_encoder()
            self._unfreeze_decoder(self.visible_mask_decoder)
            self._unfreeze_decoder(self.amodal_mask_decoder)
            print("3단계: 전체 디코더 미세조정")
            
        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def _freeze_encoder(self):
        """이미지 인코더 동결"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True

    def _freeze_decoder(self, decoder):
        """특정 디코더 동결"""
        for param in decoder.parameters():
            param.requires_grad = False

    def _unfreeze_decoder(self, decoder):
        """특정 디코더 학습 가능하도록 설정"""
        for param in decoder.parameters():
            param.requires_grad = True

    def set_training_stage(self, stage):
        """학습 단계 동적 변경"""
        self.training_stage = stage
        self._setup_training_stage(stage)

    def compute_consistency_loss(self, amodal_mask, visible_mask):
        """Amodal이 visible을 포함해야 한다는 일관성 제약"""
        amodal_probs = torch.sigmoid(amodal_mask)
        visible_probs = torch.sigmoid(visible_mask)
        
        violation = torch.clamp(visible_probs - amodal_probs, min=0)
        consistency_loss = torch.mean(violation)
        
        return consistency_loss

    def augment_with_retrieval(self, image_embeddings, query_image, query_bbox, mask_type='amodal'):
        """Retrieval을 통한 이미지 feature 보강 - UOAIS-Sim CLIP feature embedding 기반"""
        print(f"[DEBUG] augment_with_retrieval 시작 (mask_type: {mask_type})")
        print(f"[DEBUG] image_embeddings.shape: {image_embeddings.shape}, device: {image_embeddings.device}")
        print(f"[DEBUG] query_image.shape: {query_image.shape}, device: {query_image.device}")
        print(f"[DEBUG] query_bbox.shape: {query_bbox.shape}, device: {query_bbox.device}")
        
        # visible mask 예측 시에는 retrieval 사용하지 않음
        if mask_type == 'visible':
            print(f"[DEBUG] Visible mask 예측 - Retrieval 건너뛰기")
            self.retrieval_stats["total_visible_retrievals"] += 1
            return image_embeddings
        
        if not self.retrieval_enabled:
            print(f"[DEBUG] Retrieval 비활성화")
            return image_embeddings
            
        # Amodal retrieval 통계 업데이트
        self.retrieval_stats["total_amodal_retrievals"] += 1
        
        augmented_embeddings = image_embeddings.clone()
        print(f"[DEBUG] 원본 embeddings 복사 완료")
            
        # UOAIS-Sim Retrieval (mask 영역 기반 CLIP feature embedding)
        if self.retrieval_db is not None:
            try:
                print(f"[DEBUG] UOAIS-Sim Retrieval DB에서 CLIP features 가져오기 시작")
                retrieval_features = self.retrieval_db.get_retrieval_features(query_image, query_bbox, top_k=3)
                
                if retrieval_features:
                    print(f"[DEBUG] {len(retrieval_features)}개 UOAIS-Sim CLIP features 획득")
                    augmented_embeddings = self._apply_uoais_clip_features(augmented_embeddings, retrieval_features)
                    
                    # 성공 통계 업데이트
                    self.retrieval_stats["retrieval_success_count"] += 1
                    if retrieval_features:
                        avg_sim = sum(feat['similarity'] for feat in retrieval_features) / len(retrieval_features)
                        self.retrieval_stats["avg_similarity_scores"].append(avg_sim)
                else:
                    print(f"[DEBUG] UOAIS-Sim retrieval features 없음")
                    self.retrieval_stats["retrieval_failure_count"] += 1
                
            except Exception as e:
                print(f"[ERROR] UOAIS-Sim Retrieval 실패: {e}")
                print(f"[DEBUG] UOAIS-Sim Retrieval 실패로 인해 원본 embeddings 사용")
                self.retrieval_stats["retrieval_failure_count"] += 1
        
        print(f"[DEBUG] Retrieval augmentation 완료 (mask_type: {mask_type})")
        return augmented_embeddings
    
    def _extract_bbox_image(self, query_image, query_bbox):
        """쿼리 이미지에서 bbox 영역을 PIL Image로 추출"""
        try:
            print(f"[DEBUG] bbox 이미지 추출 시작")
            print(f"[DEBUG] query_image.shape: {query_image.shape}, device: {query_image.device}")
            print(f"[DEBUG] query_bbox.shape: {query_bbox.shape}, device: {query_bbox.device}")
            
            # 이미지 차원 처리
            if query_image.dim() == 3:
                query_image = query_image.unsqueeze(0)
                print(f"[DEBUG] 이미지 차원 확장 후: {query_image.shape}")
            
            # bbox 좌표 추출
            if query_bbox.dim() == 1:
                x1, y1, x2, y2 = query_bbox.int()
            else:
                x1, y1, x2, y2 = query_bbox[0].int()
            
            print(f"[DEBUG] bbox 좌표: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # bbox 유효성 검사
            if x1 >= x2 or y1 >= y2:
                print(f"[WARNING] 잘못된 bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                return None
            
            # 이미지 크기 확인
            _, _, img_h, img_w = query_image.shape
            if x2 > img_w or y2 > img_h:
                print(f"[WARNING] bbox가 이미지 범위를 벗어남: bbox=({x1},{y1},{x2},{y2}), image=({img_w},{img_h})")
                # bbox를 이미지 범위로 클리핑
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                print(f"[DEBUG] bbox 클리핑 후: ({x1},{y1},{x2},{y2})")
            
            # bbox 영역 crop
            bbox_cropped = query_image[:, :, y1:y2, x1:x2]
            print(f"[DEBUG] cropped shape: {bbox_cropped.shape}")
            
            # 텐서를 PIL Image로 변환
            bbox_pil = self._tensor_to_pil(bbox_cropped[0])
            print(f"[DEBUG] PIL Image 변환 완료: {bbox_pil.size}")
            return bbox_pil
            
        except Exception as e:
            print(f"[ERROR] bbox 이미지 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_uoais_clip_features(self, embeddings, retrieval_features):
        """UOAIS-Sim CLIP feature embeddings를 원본 embeddings에 직접 적용"""
        batch_size, channels, height, width = embeddings.shape
        device = embeddings.device
            
        for i, ret_feat in enumerate(retrieval_features):
            if i >= 3:  # 최대 3개만 사용
                    break
                
            # CLIP feature embedding 가져오기
            clip_feature = ret_feat['clip_feature']
            similarity = ret_feat['similarity']
            
            print(f"[DEBUG] CLIP feature shape: {clip_feature.shape}, similarity: {similarity:.4f}")
            
            # CLIP feature를 이미지 임베딩 차원에 맞게 확장
            # CLIP feature (1, 512) -> (1, 512, H, W)
            clip_feature_expanded = clip_feature.unsqueeze(-1).unsqueeze(-1).expand(1, clip_feature.size(1), height, width)
            
                    # 유사도 기반 가중치 적용
            weight = similarity * 0.1  # CLIP feature는 더 큰 가중치 사용
            
            # CLIP feature 차원이 이미지 임베딩보다 클 수 있으므로 적절히 조정
            if clip_feature.size(1) <= channels:
                # CLIP feature 차원이 작거나 같은 경우
                embeddings[:, :clip_feature.size(1), :, :] += weight * clip_feature_expanded
            else:
                # CLIP feature 차원이 큰 경우, 처음 channels만 사용
                embeddings += weight * clip_feature_expanded[:, :channels, :, :]
            
            print(f"[DEBUG] CLIP feature 적용 완료: weight={weight:.4f}")
        
        return embeddings
    
    def _tensor_to_pil(self, tensor):
        """텐서를 PIL Image로 변환"""
        # 텐서를 CPU로 이동하고 정규화 해제
        if tensor.device != 'cpu':
            tensor = tensor.cpu()
        
        # 정규화 해제 (ImageNet 정규화)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        
        # [0, 1] 범위로 클리핑
        tensor = torch.clamp(tensor, 0, 1)
        
        # PIL Image로 변환
        tensor = (tensor * 255).byte()
        return transforms.ToPILImage()(tensor)
    
    def get_retrieval_stats(self):
        """Retrieval 통계 정보 반환"""
        stats = self.retrieval_stats.copy()
        if stats["avg_similarity_scores"]:
            stats["avg_similarity"] = sum(stats["avg_similarity_scores"]) / len(stats["avg_similarity_scores"])
            stats["max_similarity"] = max(stats["avg_similarity_scores"])
            stats["min_similarity"] = min(stats["avg_similarity_scores"])
        else:
            stats["avg_similarity"] = 0.0
            stats["max_similarity"] = 0.0
            stats["min_similarity"] = 0.0
        
        # Retrieval DB 디버깅 정보도 포함
        if self.retrieval_db:
            stats["retrieval_db_debug"] = self.retrieval_db.get_retrieval_debug_info()
        
        return stats
    
    def save_retrieval_debug_info(self, output_dir):
        """Retrieval 디버깅 정보를 JSON 파일로 저장"""
        try:
            debug_path = os.path.join(output_dir, "retrieval_debug_info.json")
            debug_info = {
                "model_retrieval_stats": self.get_retrieval_stats(),
                "timestamp": datetime.now().isoformat(),
                "training_stage": self.training_stage,
                "retrieval_enabled": self.retrieval_enabled
            }
            
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Retrieval 디버깅 정보 저장 완료: {debug_path}")
            
            # Retrieval DB 요약 정보도 저장
            if self.retrieval_db and hasattr(self.retrieval_db, 'save_retrieval_summary'):
                self.retrieval_db.save_retrieval_summary(output_dir)
            
        except Exception as e:
            print(f"[ERROR] Retrieval 디버깅 정보 저장 실패: {e}")

    def forward(self, image: torch.Tensor, box: torch.Tensor):
        """
        Args:
            image (torch.Tensor): 입력 이미지. Shape: (B, 3, H, W)
            box (torch.Tensor): 입력 바운딩 박스. Shape: (B, 4) (x1, y1, x2, y2)

        Returns:
            Tuple[torch.Tensor, ...]: 
                - amodal_mask (B, 1, H, W)
                - amodal_iou (B, 1)
                - visible_mask (B, 1, H, W)
                - visible_iou (B, 1)
        """
        batch_size, _, img_h, img_w = image.shape

        # --- 입력 전처리 ---
        points = torch.stack([
            box[:, :2],  # top-left (x1, y1)
            box[:, 2:]   # bottom-right (x2, y2)
        ], dim=1)  # shape: (B, 2, 2)
        
        points = points.unsqueeze(1)
        labels = torch.ones(batch_size, 1, 2, device=image.device, dtype=torch.int)

        try:
            # --- 공통 이미지 및 프롬프트 임베딩 ---
            with torch.no_grad():
                image_embeddings = self.image_encoder(image)
            image_embeddings = image_embeddings.detach().requires_grad_(True)
            
            # 프롬프트 임베딩 가져오기
            rescaled_points = self.efficient_sam.get_rescaled_pts(points, img_h, img_w)
            sparse_embeddings = self.prompt_encoder(
                rescaled_points.reshape(batch_size * 1, 2, 2),
                labels.reshape(batch_size * 1, 2),
            )
            sparse_embeddings = sparse_embeddings.view(batch_size, 1, sparse_embeddings.shape[1], sparse_embeddings.shape[2])
            
            # --- Amodal 마스크 예측 (Retrieval 적용) ---
            if self.retrieval_enabled:
                amodal_embeddings = self.augment_with_retrieval(image_embeddings, image, box, mask_type='amodal')
            else:
                amodal_embeddings = image_embeddings
            
            amodal_logits, amodal_iou = self.amodal_mask_decoder(
                image_embeddings=amodal_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # --- Visible 마스크 예측 (Retrieval 미적용) ---
            visible_embeddings = self.augment_with_retrieval(image_embeddings, image, box, mask_type='visible')
            visible_logits, visible_iou = self.visible_mask_decoder(
                image_embeddings=visible_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # 마스크 후처리 (첫 번째 마스크 선택)
            if len(amodal_logits.shape) == 5:  # (B, Q, N, H, W)
                amodal_mask = amodal_logits[:, 0, 0:1, :, :]  # (B, 1, H, W)
                amodal_iou_best = amodal_iou[:, 0, 0:1]  # (B, 1)
                visible_mask = visible_logits[:, 0, 0:1, :, :]  # (B, 1, H, W)
                visible_iou_best = visible_iou[:, 0, 0:1]  # (B, 1)
            elif len(amodal_logits.shape) == 4:  # (B, N, H, W)
                amodal_mask = amodal_logits[:, 0:1, :, :]  # (B, 1, H, W)
                amodal_iou_best = amodal_iou[:, 0:1]  # (B, 1)
                visible_mask = visible_logits[:, 0:1, :, :]  # (B, 1, H, W)
                visible_iou_best = visible_iou[:, 0:1]  # (B, 1)
            else:
                raise ValueError(f"Unexpected logits shape: amodal={amodal_logits.shape}, visible={visible_logits.shape}")
            
            # 마스크를 원본 이미지 크기로 업샘플링
            if amodal_mask.shape[-2:] != (img_h, img_w):
                amodal_mask = F.interpolate(
                    amodal_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )
            if visible_mask.shape[-2:] != (img_h, img_w):
                visible_mask = F.interpolate(
                    visible_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )

            return amodal_mask, amodal_iou_best, visible_mask, visible_iou_best

        except Exception as e:
            # 모델 추론 실패 시 빈 마스크 반환 (로그 최소화)
            empty_mask = torch.zeros(batch_size, 1, img_h, img_w, device=image.device)
            empty_iou = torch.zeros(batch_size, 1, device=image.device)
            return empty_mask, empty_iou, empty_mask.clone(), empty_iou.clone()

