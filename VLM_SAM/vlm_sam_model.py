"""
VLM-SAM 통합 모델: LLaVA를 사용한 Occlusion 관계 추출 및 SAM을 활용한 마스크 예측
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import re
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np

# EfficientSAM 모듈 임포트
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

# LLaVA 관련 임포트 (main_prompt_VLM_reasoning_250704.py 방식)
try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    
    # 성공적으로 검증된 방식 사용
    LLAVA_MODEL_CLASS = LlavaForConditionalGeneration
    print("✓ LlavaForConditionalGeneration 사용 (검증된 방식)")
    
    print("✓ Transformers 모듈 임포트 성공")
    
    # LLaVA 모델 설정 (검증된 모델 사용)
    DEFAULT_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"  # 성공적으로 검증된 모델
    
    print(f"✓ LLaVA 모델 설정: {DEFAULT_LLAVA_MODEL}")
    print(f"  (HuggingFace Hub에서 자동 다운로드 및 캐시됩니다)")
    LLAVA_AVAILABLE = True
        
except ImportError as e:
    print(f"❌ 치명적 오류: Transformers 라이브러리를 임포트할 수 없습니다: {e}")
    print("VLM-SAM 모델은 transformers 라이브러리가 필수입니다.")
    print("pip install transformers로 설치하세요.")
    raise ImportError(f"Transformers 라이브러리가 필요하지만 임포트할 수 없습니다: {e}")

# Attention Extractor 임포트
from attention_extractor import AttentionExtractor

# Point Sampler 임포트
from point_sampler import AttentionPointSampler

# Integrated Visualizer 임포트
from integrated_visualizer import IntegratedVisualizer

# D2SA Dataset 임포트
from d2sa_dataset import D2SADataset
from torch.utils.data import DataLoader
from torchvision import transforms


class OcclusionAnalyzer:
    """VLM을 사용하여 이미지에서 occlusion 관계를 분석하는 클래스"""
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path (str): 사용할 LLaVA 모델 경로 (None이면 기본 로컬 경로 사용)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 경로 설정
        if model_path is None:
            model_path = DEFAULT_LLAVA_MODEL
        
        print(f"🔄 LLaVA 모델 로딩 시작: {model_path}")
        
        # LLaVA 모델 로딩 (필수) - 여러 모델 시도
        success = False
        last_error = None
        
        # 단일 모델 시도 (검증된 모델 사용)
        try:
            success = self._try_load_model(model_path)
        except Exception as e:
            last_error = e
        
        if not success:
            print(f"❌ 치명적 오류: 모든 LLaVA 모델 로드 실패")
            print("가능한 해결 방법:")
            print("1. 인터넷 연결 확인")
            print("2. HuggingFace 접근 권한 확인")
            print("3. 충분한 메모리 (RAM/VRAM) 확인")
            print("4. transformers 라이브러리 업데이트: pip install --upgrade transformers")
            raise RuntimeError(f"모든 LLaVA 모델 로드 실패. 마지막 오류: {last_error}")
        
        # 프롬프트 템플릿 초기화
        self.__init_prompts__()
    
    def _try_load_model(self, model_path: str) -> bool:
        """개별 모델 로드 시도"""
        try:
            # Processor 로드 (검증된 방식)
            print("📦 LLaVA Processor 로드 중...")
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                use_fast=False  # main_prompt_VLM_reasoning_250704.py와 동일한 설정
            )
            print("✓ LLaVA Processor 로드 완료")
            
            # Model 로드 (검증된 방식)
            print("🤖 LLaVA Model 로드 중...")
            self.model = LLAVA_MODEL_CLASS.from_pretrained(model_path).to(self.device)
            print(f"✓ LLaVA 모델 로드 완료: {model_path}")
            print(f"✓ 디바이스: {self.device}")
            
            # Attention Extractor 초기화
            self.attention_extractor = AttentionExtractor(self.model, self.processor)
            print("✓ Attention Extractor 초기화 완료")
            
            # Point Sampler 초기화
            self.point_sampler = AttentionPointSampler()
            print("✓ Point Sampler 초기화 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패 ({model_path}): {e}")
            return False
    
    def __init_prompts__(self):
        """프롬프트 템플릿 초기화 (생성자 끝에서 호출)"""
        self.occlusion_prompt = (
            "USER: <image>Analyze the image for occlusion relationships.\n"
            "1. Visible Objects: List all clearly visible objects in the scene using basic object names (e.g., 'car', 'person', 'chair'). Separate by commas.\n"
            "2. Occluded Objects: List objects that are partially hidden by other objects. Use basic object names. If none, state 'None'.\n"
            "3. Occlusion Relationships: For each occluded object, identify what is occluding it.\n\n"
            "Respond strictly in this format:\n"
            "VISIBLE_OBJECTS: [answer to 1]\n"
            "OCCLUDED_OBJECTS: [answer to 2]\n"
            "RELATIONSHIPS: [answer to 3]\n"
            "ASSISTANT:"
        )

    def analyze_occlusion(self, image: Image.Image) -> Dict:
        """
        이미지에서 occlusion 관계를 분석합니다.
        
        Args:
            image (PIL.Image): 분석할 이미지
            
        Returns:
            Dict: occlusion 관계 정보
        """
        # LLaVA 모델이 필수이므로 항상 존재해야 함
        assert self.model is not None and self.processor is not None, "LLaVA 모델이 로드되지 않았습니다!"
        
        try:
            # main_prompt_VLM_reasoning_250704.py와 동일한 방식
            inputs = self.processor(
                text=self.occlusion_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # 모델 추론 (main_prompt_VLM_reasoning_250704.py와 동일한 방식)
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # 검증된 설정
                    temperature=0.0     # 검증된 설정
                )
            
            # 응답 디코딩
            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # LLaVA 응답 파싱
            occlusion_data = self._parse_llava_response(response)
            
            return occlusion_data
            
        except Exception as e:
            print(f"❌ Occlusion 분석 중 오류 발생: {e}")
            print("LLaVA 모델을 사용한 VLM 분석이 실패했습니다.")
            # 실제 VLM 분석 실패 시 예외 발생
            raise RuntimeError(f"VLM 분석 실패: {e}")
    
    def extract_attention_maps(self, image: Image.Image, prompt: str = None, use_vlsam_method: bool = True) -> Tuple[Dict, np.ndarray]:
        """
        VLM에서 attention map을 추출합니다.
        
        Args:
            image (PIL.Image): 분석할 이미지
            prompt (str): 사용할 프롬프트 (None이면 기본 프롬프트 사용)
            use_vlsam_method (bool): VL-SAM 논문 방식 사용 여부
            
        Returns:
            Tuple[Dict, np.ndarray]: (layer별 attention maps, 집계된 attention map)
        """
        # LLaVA 모델이 필수이므로 attention_extractor가 항상 존재해야 함
        assert self.attention_extractor is not None, "Attention Extractor가 초기화되지 않았습니다!"
        
        try:
            # 기본 프롬프트 설정
            if prompt is None:
                prompt = "Analyze the objects in this image and their spatial relationships."
            
            print(f"🧠 VLM Attention Map 추출 중 ({'VL-SAM 방식' if use_vlsam_method else '기본 방식'})...")
            
            # VL-SAM 방식 또는 기본 방식 선택
            raw_attention_maps = self.attention_extractor.extract_attention_maps(
                image, prompt, use_vlsam_method=use_vlsam_method
            )
            
            if not raw_attention_maps:
                print("❌ Attention map 추출 실패")
                raise RuntimeError("VLM에서 attention map을 추출할 수 없습니다.")
            
            # 후처리
            processed_maps = self.attention_extractor.process_attention_maps(
                raw_attention_maps, 
                image_size=(1024, 1024)
            )
            
            # 집계
            aggregated_map = self.attention_extractor.aggregate_attention_maps(processed_maps)
            
            print(f"✓ Attention Map 추출 완료: {len(processed_maps)}개 layer")
            
            return processed_maps, aggregated_map
            
        except Exception as e:
            print(f"❌ VLM Attention Map 추출 중요 실패!")
            print(f"   오류 내용: {e}")
            print(f"❌ 실제 VLM attention map을 사용해야 하지만 추출에 실패했습니다.")
            print(f"❌ 실제 VLM attention map을 사용해야 하므로 프로그램을 중단합니다.")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"VLM Attention Map 추출 중요 실패: {e}")
    
    def generate_sam_prompts(self, image: Image.Image, attention_maps: Dict, aggregated_attention: np.ndarray, 
                           bbox: np.ndarray = None, target_class: str = None, use_vlsam_method: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attention map에서 SAM prompt용 point를 생성합니다.
        
        Args:
            image: PIL 이미지
            attention_maps: Layer별 attention maps
            aggregated_attention: 집계된 attention map
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            target_class: 타겟 객체 클래스
            use_vlsam_method: VL-SAM 논문 방식 사용 여부
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (points, labels) - SAM prompt 형식
        """
        # LLaVA 모델이 필수이므로 point_sampler가 항상 존재해야 함
        assert self.point_sampler is not None, "Point Sampler가 초기화되지 않았습니다!"
        
        try:
            print(f"🎯 SAM Prompt 생성 중 ({'VL-SAM 방식' if use_vlsam_method else '적응적 방식'})...")
            
            if use_vlsam_method:
                # VL-SAM 논문 방식 사용
                positive_points, negative_points, labels = self.point_sampler.sample_points_from_attention(
                    aggregated_attention,
                    bbox=bbox,
                    use_vlsam_method=True
                )
            else:
                # 기존 적응적 point sampling 사용
                positive_points, negative_points, labels = self.point_sampler.adaptive_point_sampling(
                    aggregated_attention,
                    bbox=bbox,
                    target_object_class=target_class
                )
            
            # SAM 형식으로 변환 (모든 포인트 결합)
            if len(positive_points) > 0 and len(negative_points) > 0:
                all_points = np.concatenate([positive_points, negative_points], axis=0)
                all_labels = np.concatenate([
                    np.ones(len(positive_points), dtype=int),
                    np.zeros(len(negative_points), dtype=int)
                ], axis=0)
            elif len(positive_points) > 0:
                all_points = positive_points
                all_labels = np.ones(len(positive_points), dtype=int)
            elif len(negative_points) > 0:
                all_points = negative_points
                all_labels = np.zeros(len(negative_points), dtype=int)
            else:
                # Fallback: 바운딩 박스 중심점 사용
                if bbox is not None:
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    all_points = np.array([[center_x, center_y]])
                    all_labels = np.array([1])
                else:
                    all_points = np.array([[512, 512]])
                    all_labels = np.array([1])
            
            print(f"✓ SAM Prompt 생성 완료: {len(all_points)}개 포인트 ({np.sum(all_labels)}개 positive)")
            
            return all_points, all_labels
            
        except Exception as e:
            print(f"❌ SAM Prompt 생성 실패: {e}")
            raise RuntimeError(f"VLM 기반 SAM Prompt 생성 실패: {e}")
    
    def _parse_llava_response(self, response: str) -> Dict:
        print(f"\n===== LLaVA Raw Response =====")
        print(response)
        print("==============================\n")
        
        # 기본값 초기화
        visible_objects = []
        occluded_objects = []
        occlusion_relationships = []
        
        try:
            # ASSISTANT 부분 추출
            assistant_response = response.split("ASSISTANT:")[-1].strip()
            
            # 각 섹션 파싱
            import re
            
            visible_match = re.search(r"VISIBLE_OBJECTS:(.*?)(?=OCCLUDED_OBJECTS:|$)", assistant_response, re.IGNORECASE | re.DOTALL)
            occluded_match = re.search(r"OCCLUDED_OBJECTS:(.*?)(?=RELATIONSHIPS:|$)", assistant_response, re.IGNORECASE | re.DOTALL)
            relationships_match = re.search(r"RELATIONSHIPS:(.*)", assistant_response, re.IGNORECASE | re.DOTALL)
            
            if visible_match:
                visible_str = visible_match.group(1).strip()
                if visible_str and 'none' not in visible_str.lower():
                    visible_objects = [obj.strip() for obj in visible_str.split(',') if obj.strip()]
            
            if occluded_match:
                occluded_str = occluded_match.group(1).strip()
                if occluded_str and 'none' not in occluded_str.lower():
                    occluded_objects = [obj.strip() for obj in occluded_str.split(',') if obj.strip()]
            
            if relationships_match:
                relationships_str = relationships_match.group(1).strip()
                if relationships_str and 'none' not in relationships_str.lower():
                    # "A is occluded by B" 형식 파싱
                    for relationship in relationships_str.split(';'):
                        relationship = relationship.strip()
                        if 'is occluded by' in relationship.lower():
                            parts = relationship.lower().split('is occluded by')
                            if len(parts) == 2:
                                occluded = parts[0].strip()
                                occluder = parts[1].strip()
                                occlusion_relationships.append({
                                    "occluded": occluded,
                                    "occluder": occluder
                                })
            
        except Exception as e:
            print(f"응답 파싱 중 오류: {e}")
        
        # 결과 구성
        result = {
            "visible_objects": visible_objects,
            "occluded_objects": occluded_objects,
            "occlusion_relationships": occlusion_relationships
        }
        
        print(f"파싱 결과: {result}")
        return result
    
    def _parse_response(self, response: str) -> Dict:
        """LLaVA 응답에서 JSON 데이터를 추출하여 파싱합니다."""
        try:
            # JSON 블록 찾기
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                occlusion_data = json.loads(json_str)
                return occlusion_data
            else:
                print("❌ 응답에서 JSON을 찾을 수 없습니다.")
                raise ValueError("LLaVA 응답에서 유효한 JSON을 찾을 수 없습니다.")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            raise ValueError(f"LLaVA 응답의 JSON 파싱 실패: {e}")
    


class VLMSAMModel(nn.Module):
    """
    VLM (LLaVA)과 SAM을 통합한 Zero-shot Amodal Instance Segmentation 모델
    
    1. VLM으로 occlusion 관계 분석
    2. VLM decoder에서 attention map 생성  
    3. Attention map을 point sampling으로 변환
    4. SAM으로 amodal/visible 마스크 예측
    5. D2SA 데이터셋 직접 통합
    """
    
    def __init__(self, llava_model_path=None, use_d2sa=True):
        super().__init__()
        
        print("=== VLM-SAM 통합 모델 초기화 시작 ===")
        
        # 1. Occlusion Analyzer 초기화 (로컬 LLaVA 모델 사용)
        self.occlusion_analyzer = OcclusionAnalyzer(llava_model_path)
        print("✓ Occlusion Analyzer 초기화 완료")
        
        # 2. EfficientSAM 모델 로드
        self.efficient_sam = build_efficient_sam_vitt().eval()
        print("✓ EfficientSAM (ViT-Tiny) 모델 로드 완료")

        # 3. SAM 컴포넌트 분리
        self.image_encoder = self.efficient_sam.image_encoder
        self.prompt_encoder = self.efficient_sam.prompt_encoder
        
        # 4. 태스크별 특화된 디코더 생성
        self.amodal_mask_decoder = self._create_amodal_decoder()
        self.visible_mask_decoder = self._create_visible_decoder()
        print("✓ Amodal 및 Visible 마스크를 위한 태스크별 특화 디코더 생성 완료")
        
        # 5. 모든 SAM 파라미터 고정 (zero-shot이므로 학습 없음)
        self._freeze_sam_parameters()
        print("✓ SAM 파라미터 고정 완료 (Zero-shot 모드)")
        
        # 6. Integrated Visualizer 초기화
        self.visualizer = IntegratedVisualizer()
        print("✓ Integrated Visualizer 초기화 완료")
        
        # 7. D2SA 데이터셋 초기화 (옵션)
        self.use_d2sa = use_d2sa
        self.d2sa_dataset = None
        self.d2sa_dataloader = None
        
        if self.use_d2sa:
            self._initialize_d2sa_dataset()
        
        print("=== VLM-SAM 통합 모델 초기화 완료 ===\n")

    def _create_amodal_decoder(self):
        """Amodal 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Amodal 특화: 더 넓은 영역을 예측하도록 초기화
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                param.data *= 1.1  # 10% 확장
        
        return decoder
    
    def _create_visible_decoder(self):
        """Visible 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Visible 특화: 더 정확한 경계를 예측하도록 초기화
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                param.data *= 0.9  # 10% 축소
        
        return decoder

    def _freeze_sam_parameters(self):
        """SAM의 모든 파라미터를 고정합니다 (Zero-shot 모드)"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.amodal_mask_decoder.parameters():
            param.requires_grad = False
        for param in self.visible_mask_decoder.parameters():
            param.requires_grad = False
    
    def _initialize_d2sa_dataset(self, max_samples: int = 50):
        """D2SA 데이터셋을 초기화합니다."""
        print("📊 D2SA 데이터셋 초기화 중...")
        
        # D2SA 경로 설정
        D2SA_ROOT = "/root/datasets/D2SA"
        TRAIN_ANNOTATION_FILE = os.path.join(D2SA_ROOT, "D2S_amodal_augmented.json")
        IMAGE_DIR = os.path.join(D2SA_ROOT, "images")
        
        # 데이터 변환 설정
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        try:
            self.d2sa_dataset = D2SADataset(
                annotation_file=TRAIN_ANNOTATION_FILE,
                image_dir=IMAGE_DIR,
                transform=transform,
                max_samples=max_samples
            )
            
            self.d2sa_dataloader = DataLoader(
                self.d2sa_dataset, 
                batch_size=1, 
                shuffle=True, 
                num_workers=0
            )
            
            print(f"✓ D2SA 데이터셋 로드 완료: {len(self.d2sa_dataset)}개 샘플")
            
        except Exception as e:
            print(f"❌ D2SA 데이터셋 로드 실패: {e}")
            print("실제 D2SA 데이터가 필요합니다.")
            self.use_d2sa = False
    
    def get_category_name(self, category_id: int) -> str:
        """COCO 카테고리 ID를 카테고리 이름으로 변환"""
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
    
    def get_d2sa_sample(self, index: int = None) -> Tuple[torch.Tensor, torch.Tensor, Image.Image, str, Dict]:
        """
        D2SA 데이터셋에서 샘플을 가져옵니다.
        
        Args:
            index: 특정 인덱스 (None이면 랜덤)
            
        Returns:
            Tuple: (image_tensor, bbox_tensor, pil_image, category_name, annotation_info)
        """
        if not self.use_d2sa or self.d2sa_dataset is None:
            raise RuntimeError("D2SA 데이터셋이 사용 불가능합니다. 실제 데이터가 필요합니다.")
        
        try:
            if index is None:
                # 랜덤 샘플 선택
                index = np.random.randint(0, len(self.d2sa_dataset))
            
            # 데이터 로드 (D2SADataset은 튜플 반환)
            data = self.d2sa_dataset[index]
            image, bbox, text, amodal_mask, visible_mask, invisible_mask, annotation_info = data
            
            # PIL 이미지 생성
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # 카테고리 이름 추출
            category_id = annotation_info['category_id'].item()
            category_name = self.get_category_name(category_id)
            
            # 어노테이션 정보 딕셔너리로 변환
            ann_dict = {
                'image_id': annotation_info['image_id'].item(),
                'category_id': category_id,
                'category_name': category_name,
                'file_name': annotation_info['file_name'],
                'area': annotation_info['area'].item(),
                'occlude_rate': annotation_info.get('occlude_rate', torch.tensor(0.0)).item()
            }
            
            print(f"📸 D2SA 샘플 로드 완료:")
            print(f"  - 인덱스: {index}")
            print(f"  - 이미지 ID: {ann_dict['image_id']}")
            print(f"  - 카테고리: {category_name} (ID: {category_id})")
            print(f"  - 파일명: {ann_dict['file_name']}")
            print(f"  - Occlusion rate: {ann_dict['occlude_rate']:.3f}")
            
            return image.unsqueeze(0), bbox.unsqueeze(0), pil_image, category_name, ann_dict
            
        except Exception as e:
            print(f"❌ D2SA 샘플 로드 실패 (index={index}): {e}")
            raise RuntimeError(f"D2SA 샘플 로드 실패: {e}")
    

    def extract_occlusion_info(self, image_pil: Image.Image, bbox: np.ndarray = None, target_class: str = None, 
                             use_vlsam_method: bool = True, auto_detect_bbox: bool = True) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        VLM을 사용하여 이미지에서 occlusion 정보와 attention map, SAM prompt를 추출합니다.
        
        Args:
            image_pil (PIL.Image): 분석할 이미지
            bbox (np.ndarray, optional): 바운딩 박스 [x1, y1, x2, y2]
            target_class (str, optional): 타겟 객체 클래스
            use_vlsam_method (bool): VL-SAM 논문 방식 사용 여부
            auto_detect_bbox (bool): bbox가 없을 때 자동 탐지 수행 여부
            
        Returns:
            Tuple: (occlusion 관계 정보, layer별 attention maps, 집계된 attention map, sam_points, sam_labels)
        """
        print(f"🔍 VLM을 사용한 Occlusion 관계 분석 시작 ({'VL-SAM 방식' if use_vlsam_method else '기본 방식'})...")
        
        # 1. Occlusion 관계 분석
        occlusion_info = self.occlusion_analyzer.analyze_occlusion(image_pil)
        
        print(f"✓ Occlusion 분석 완료:")
        print(f"  - 보이는 물체: {occlusion_info.get('visible_objects', [])}")
        print(f"  - 가려진 물체: {occlusion_info.get('occluded_objects', [])}")
        print(f"  - Occlusion 관계: {occlusion_info.get('occlusion_relationships', [])}")
        
        # 2. Attention Map 추출 (VL-SAM 방식 선택 가능)
        attention_maps, aggregated_attention = self.occlusion_analyzer.extract_attention_maps(
            image_pil, use_vlsam_method=use_vlsam_method
        )
        
        # 3. SAM Prompt 생성 (VL-SAM 방식 선택 가능)
        sam_points, sam_labels = self.occlusion_analyzer.generate_sam_prompts(
            image_pil, attention_maps, aggregated_attention, bbox, target_class, use_vlsam_method=use_vlsam_method
        )
        
        return occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels

    def forward(self, image: torch.Tensor, box: torch.Tensor, image_pil: Optional[Image.Image] = None, text: str = None):
        """
        메인 forward 함수
        
        Args:
            image (torch.Tensor): 입력 이미지 텐서. Shape: (B, 3, H, W)
            box (torch.Tensor): 입력 바운딩 박스. Shape: (B, 4) (x1, y1, x2, y2)
            image_pil (PIL.Image, optional): PIL 이미지 (VLM 분석용)
            text (str, optional): 객체 클래스 텍스트
            
        Returns:
            Tuple: (amodal_mask, amodal_iou, visible_mask, visible_iou, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels)
        """
        batch_size, _, img_h, img_w = image.shape
        
        # 1. VLM을 사용한 Occlusion 관계 분석, Attention Map 추출, SAM Prompt 생성 (첫 번째 이미지만 사용)
        occlusion_info = None
        attention_maps = {}
        aggregated_attention = np.zeros((img_h, img_w))
        sam_points = np.array([[512, 512]])  # 기본값
        sam_labels = np.array([1])  # 기본값
        
        if image_pil is not None:
            # 타겟 클래스 추출 (텍스트에서)
            target_class = None
            if text:
                # "a clementine" -> "clementine" 형태로 변환
                target_class = text.replace("a ", "").replace("an ", "").strip()
            
            occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = self.extract_occlusion_info(
                image_pil, box[0].cpu().numpy(), target_class, use_vlsam_method=True
            )
        
        # 2. SAM을 사용한 마스크 예측 (VLM-guided points 사용)
        print("🎯 SAM을 사용한 마스크 예측 시작...")
        print(f"  📍 VLM-guided points 사용: {sam_points.shape}")
        
        # VLM에서 추출한 attention-based points를 SAM prompt로 사용
        if len(sam_points) > 0:
            # SAM points를 EfficientSAM 형태로 변환
            # sam_points: (N, 2), sam_labels: (N,)
            num_points = len(sam_points)
            
            # 포인트들을 배치 형태로 재구성: (B, Q, N, 2)
            points = torch.from_numpy(sam_points).float().to(image.device)
            points = points.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            
            # 라벨들도 동일하게 변환: (B, Q, N)
            labels = torch.from_numpy(sam_labels).long().to(image.device)
            labels = labels.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            
            print(f"  ✓ VLM Points 변환 완료: {num_points}개 포인트")
            print(f"    - Positive: {torch.sum(labels).item()}개")
            print(f"    - Negative: {num_points - torch.sum(labels).item()}개")
            
        else:
            # Fallback: 바운딩 박스 중심점 사용
            print("  ⚠️ VLM points가 없어 바운딩 박스 중심점 사용")
            center_x = (box[:, 0] + box[:, 2]) / 2
            center_y = (box[:, 1] + box[:, 3]) / 2
            
            points = torch.stack([center_x, center_y], dim=1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2)
            labels = torch.ones(batch_size, 1, 1, device=image.device, dtype=torch.int)

        try:
            # 공통 이미지 및 프롬프트 임베딩
            with torch.no_grad():
                image_embeddings = self.image_encoder(image)
            image_embeddings = image_embeddings.detach()
            
            # 프롬프트 임베딩 (VLM-guided points 처리)
            if len(sam_points) > 0:
                # 다중 포인트 처리
                num_points = points.shape[2]  # N개 포인트
                
                # Points rescaling
                rescaled_points = self.efficient_sam.get_rescaled_pts(points, img_h, img_w)
                
                # Sparse embeddings 생성
                sparse_embeddings = self.prompt_encoder(
                    rescaled_points.reshape(batch_size, num_points, 2),
                    labels.reshape(batch_size, num_points),
                )
                
                print(f"  ✓ Sparse embeddings 생성: {sparse_embeddings.shape}")
                
            else:
                # Fallback: 단일 포인트 처리
                rescaled_points = self.efficient_sam.get_rescaled_pts(points, img_h, img_w)
                sparse_embeddings = self.prompt_encoder(
                    rescaled_points.reshape(batch_size, 1, 2),
                    labels.reshape(batch_size, 1),
                )
            
            # Sparse embeddings shape 조정
            if len(sparse_embeddings.shape) == 3:
                sparse_embeddings = sparse_embeddings.unsqueeze(1)  # (B, 1, embedding_dim, embedding_size)
            
            # Amodal 마스크 예측
            amodal_logits, amodal_iou = self.amodal_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # Visible 마스크 예측
            visible_logits, visible_iou = self.visible_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # 마스크 후처리
            if len(amodal_logits.shape) == 5:  # (B, Q, N, H, W)
                amodal_mask = amodal_logits[:, 0, 0:1, :, :]
                amodal_iou_best = amodal_iou[:, 0, 0:1]
                visible_mask = visible_logits[:, 0, 0:1, :, :]
                visible_iou_best = visible_iou[:, 0, 0:1]
            elif len(amodal_logits.shape) == 4:  # (B, N, H, W)
                amodal_mask = amodal_logits[:, 0:1, :, :]
                amodal_iou_best = amodal_iou[:, 0:1]
                visible_mask = visible_logits[:, 0:1, :, :]
                visible_iou_best = visible_iou[:, 0:1]
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

            print("✓ 마스크 예측 완료")
            
            return amodal_mask, amodal_iou_best, visible_mask, visible_iou_best, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels

        except Exception as e:
            print(f"모델 추론 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 빈 마스크 반환
            empty_mask = torch.zeros(batch_size, 1, img_h, img_w, device=image.device)
            empty_iou = torch.zeros(batch_size, 1, device=image.device)
            return empty_mask, empty_iou, empty_mask.clone(), empty_iou.clone(), occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels

    def create_pipeline_visualization(self, 
                                    image_pil: Image.Image,
                                    results: Tuple,
                                    bbox: np.ndarray = None,
                                    gt_amodal: Optional[np.ndarray] = None,
                                    gt_visible: Optional[np.ndarray] = None,
                                    save_path: str = None,
                                    title: str = "VLM-SAM Pipeline") -> None:
        """
        전체 파이프라인의 시각화를 생성합니다.
        
        Args:
            image_pil: 원본 PIL 이미지
            results: forward 함수의 반환값
            bbox: 바운딩 박스
            gt_amodal: GT amodal mask (optional)
            gt_visible: GT visible mask (optional)
            save_path: 저장 경로
            title: 제목
        """
        try:
            # Results unpacking
            pred_amodal, pred_amodal_iou, pred_visible, pred_visible_iou, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = results
            
            # PIL 이미지를 numpy array로 변환
            image_np = np.array(image_pil)
            
            # 예측 마스크를 numpy로 변환
            pred_amodal_np = torch.sigmoid(pred_amodal[0]).squeeze().cpu().numpy()
            pred_visible_np = torch.sigmoid(pred_visible[0]).squeeze().cpu().numpy()
            
            print(f"🎨 파이프라인 시각화 생성 중...")
            print(f"  - 이미지 크기: {image_np.shape}")
            print(f"  - Attention maps: {len(attention_maps)}개")
            print(f"  - SAM points: {len(sam_points)}개")
            print(f"  - 예측 마스크: Amodal {pred_amodal_np.shape}, Visible {pred_visible_np.shape}")
            
            # 통합 시각화 생성
            self.visualizer.visualize_complete_pipeline(
                image=image_np,
                attention_maps=attention_maps,
                aggregated_attention=aggregated_attention,
                sam_points=sam_points,
                sam_labels=sam_labels,
                pred_amodal_mask=pred_amodal_np,
                pred_visible_mask=pred_visible_np,
                gt_amodal_mask=gt_amodal,
                gt_visible_mask=gt_visible,
                bbox=bbox,
                save_path=save_path,
                title=title
            )
            
            print(f"✓ 파이프라인 시각화 완료")
            
        except Exception as e:
            print(f"❌ 파이프라인 시각화 실패: {e}")
            import traceback
            traceback.print_exc()
    
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
        try:
            self.visualizer.create_attention_analysis(
                attention_maps=attention_maps,
                aggregated_attention=aggregated_attention,
                save_path=save_path
            )
            print(f"✓ Attention 분석 시각화 완료")
            
        except Exception as e:
            print(f"❌ Attention 분석 시각화 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def process_d2sa_sample(self, index: int = None, save_visualization: bool = True, save_vlm_analysis: bool = True, output_dir: str = "./outputs/d2sa_vlm_sam_250909") -> Dict:
        """
        D2SA 샘플을 처리하고 결과를 반환합니다.
        
        Args:
            index: 처리할 샘플 인덱스 (None이면 랜덤)
            save_visualization: 시각화 저장 여부
            save_vlm_analysis: VLM 분석 결과 JSON 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 처리 결과
        """
        print(f"🚀 D2SA 샘플 처리 시작 (index={index})")
        
        # 출력 디렉토리 생성
        if save_visualization or save_vlm_analysis:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # D2SA 샘플 로드
            image_tensor, bbox_tensor, pil_image, category_name, ann_info = self.get_d2sa_sample(index)
            
            # VLM-SAM 파이프라인 실행
            print(f"🔄 VLM-SAM 파이프라인 실행 중...")
            
            with torch.no_grad():
                results = self.forward(
                    image=image_tensor,
                    box=bbox_tensor,
                    image_pil=pil_image,
                    text=category_name
                )
            
            # 결과 언패킹
            pred_amodal, pred_amodal_iou, pred_visible, pred_visible_iou, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = results
            
            print(f"✅ 파이프라인 실행 완료!")
            print(f"  - Pred Amodal IoU: {pred_amodal_iou[0].item():.3f}")
            print(f"  - Pred Visible IoU: {pred_visible_iou[0].item():.3f}")
            print(f"  - VLM Points: {len(sam_points)}개")
            print(f"  - Attention Layers: {len(attention_maps)}개")
            
            # 시각화 저장
            viz_path = None
            if save_visualization:
                viz_filename = f"d2sa_{ann_info['image_id']}_{category_name}_{index or 'random'}.png"
                viz_path = os.path.join(output_dir, viz_filename)
                
                self.create_pipeline_visualization(
                    image_pil=pil_image,
                    results=results,
                    bbox=bbox_tensor[0].cpu().numpy(),
                    save_path=viz_path,
                    title=f"D2SA Sample - {category_name} (ID: {ann_info['image_id']})"
                )
                
                print(f"🎨 시각화 저장: {viz_path}")
            
            # VLM 분석 결과 JSON 저장
            vlm_json_path = None
            if save_vlm_analysis and occlusion_info:
                vlm_filename = f"vlm_analysis_{ann_info['image_id']}_{category_name}_{index or 'random'}.json"
                vlm_json_path = os.path.join(output_dir, vlm_filename)
                
                # VLM 분석 결과 정리
                vlm_analysis_data = {
                    "image_info": {
                        "image_id": ann_info['image_id'],
                        "file_name": ann_info['file_name'],
                        "category_name": category_name,
                        "category_id": ann_info['category_id'],
                        "occlusion_rate": ann_info.get('occlude_rate', 0.0),
                        "sample_index": index
                    },
                    "vlm_analysis": {
                        "visible_objects": occlusion_info.get('visible_objects', []),
                        "occluded_objects": occlusion_info.get('occluded_objects', []),
                        "occlusion_relationships": occlusion_info.get('occlusion_relationships', [])
                    },
                    "analysis_stats": {
                        "num_visible_objects": len(occlusion_info.get('visible_objects', [])),
                        "num_occluded_objects": len(occlusion_info.get('occluded_objects', [])),
                        "num_occlusion_relationships": len(occlusion_info.get('occlusion_relationships', [])),
                        "all_detected_objects": list(set(
                            occlusion_info.get('visible_objects', []) + 
                            occlusion_info.get('occluded_objects', [])
                        )),
                        "ground_truth_object": category_name
                    },
                    "processing_info": {
                        "pred_amodal_iou": float(pred_amodal_iou[0].item()),
                        "pred_visible_iou": float(pred_visible_iou[0].item()),
                        "num_sam_points": len(sam_points),
                        "num_attention_layers": len(attention_maps)
                    }
                }
                
                # JSON 파일로 저장
                with open(vlm_json_path, 'w', encoding='utf-8') as f:
                    json.dump(vlm_analysis_data, f, indent=2, ensure_ascii=False)
                
                print(f"📄 VLM 분석 결과 저장: {vlm_json_path}")
            
            # 결과 정리
            result = {
                "sample_info": ann_info,
                "processing_results": {
                    "pred_amodal_iou": float(pred_amodal_iou[0].item()),
                    "pred_visible_iou": float(pred_visible_iou[0].item()),
                    "num_sam_points": len(sam_points),
                    "num_positive_points": int(np.sum(sam_labels)) if len(sam_labels) > 0 else 0,
                    "num_negative_points": int(len(sam_points) - np.sum(sam_labels)) if len(sam_labels) > 0 else 0,
                    "num_attention_layers": len(attention_maps),
                    "attention_stats": {
                        "mean": float(np.mean(aggregated_attention)),
                        "std": float(np.std(aggregated_attention)),
                        "max": float(np.max(aggregated_attention)),
                        "min": float(np.min(aggregated_attention))
                    }
                },
                "vlm_analysis": occlusion_info,
                "visualization_path": viz_path if save_visualization else None,
                "vlm_json_path": vlm_json_path if save_vlm_analysis else None,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ D2SA 샘플 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "sample_info": {"error": str(e)},
                "status": "error",
                "error_details": traceback.format_exc()
            }
    
    def process_multiple_d2sa_samples(self, num_samples: int = 5, save_vlm_analysis: bool = True, output_dir: str = "./outputs/d2sa_vlm_sam_250909") -> List[Dict]:
        """
        여러 D2SA 샘플을 배치 처리합니다.
        
        Args:
            num_samples: 처리할 샘플 수
            save_vlm_analysis: VLM 분석 결과 JSON 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            List[Dict]: 처리 결과 리스트
        """
        print(f"🚀 D2SA 배치 처리 시작: {num_samples}개 샘플")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        successful = 0
        failed = 0
        
        for i in range(num_samples):
            print(f"\n--- 샘플 {i+1}/{num_samples} 처리 중 ---")
            
            try:
                result = self.process_d2sa_sample(
                    index=None,  # 랜덤 선택
                    save_visualization=True,
                    save_vlm_analysis=save_vlm_analysis,
                    output_dir=output_dir
                )
                
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ 샘플 {i+1} 처리 중 오류: {e}")
                failed += 1
                
                error_result = {
                    "sample_index": i,
                    "status": "error",
                    "error": str(e)
                }
                results.append(error_result)
        
        print(f"\n📊 배치 처리 완료!")
        print(f"  - 성공: {successful}개")
        print(f"  - 실패: {failed}개")
        print(f"  - 성공률: {successful/num_samples*100:.1f}%")
        
        # VLM 분석 통합 요약 생성
        vlm_summary = None
        if save_vlm_analysis:
            vlm_summary = self._create_vlm_analysis_summary(results, output_dir)
        
        # 결과 요약 저장
        summary = {
            "total_samples": num_samples,
            "successful": successful,
            "failed": failed,
            "success_rate": successful/num_samples*100,
            "output_directory": output_dir,
            "vlm_analysis_enabled": save_vlm_analysis,
            "vlm_summary_path": vlm_summary.get("summary_path") if vlm_summary else None,
            "results": results
        }
        
        summary_path = os.path.join(output_dir, "batch_processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 배치 처리 요약 저장: {summary_path}")
        
        return results
    
    def _create_vlm_analysis_summary(self, results: List[Dict], output_dir: str) -> Dict:
        """
        VLM 분석 결과들의 통합 요약을 생성합니다.
        
        Args:
            results: 처리 결과 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: VLM 분석 요약 정보
        """
        print(f"📊 VLM 분석 통합 요약 생성 중...")
        
        successful_results = [r for r in results if r.get("status") == "success" and r.get("vlm_analysis")]
        
        if not successful_results:
            print("⚠️ 성공적인 VLM 분석 결과가 없어 요약을 생성할 수 없습니다.")
            return {"summary_path": None}
        
        # 모든 감지된 객체 클래스 수집
        all_visible_objects = []
        all_occluded_objects = []
        all_detected_objects = []
        object_frequency = {}
        category_analysis = {}
        
        for result in successful_results:
            vlm_data = result["vlm_analysis"]
            sample_info = result["sample_info"]
            
            # 각 카테고리별 분석
            category_name = sample_info["category_name"]
            if category_name not in category_analysis:
                category_analysis[category_name] = {
                    "count": 0,
                    "visible_objects": [],
                    "occluded_objects": [],
                    "avg_visible_count": 0,
                    "avg_occluded_count": 0,
                    "ground_truth_detected": 0
                }
            
            category_info = category_analysis[category_name]
            category_info["count"] += 1
            
            # 가시적 객체
            visible_objs = vlm_data.get("visible_objects", [])
            all_visible_objects.extend(visible_objs)
            category_info["visible_objects"].extend(visible_objs)
            
            # 가려진 객체
            occluded_objs = vlm_data.get("occluded_objects", [])
            all_occluded_objects.extend(occluded_objs)
            category_info["occluded_objects"].extend(occluded_objs)
            
            # 모든 감지된 객체
            detected_objs = visible_objs + occluded_objs
            all_detected_objects.extend(detected_objs)
            
            # Ground Truth 객체가 감지되었는지 확인
            if category_name.lower() in [obj.lower() for obj in detected_objs]:
                category_info["ground_truth_detected"] += 1
            
            # 객체 빈도 계산
            for obj in detected_objs:
                obj_lower = obj.lower()
                object_frequency[obj_lower] = object_frequency.get(obj_lower, 0) + 1
        
        # 카테고리별 통계 계산
        for category_name, info in category_analysis.items():
            if info["count"] > 0:
                info["avg_visible_count"] = len(info["visible_objects"]) / info["count"]
                info["avg_occluded_count"] = len(info["occluded_objects"]) / info["count"]
                info["ground_truth_detection_rate"] = info["ground_truth_detected"] / info["count"] * 100
                # 중복 제거
                info["unique_visible_objects"] = list(set(info["visible_objects"]))
                info["unique_occluded_objects"] = list(set(info["occluded_objects"]))
        
        # 전체 통계
        total_stats = {
            "total_samples_analyzed": len(successful_results),
            "total_unique_objects_detected": len(set(all_detected_objects)),
            "total_unique_visible_objects": len(set(all_visible_objects)),
            "total_unique_occluded_objects": len(set(all_occluded_objects)),
            "most_frequent_objects": sorted(object_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            "avg_objects_per_image": len(all_detected_objects) / len(successful_results) if successful_results else 0,
            "avg_visible_per_image": len(all_visible_objects) / len(successful_results) if successful_results else 0,
            "avg_occluded_per_image": len(all_occluded_objects) / len(successful_results) if successful_results else 0
        }
        
        # VLM 분석 요약 데이터
        vlm_analysis_summary = {
            "summary_info": {
                "analysis_date": json.dumps({"timestamp": "generated"}).replace('"generated"', f'"{str(np.datetime64("now"))[:-10]}"'),
                "total_samples": len(successful_results),
                "analysis_method": "LLaVA-1.5-7b VLM"
            },
            "overall_statistics": total_stats,
            "category_analysis": category_analysis,
            "detailed_results": [
                {
                    "image_id": r["sample_info"]["image_id"],
                    "file_name": r["sample_info"]["file_name"],
                    "ground_truth_category": r["sample_info"]["category_name"],
                    "detected_visible": r["vlm_analysis"].get("visible_objects", []),
                    "detected_occluded": r["vlm_analysis"].get("occluded_objects", []),
                    "occlusion_relationships": r["vlm_analysis"].get("occlusion_relationships", []),
                    "processing_success": True
                }
                for r in successful_results
            ]
        }
        
        # VLM 분석 요약 JSON 저장
        vlm_summary_path = os.path.join(output_dir, "vlm_analysis_summary.json")
        with open(vlm_summary_path, 'w', encoding='utf-8') as f:
            json.dump(vlm_analysis_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 VLM 분석 통합 요약 저장: {vlm_summary_path}")
        print(f"📊 요약 통계:")
        print(f"  - 분석된 샘플: {total_stats['total_samples_analyzed']}개")
        print(f"  - 감지된 고유 객체: {total_stats['total_unique_objects_detected']}개")
        print(f"  - 이미지당 평균 객체: {total_stats['avg_objects_per_image']:.1f}개")
        print(f"  - 가장 빈번한 객체: {total_stats['most_frequent_objects'][:3]}")
        
        return {
            "summary_path": vlm_summary_path,
            "statistics": total_stats,
            "category_analysis": category_analysis
        }
    
    def iterative_refinement(self, image: torch.Tensor, initial_points: np.ndarray, 
                           initial_labels: np.ndarray, attention_map: np.ndarray,
                           num_iterations: int = 2) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        3.4 Iterative Refinement 구현
        PerSAM 방식의 cascaded post-refinement와 마스크된 attention map을 이용한 반복적 개선
        
        Args:
            image: 입력 이미지 텐서
            initial_points: 초기 SAM 포인트
            initial_labels: 초기 SAM 라벨
            attention_map: 집계된 attention map
            num_iterations: 반복 횟수
            
        Returns:
            Tuple: (최종 amodal mask, 최종 visible mask, 반복 결과 리스트)
        """
        print(f"🔄 Iterative Refinement 시작: {num_iterations}회 반복")
        
        batch_size, _, img_h, img_w = image.shape
        current_points = initial_points.copy()
        current_labels = initial_labels.copy()
        refinement_results = []
        
        # 초기 마스크 생성
        amodal_masks = []
        visible_masks = []
        
        for iteration in range(num_iterations + 1):  # 0: 초기, 1~N: 반복
            print(f"  📍 반복 {iteration}/{num_iterations}")
            
            # SAM 마스크 예측
            amodal_mask, visible_mask = self._predict_sam_masks(
                image, current_points, current_labels
            )
            
            amodal_masks.append(amodal_mask)
            visible_masks.append(visible_mask)
            
            # 반복 결과 저장
            iteration_result = {
                "iteration": iteration,
                "points": current_points.copy(),
                "labels": current_labels.copy(),
                "amodal_iou": self._calculate_mask_iou(amodal_mask, None),  # GT 없이 기본값
                "visible_iou": self._calculate_mask_iou(visible_mask, None)
            }
            refinement_results.append(iteration_result)
            
            if iteration < num_iterations:
                # 다음 반복을 위한 포인트 업데이트
                if iteration == 0:
                    # 첫 번째 반복: PerSAM 방식 - 마스크를 추가 프롬프트로 사용
                    current_points, current_labels = self._per_sam_refinement(
                        current_points, current_labels, amodal_mask, visible_mask
                    )
                else:
                    # 후속 반복: 마스크된 attention map에서 새로운 포인트 생성
                    masked_attention = self._mask_attention_map(attention_map, amodal_mask)
                    new_points, new_labels = self._generate_points_from_masked_attention(
                        masked_attention, current_points, current_labels
                    )
                    current_points = new_points
                    current_labels = new_labels
                
                print(f"    ✓ 포인트 업데이트: {len(current_points)}개")
        
        # 최종 마스크 선택 (마지막 반복 결과)
        final_amodal = amodal_masks[-1]
        final_visible = visible_masks[-1]
        
        print(f"✅ Iterative Refinement 완료: {len(refinement_results)}개 반복")
        return final_amodal, final_visible, refinement_results
    
    def multi_scale_ensemble(self, image_pil: Image.Image, target_class: str = None,
                           num_iterations: int = 2) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        3.5 Multi-scale Ensemble 구현
        이미지를 4개 서브이미지로 분할하여 처리하고 NMS로 집계
        
        Args:
            image_pil: 입력 PIL 이미지
            target_class: 타겟 객체 클래스
            num_iterations: 각 서브이미지에서의 반복 횟수
            
        Returns:
            Tuple: (집계된 amodal mask, 집계된 visible mask, 서브이미지 결과들)
        """
        print(f"🔍 Multi-scale Ensemble 시작: 4개 서브이미지 처리")
        
        w, h = image_pil.size
        sub_image_size = (w // 2, h // 2)
        
        # 4개 서브이미지 생성 (4개 모서리에서)
        sub_images = self._create_sub_images(image_pil, sub_image_size)
        
        sub_results = []
        all_amodal_masks = []
        all_visible_masks = []
        
        for i, (sub_image, offset) in enumerate(sub_images):
            print(f"  📸 서브이미지 {i+1}/4 처리 중...")
            
            # 서브이미지에 대한 VLM-SAM 파이프라인 실행
            sub_result = self._process_sub_image(
                sub_image, target_class, offset, num_iterations
            )
            
            sub_results.append(sub_result)
            
            if sub_result["status"] == "success":
                all_amodal_masks.append(sub_result["amodal_mask"])
                all_visible_masks.append(sub_result["visible_mask"])
        
        # NMS로 결과 집계
        if len(all_amodal_masks) > 0:
            print(f"  🔄 NMS 집계 중... ({len(all_amodal_masks)}개 서브이미지 결과)")
            
            # 서브이미지 마스크들을 원본 크기로 복원하고 집계
            final_amodal, final_visible = self._aggregate_with_nms(
                all_amodal_masks, all_visible_masks, sub_results, (w, h)
            )
            
            print(f"✅ Multi-scale Ensemble 완료")
            return final_amodal, final_visible, {"sub_results": sub_results}
        else:
            print(f"❌ 모든 서브이미지 처리 실패")
            # Fallback: 원본 이미지로 처리
            return self._fallback_original_image_processing(image_pil, target_class)
    
    def _create_sub_images(self, image_pil: Image.Image, sub_size: tuple) -> List[Tuple[Image.Image, tuple]]:
        """4개 서브이미지 생성 (4개 모서리에서)"""
        w, h = image_pil.size
        sub_w, sub_h = sub_size
        
        sub_images = []
        
        # 4개 모서리 위치
        positions = [
            (0, 0),  # 좌상단
            (w - sub_w, 0),  # 우상단
            (0, h - sub_h),  # 좌하단
            (w - sub_w, h - sub_h)  # 우하단
        ]
        
        for x, y in positions:
            # 서브이미지 크기 조정 (이미지 경계 내로)
            actual_w = min(sub_w, w - x)
            actual_h = min(sub_h, h - y)
            
            if actual_w > 0 and actual_h > 0:
                sub_image = image_pil.crop((x, y, x + actual_w, y + actual_h))
                # 원본 크기로 리사이즈
                sub_image = sub_image.resize((sub_w, sub_h), Image.Resampling.LANCZOS)
                sub_images.append((sub_image, (x, y)))
        
        return sub_images
    
    def _process_sub_image(self, sub_image: Image.Image, target_class: str, 
                          offset: tuple, num_iterations: int) -> Dict:
        """개별 서브이미지 처리"""
        try:
            # 서브이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(sub_image).unsqueeze(0)
            
            # VLM 분석 및 attention map 추출
            occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = self.extract_occlusion_info(
                sub_image, bbox=None, target_class=target_class, use_vlsam_method=True, auto_detect_bbox=True
            )
            
            if len(sam_points) == 0:
                return {"status": "error", "error": "No SAM points generated"}
            
            # Iterative Refinement 적용
            final_amodal, final_visible, refinement_results = self.iterative_refinement(
                image_tensor, sam_points, sam_labels, aggregated_attention, num_iterations
            )
            
            return {
                "status": "success",
                "amodal_mask": final_amodal,
                "visible_mask": final_visible,
                "refinement_results": refinement_results,
                "offset": offset,
                "occlusion_info": occlusion_info
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _aggregate_with_nms(self, amodal_masks: List[torch.Tensor], visible_masks: List[torch.Tensor],
                           sub_results: List[Dict], original_size: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """NMS를 사용하여 서브이미지 결과들을 집계"""
        w, h = original_size
        
        # 각 서브이미지 마스크를 원본 크기로 복원
        restored_amodal_masks = []
        restored_visible_masks = []
        
        for i, (amodal_mask, visible_mask, sub_result) in enumerate(zip(amodal_masks, visible_masks, sub_results)):
            if sub_result["status"] != "success":
                continue
                
            offset_x, offset_y = sub_result["offset"]
            
            # 원본 크기로 복원
            restored_amodal = self._restore_mask_to_original_size(
                amodal_mask, (w, h), (offset_x, offset_y)
            )
            restored_visible = self._restore_mask_to_original_size(
                visible_mask, (w, h), (offset_x, offset_y)
            )
            
            restored_amodal_masks.append(restored_amodal)
            restored_visible_masks.append(restored_visible)
        
        if len(restored_amodal_masks) == 0:
            # Fallback: 빈 마스크 반환
            empty_amodal = torch.zeros(1, 1, h, w)
            empty_visible = torch.zeros(1, 1, h, w)
            return empty_amodal, empty_visible
        
        # 마스크들을 평균으로 집계 (간단한 NMS 대안)
        final_amodal = torch.stack(restored_amodal_masks).mean(dim=0)
        final_visible = torch.stack(restored_visible_masks).mean(dim=0)
        
        # 임계값 적용
        final_amodal = (torch.sigmoid(final_amodal) > 0.5).float()
        final_visible = (torch.sigmoid(final_visible) > 0.5).float()
        
        return final_amodal, final_visible
    
    def _restore_mask_to_original_size(self, mask: torch.Tensor, original_size: tuple, 
                                     offset: tuple) -> torch.Tensor:
        """마스크를 원본 크기로 복원"""
        w, h = original_size
        offset_x, offset_y = offset
        
        # 원본 크기로 리사이즈
        restored = F.interpolate(
            mask, size=(h, w), mode='bilinear', align_corners=False
        )
        
        # 오프셋 적용 (서브이미지 위치에 맞게)
        if offset_x > 0 or offset_y > 0:
            # 오프셋이 있는 경우 해당 영역만 유지
            result = torch.zeros(1, 1, h, w, device=mask.device)
            sub_h, sub_w = mask.shape[-2:]
            result[:, :, offset_y:offset_y+sub_h, offset_x:offset_x+sub_w] = restored
            return result
        
        return restored
    
    def _fallback_original_image_processing(self, image_pil: Image.Image, target_class: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """서브이미지 처리 실패 시 원본 이미지로 fallback"""
        print("  🔄 Fallback: 원본 이미지로 처리")
        
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image_pil).unsqueeze(0)
        
        # 기본 VLM-SAM 파이프라인 실행
        occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = self.extract_occlusion_info(
            image_pil, bbox=None, target_class=target_class, use_vlsam_method=True, auto_detect_bbox=True
        )
        
        if len(sam_points) > 0:
            amodal_mask, visible_mask = self._predict_sam_masks(image_tensor, sam_points, sam_labels)
        else:
            # 빈 마스크 반환
            h, w = image_tensor.shape[-2:]
            amodal_mask = torch.zeros(1, 1, h, w)
            visible_mask = torch.zeros(1, 1, h, w)
        
        return amodal_mask, visible_mask
    
    def _predict_sam_masks(self, image: torch.Tensor, points: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAM을 사용하여 마스크 예측"""
        batch_size, _, img_h, img_w = image.shape
        
        # 포인트를 텐서로 변환
        points_tensor = torch.from_numpy(points).float().to(image.device)
        points_tensor = points_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        
        labels_tensor = torch.from_numpy(labels).long().to(image.device)
        labels_tensor = labels_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        
        with torch.no_grad():
            # 이미지 임베딩
            image_embeddings = self.image_encoder(image)
            image_embeddings = image_embeddings.detach()
            
            # 포인트 리스케일링
            rescaled_points = self.efficient_sam.get_rescaled_pts(points_tensor, img_h, img_w)
            
            # Sparse embeddings 생성
            sparse_embeddings = self.prompt_encoder(
                rescaled_points.reshape(batch_size, len(points), 2),
                labels_tensor.reshape(batch_size, len(points)),
            )
            
            if len(sparse_embeddings.shape) == 3:
                sparse_embeddings = sparse_embeddings.unsqueeze(1)
            
            # Amodal 마스크 예측
            amodal_logits, _ = self.amodal_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # Visible 마스크 예측
            visible_logits, _ = self.visible_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # 마스크 후처리
            if len(amodal_logits.shape) == 5:
                amodal_mask = amodal_logits[:, 0, 0:1, :, :]
                visible_mask = visible_logits[:, 0, 0:1, :, :]
            elif len(amodal_logits.shape) == 4:
                amodal_mask = amodal_logits[:, 0:1, :, :]
                visible_mask = visible_logits[:, 0:1, :, :]
            else:
                raise ValueError(f"Unexpected logits shape: amodal={amodal_logits.shape}")
            
            # 원본 이미지 크기로 업샘플링
            if amodal_mask.shape[-2:] != (img_h, img_w):
                amodal_mask = F.interpolate(
                    amodal_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )
            if visible_mask.shape[-2:] != (img_h, img_w):
                visible_mask = F.interpolate(
                    visible_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )
            
            return amodal_mask, visible_mask
    
    def _per_sam_refinement(self, points: np.ndarray, labels: np.ndarray, 
                          amodal_mask: torch.Tensor, visible_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """PerSAM 방식의 cascaded post-refinement"""
        print("    🔄 PerSAM 방식 적용: 마스크를 추가 프롬프트로 사용")
        
        # 마스크에서 추가 포인트 추출
        amodal_np = torch.sigmoid(amodal_mask[0]).squeeze().cpu().numpy()
        visible_np = torch.sigmoid(visible_mask[0]).squeeze().cpu().numpy()
        
        # 마스크 경계에서 포인트 추출
        additional_points = self._extract_points_from_mask_boundary(amodal_np, visible_np)
        
        if len(additional_points) > 0:
            # 기존 포인트와 결합
            combined_points = np.concatenate([points, additional_points], axis=0)
            combined_labels = np.concatenate([
                labels, 
                np.ones(len(additional_points), dtype=int)  # 추가 포인트는 positive로 가정
            ], axis=0)
            
            print(f"      ✓ 추가 포인트: {len(additional_points)}개 (총 {len(combined_points)}개)")
            return combined_points, combined_labels
        else:
            print("      ⚠️ 추가 포인트 없음")
            return points, labels
    
    def _mask_attention_map(self, attention_map: np.ndarray, mask: torch.Tensor) -> np.ndarray:
        """마스크를 사용하여 attention map을 마스킹"""
        mask_np = torch.sigmoid(mask[0]).squeeze().cpu().numpy()
        
        # 마스크 영역 내의 attention만 유지
        masked_attention = attention_map.copy()
        masked_attention[mask_np < 0.5] *= 0.1  # 마스크 외부는 10%로 감소
        
        return masked_attention
    
    def _generate_points_from_masked_attention(self, masked_attention: np.ndarray, 
                                             current_points: np.ndarray, 
                                             current_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """마스크된 attention map에서 새로운 포인트 생성"""
        print("    🎯 마스크된 attention에서 포인트 재생성")
        
        # VL-SAM 방식으로 포인트 샘플링
        positive_points, negative_points, labels = self.point_sampler.sample_points_from_attention(
            masked_attention,
            bbox=None,  # 마스크된 attention이므로 bbox 불필요
            use_vlsam_method=True
        )
        
        if len(positive_points) > 0 and len(negative_points) > 0:
            all_points = np.concatenate([positive_points, negative_points], axis=0)
            all_labels = np.concatenate([
                np.ones(len(positive_points), dtype=int),
                np.zeros(len(negative_points), dtype=int)
            ], axis=0)
        elif len(positive_points) > 0:
            all_points = positive_points
            all_labels = np.ones(len(positive_points), dtype=int)
        elif len(negative_points) > 0:
            all_points = negative_points
            all_labels = np.zeros(len(negative_points), dtype=int)
        else:
            # Fallback: 기존 포인트 유지
            all_points = current_points
            all_labels = current_labels
        
        return all_points, all_labels
    
    def _extract_points_from_mask_boundary(self, amodal_mask: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        """마스크 경계에서 포인트 추출"""
        from scipy import ndimage
        
        # 마스크 경계 찾기
        amodal_boundary = self._find_mask_boundary(amodal_mask)
        visible_boundary = self._find_mask_boundary(visible_mask)
        
        # 경계 포인트들 수집
        boundary_points = []
        
        if np.sum(amodal_boundary) > 0:
            amodal_points = np.column_stack(np.where(amodal_boundary))
            # 샘플링 (최대 5개)
            if len(amodal_points) > 5:
                indices = np.random.choice(len(amodal_points), 5, replace=False)
                amodal_points = amodal_points[indices]
            boundary_points.extend(amodal_points)
        
        if np.sum(visible_boundary) > 0:
            visible_points = np.column_stack(np.where(visible_boundary))
            # 샘플링 (최대 3개)
            if len(visible_points) > 3:
                indices = np.random.choice(len(visible_points), 3, replace=False)
                visible_points = visible_points[indices]
            boundary_points.extend(visible_points)
        
        if len(boundary_points) > 0:
            return np.array(boundary_points)
        else:
            return np.array([])
    
    def _find_mask_boundary(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """마스크의 경계를 찾기"""
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # 경계 찾기 (morphological gradient)
        from scipy import ndimage
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = ndimage.binary_dilation(binary_mask, structure=kernel)
        eroded = ndimage.binary_erosion(binary_mask, structure=kernel)
        boundary = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        
        return boundary > 0
    
    def _calculate_mask_iou(self, pred_mask: torch.Tensor, gt_mask: np.ndarray = None) -> float:
        """마스크 IoU 계산 (GT가 없으면 기본값 반환)"""
        if gt_mask is None:
            # GT가 없으면 마스크의 평균값을 IoU로 사용
            return float(torch.sigmoid(pred_mask).mean().item())
        else:
            # 실제 IoU 계산
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            gt_tensor = torch.from_numpy(gt_mask).float().to(pred_mask.device)
            
            intersection = (pred_binary * gt_tensor).sum()
            union = pred_binary.sum() + gt_tensor.sum() - intersection
            
            if union > 0:
                return float(intersection / union)
            else:
                return 0.0
    
    def process_with_refinement(self, image_pil: Image.Image, target_class: str = None,
                              use_iterative_refinement: bool = True, use_multi_scale: bool = True,
                              num_iterations: int = 2, save_visualization: bool = True,
                              output_dir: str = "./outputs/d2sa_vlm_sam_250909") -> Dict:
        """
        VL-SAM 논문의 Iterative Refinement와 Multi-scale Ensemble을 포함한 완전한 파이프라인
        
        Args:
            image_pil: 입력 PIL 이미지
            target_class: 타겟 객체 클래스
            use_iterative_refinement: Iterative Refinement 사용 여부
            use_multi_scale: Multi-scale Ensemble 사용 여부
            num_iterations: 반복 횟수
            save_visualization: 시각화 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 처리 결과
        """
        print(f"🚀 VL-SAM 완전 파이프라인 시작")
        print(f"  - Iterative Refinement: {'활성화' if use_iterative_refinement else '비활성화'}")
        print(f"  - Multi-scale Ensemble: {'활성화' if use_multi_scale else '비활성화'}")
        print(f"  - 반복 횟수: {num_iterations}")
        
        # 출력 디렉토리 생성
        if save_visualization:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            if use_multi_scale:
                # Multi-scale Ensemble 사용
                print("🔍 Multi-scale Ensemble 모드")
                final_amodal, final_visible, ensemble_results = self.multi_scale_ensemble(
                    image_pil, target_class, num_iterations
                )
                
                # 결과 정리
                result = {
                    "processing_mode": "multi_scale_ensemble",
                    "amodal_mask": final_amodal,
                    "visible_mask": final_visible,
                    "ensemble_results": ensemble_results,
                    "iterative_refinement": use_iterative_refinement,
                    "multi_scale": use_multi_scale,
                    "num_iterations": num_iterations,
                    "status": "success"
                }
                
            else:
                # 단일 이미지 처리 (Iterative Refinement 포함)
                print("🖼️ 단일 이미지 처리 모드")
                
                # 이미지를 텐서로 변환
                transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                ])
                
                image_tensor = transform(image_pil).unsqueeze(0)
                
                # VLM 분석 및 attention map 추출
                occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = self.extract_occlusion_info(
                    image_pil, bbox=None, target_class=target_class, use_vlsam_method=True, auto_detect_bbox=True
                )
                
                if len(sam_points) == 0:
                    raise RuntimeError("No SAM points generated")
                
                if use_iterative_refinement:
                    # Iterative Refinement 적용
                    final_amodal, final_visible, refinement_results = self.iterative_refinement(
                        image_tensor, sam_points, sam_labels, aggregated_attention, num_iterations
                    )
                else:
                    # 기본 SAM 마스크 예측
                    final_amodal, final_visible = self._predict_sam_masks(image_tensor, sam_points, sam_labels)
                    refinement_results = []
                
                # 결과 정리
                result = {
                    "processing_mode": "single_image",
                    "amodal_mask": final_amodal,
                    "visible_mask": final_visible,
                    "occlusion_info": occlusion_info,
                    "attention_maps": attention_maps,
                    "aggregated_attention": aggregated_attention,
                    "sam_points": sam_points,
                    "sam_labels": sam_labels,
                    "refinement_results": refinement_results,
                    "iterative_refinement": use_iterative_refinement,
                    "multi_scale": use_multi_scale,
                    "num_iterations": num_iterations,
                    "status": "success"
                }
            
            # 시각화 저장
            if save_visualization:
                import hashlib
                image_hash = hashlib.md5(str(image_pil.size).encode()).hexdigest()[:8]
                viz_filename = f"vl_sam_refined_{image_hash}_{target_class or 'unknown'}.png"
                viz_path = os.path.join(output_dir, viz_filename)
                
                # 시각화 생성
                self._create_refinement_visualization(
                    image_pil, result, viz_path, target_class
                )
                
                result["visualization_path"] = viz_path
                print(f"🎨 시각화 저장: {viz_path}")
            
            print(f"✅ VL-SAM 완전 파이프라인 완료")
            return result
            
        except Exception as e:
            print(f"❌ VL-SAM 파이프라인 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "processing_mode": "error",
                "error": str(e),
                "error_details": traceback.format_exc(),
                "status": "error"
            }
    
    def _create_refinement_visualization(self, image_pil: Image.Image, result: Dict, 
                                       save_path: str, target_class: str = None):
        """Refinement 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'VL-SAM Refined Pipeline - {target_class or "Unknown Object"}', fontsize=16)
            
            # 원본 이미지
            axes[0, 0].imshow(image_pil)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Amodal 마스크
            if "amodal_mask" in result:
                amodal_np = torch.sigmoid(result["amodal_mask"][0]).squeeze().cpu().numpy()
                axes[0, 1].imshow(amodal_np, cmap='hot')
                axes[0, 1].set_title('Amodal Mask')
                axes[0, 1].axis('off')
            
            # Visible 마스크
            if "visible_mask" in result:
                visible_np = torch.sigmoid(result["visible_mask"][0]).squeeze().cpu().numpy()
                axes[0, 2].imshow(visible_np, cmap='hot')
                axes[0, 2].set_title('Visible Mask')
                axes[0, 2].axis('off')
            
            # Attention Map
            if "aggregated_attention" in result:
                attention_np = result["aggregated_attention"]
                axes[1, 0].imshow(attention_np, cmap='viridis')
                axes[1, 0].set_title('Aggregated Attention')
                axes[1, 0].axis('off')
            
            # SAM Points
            if "sam_points" in result and "sam_labels" in result:
                axes[1, 1].imshow(image_pil)
                points = result["sam_points"]
                labels = result["sam_labels"]
                
                if len(points) > 0:
                    pos_points = points[labels == 1]
                    neg_points = points[labels == 0]
                    
                    if len(pos_points) > 0:
                        axes[1, 1].scatter(pos_points[:, 0], pos_points[:, 1], 
                                          c='red', s=100, marker='+', label='Positive')
                    if len(neg_points) > 0:
                        axes[1, 1].scatter(neg_points[:, 0], neg_points[:, 1], 
                                          c='blue', s=100, marker='x', label='Negative')
                
                axes[1, 1].set_title('SAM Points')
                axes[1, 1].axis('off')
            
            # Refinement 결과
            if "refinement_results" in result and len(result["refinement_results"]) > 0:
                iterations = [r["iteration"] for r in result["refinement_results"]]
                amodal_ious = [r["amodal_iou"] for r in result["refinement_results"]]
                visible_ious = [r["visible_iou"] for r in result["refinement_results"]]
                
                axes[1, 2].plot(iterations, amodal_ious, 'r-o', label='Amodal IoU')
                axes[1, 2].plot(iterations, visible_ious, 'b-s', label='Visible IoU')
                axes[1, 2].set_xlabel('Iteration')
                axes[1, 2].set_ylabel('IoU')
                axes[1, 2].set_title('Refinement Progress')
                axes[1, 2].legend()
                axes[1, 2].grid(True)
            else:
                axes[1, 2].text(0.5, 0.5, 'No Refinement Data', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Refinement Progress')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 시각화 생성 실패: {e}")
    
    def process_all_objects_sequentially(self, image_pil: Image.Image, 
                                       save_visualization: bool = True,
                                       output_dir: str = "./outputs/sequential_objects") -> Dict:
        """
        VLM이 출력한 모든 객체(가려진 객체 포함)에 대해 순차적으로 segmentation 수행
        
        Args:
            image_pil: 입력 PIL 이미지
            save_visualization: 시각화 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 모든 객체에 대한 segmentation 결과
        """
        print(f"🔄 순차적 객체 처리 시작: VLM이 감지한 모든 객체에 대해 segmentation 수행")
        
        # 출력 디렉토리 생성
        if save_visualization:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. VLM을 사용한 전체 객체 분석
            print("🔍 VLM을 사용한 전체 객체 분석...")
            occlusion_info = self.occlusion_analyzer.analyze_occlusion(image_pil)
            
            # 모든 감지된 객체 수집 (가려진 객체 + 보이는 객체)
            all_objects = []
            all_objects.extend(occlusion_info.get('visible_objects', []))
            all_objects.extend(occlusion_info.get('occluded_objects', []))
            
            # 중복 제거
            unique_objects = list(set(all_objects))
            
            print(f"✓ VLM이 감지한 객체: {len(unique_objects)}개")
            print(f"  - 보이는 객체: {occlusion_info.get('visible_objects', [])}")
            print(f"  - 가려진 객체: {occlusion_info.get('occluded_objects', [])}")
            
            if len(unique_objects) == 0:
                print("⚠️ 감지된 객체가 없습니다.")
                return {
                    "status": "no_objects",
                    "message": "VLM이 감지한 객체가 없습니다.",
                    "occlusion_info": occlusion_info
                }
            
            # 2. 각 객체에 대해 순차적으로 segmentation 수행
            object_results = []
            all_masks = []
            
            for i, object_name in enumerate(unique_objects):
                print(f"\\n🎯 객체 {i+1}/{len(unique_objects)} 처리: '{object_name}'")
                
                try:
                    # 개별 객체에 대한 segmentation 수행
                    object_result = self._process_single_object(
                        image_pil, object_name, i, save_visualization, output_dir
                    )
                    
                    object_results.append(object_result)
                    
                    if object_result["status"] == "success":
                        all_masks.append({
                            "object_name": object_name,
                            "amodal_mask": object_result["amodal_mask"],
                            "visible_mask": object_result["visible_mask"],
                            "attention_map": object_result["attention_map"],
                            "sam_points": object_result["sam_points"]
                        })
                        print(f"  ✅ '{object_name}' segmentation 성공")
                    else:
                        print(f"  ❌ '{object_name}' segmentation 실패: {object_result.get('error', 'Unknown')}")
                
                except Exception as e:
                    print(f"  ❌ '{object_name}' 처리 중 오류: {e}")
                    object_results.append({
                        "object_name": object_name,
                        "status": "error",
                        "error": str(e)
                    })
            
            # 3. 전체 결과 정리
            successful_objects = [r for r in object_results if r["status"] == "success"]
            failed_objects = [r for r in object_results if r["status"] == "error"]
            
            print(f"\\n📊 순차적 객체 처리 완료!")
            print(f"  - 성공: {len(successful_objects)}개")
            print(f"  - 실패: {len(failed_objects)}개")
            print(f"  - 성공률: {len(successful_objects)/len(unique_objects)*100:.1f}%")
            
            # 4. 통합 시각화 생성
            if save_visualization and len(all_masks) > 0:
                self._create_sequential_visualization(
                    image_pil, all_masks, object_results, output_dir
                )
            
            # 5. 최종 결과 반환
            result = {
                "status": "success",
                "total_objects": len(unique_objects),
                "successful_objects": len(successful_objects),
                "failed_objects": len(failed_objects),
                "success_rate": len(successful_objects)/len(unique_objects)*100,
                "occlusion_info": occlusion_info,
                "object_results": object_results,
                "all_masks": all_masks,
                "processing_mode": "sequential_all_objects"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 순차적 객체 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "error_details": traceback.format_exc()
            }
    
    def _process_single_object(self, image_pil: Image.Image, object_name: str, 
                             object_index: int, save_visualization: bool, 
                             output_dir: str) -> Dict:
        """
        개별 객체에 대한 segmentation 수행
        
        Args:
            image_pil: 입력 이미지
            object_name: 처리할 객체 이름
            object_index: 객체 인덱스
            save_visualization: 시각화 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            Dict: 개별 객체 segmentation 결과
        """
        try:
            # 1. 객체별 attention map 추출
            print(f"    🧠 '{object_name}'에 대한 attention map 추출...")
            
            # 객체별 프롬프트 생성
            object_prompt = f"USER: <image>Focus on the {object_name} in this image. Describe its location, shape, and any visible parts.\nASSISTANT:"
            
            # VLM을 사용한 객체별 attention map 추출
            attention_maps, aggregated_attention = self.occlusion_analyzer.extract_attention_maps(
                image_pil, prompt=object_prompt, use_vlsam_method=True
            )
            
            # 2. SAM prompt 생성 (attention map 기반)
            print(f"    🎯 '{object_name}'에 대한 SAM prompt 생성...")
            
            sam_points, sam_labels = self.occlusion_analyzer.generate_sam_prompts(
                image_pil, attention_maps, aggregated_attention, 
                bbox=None, target_class=object_name, use_vlsam_method=True
            )
            
            if len(sam_points) == 0:
                return {
                    "object_name": object_name,
                    "object_index": object_index,
                    "status": "error",
                    "error": "No SAM points generated"
                }
            
            # 3. SAM을 사용한 마스크 예측
            print(f"    🎭 '{object_name}'에 대한 마스크 예측...")
            
            # 이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(image_pil).unsqueeze(0)
            
            # SAM 마스크 예측
            amodal_mask, visible_mask = self._predict_sam_masks(
                image_tensor, sam_points, sam_labels
            )
            
            # 4. 결과 정리
            result = {
                "object_name": object_name,
                "object_index": object_index,
                "status": "success",
                "amodal_mask": amodal_mask,
                "visible_mask": visible_mask,
                "attention_map": aggregated_attention,
                "attention_maps": attention_maps,
                "sam_points": sam_points,
                "sam_labels": sam_labels,
                "amodal_iou": self._calculate_mask_iou(amodal_mask, None),
                "visible_iou": self._calculate_mask_iou(visible_mask, None)
            }
            
            # 5. 개별 객체 시각화 (선택적)
            if save_visualization:
                self._create_single_object_visualization(
                    image_pil, result, output_dir
                )
            
            return result
            
        except Exception as e:
            return {
                "object_name": object_name,
                "object_index": object_index,
                "status": "error",
                "error": str(e)
            }
    
    def _create_sequential_visualization(self, image_pil: Image.Image, all_masks: List[Dict], 
                                       object_results: List[Dict], output_dir: str):
        """순차적 객체 처리 결과 통합 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            num_objects = len(all_masks)
            if num_objects == 0:
                return
            
            # 서브플롯 레이아웃 계산
            cols = min(4, num_objects + 1)  # +1 for original image
            rows = (num_objects + 1 + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Sequential Object Processing - {num_objects} Objects', fontsize=16)
            
            # 원본 이미지
            axes[0, 0].imshow(image_pil)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 각 객체별 결과
            for i, mask_data in enumerate(all_masks):
                row = (i + 1) // cols
                col = (i + 1) % cols
                
                if row < rows and col < cols:
                    # Amodal 마스크
                    amodal_np = torch.sigmoid(mask_data["amodal_mask"][0]).squeeze().cpu().numpy()
                    axes[row, col].imshow(amodal_np, cmap='hot')
                    axes[row, col].set_title(f'{mask_data["object_name"]}\\n(Amodal)')
                    axes[row, col].axis('off')
            
            # 빈 서브플롯 숨기기
            for i in range(num_objects + 1, rows * cols):
                row = i // cols
                col = i % cols
                if row < rows and col < cols:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # 저장
            viz_path = os.path.join(output_dir, "sequential_objects_overview.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"🎨 순차적 객체 처리 시각화 저장: {viz_path}")
            
        except Exception as e:
            print(f"❌ 순차적 시각화 생성 실패: {e}")
    
    def _create_single_object_visualization(self, image_pil: Image.Image, result: Dict, output_dir: str):
        """개별 객체 시각화"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Object: {result["object_name"]}', fontsize=14)
            
            # 원본 이미지
            axes[0].imshow(image_pil)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Amodal 마스크
            amodal_np = torch.sigmoid(result["amodal_mask"][0]).squeeze().cpu().numpy()
            axes[1].imshow(amodal_np, cmap='hot')
            axes[1].set_title('Amodal Mask')
            axes[1].axis('off')
            
            # Visible 마스크
            visible_np = torch.sigmoid(result["visible_mask"][0]).squeeze().cpu().numpy()
            axes[2].imshow(visible_np, cmap='hot')
            axes[2].set_title('Visible Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            safe_name = result["object_name"].replace(" ", "_").replace("/", "_")
            viz_path = os.path.join(output_dir, f"object_{result['object_index']:02d}_{safe_name}.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ 개별 객체 시각화 생성 실패: {e}")


# 사용 예시
if __name__ == '__main__':
    print("=== VLM-SAM 모델 with D2SA 데이터셋 테스트 시작 ===")
    
    # VLM-SAM 모델 초기화 (D2SA 데이터셋 포함)
    print("🔧 모델 초기화 중...")
    model = VLMSAMModel(use_d2sa=True)
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 모델 정보:")
    print(f"  - 전체 파라미터 수: {total_params:,}")
    print(f"  - 학습 가능한 파라미터 수: {trainable_params:,}")
    print(f"  - Zero-shot 모드: {'✓' if trainable_params == 0 else '✗'}")
    print(f"  - D2SA 데이터셋 사용: {'✓' if model.use_d2sa else '✗'}")
    
    if model.use_d2sa and model.d2sa_dataset:
        print(f"  - D2SA 샘플 수: {len(model.d2sa_dataset)}")
    
    # 방법 1: 단일 D2SA 샘플 처리
    print(f"\n=== 방법 1: 단일 D2SA 샘플 처리 ===")
    result = model.process_d2sa_sample(
        index=None,  # 랜덤 선택
        save_visualization=True,
        save_vlm_analysis=True,
        output_dir="./outputs/d2sa_vlm_sam_250909"
    )
    
    if result["status"] == "success":
        print(f"✅ 단일 샘플 처리 성공!")
        print(f"  - 카테고리: {result['sample_info']['category_name']}")
        print(f"  - Amodal IoU: {result['processing_results']['pred_amodal_iou']:.3f}")
        print(f"  - Visible IoU: {result['processing_results']['pred_visible_iou']:.3f}")
        print(f"  - VLM Points: {result['processing_results']['num_sam_points']}개")
        print(f"  - 시각화: {result['visualization_path']}")
        print(f"  - VLM 분석 JSON: {result['vlm_json_path']}")
    else:
        print(f"❌ 단일 샘플 처리 실패: {result.get('error', 'Unknown error')}")
    
    # 방법 2: 여러 D2SA 샘플 배치 처리
    print(f"\n=== 방법 2: 배치 D2SA 샘플 처리 ===")
    batch_results = model.process_multiple_d2sa_samples(
        num_samples=3,
        save_vlm_analysis=True,
        output_dir="./outputs/d2sa_vlm_sam_250909"
    )
    
    print(f"✅ 배치 처리 완료!")
    successful_results = [r for r in batch_results if r.get("status") == "success"]
    print(f"  - 성공한 샘플: {len(successful_results)}개")
    
    if successful_results:
        avg_amodal_iou = np.mean([r["processing_results"]["pred_amodal_iou"] for r in successful_results])
        avg_visible_iou = np.mean([r["processing_results"]["pred_visible_iou"] for r in successful_results])
        avg_points = np.mean([r["processing_results"]["num_sam_points"] for r in successful_results])
        
        print(f"  - 평균 Amodal IoU: {avg_amodal_iou:.3f}")
        print(f"  - 평균 Visible IoU: {avg_visible_iou:.3f}")
        print(f"  - 평균 VLM Points: {avg_points:.1f}개")
    
    # 방법 3: 순차적 객체 처리 (새로운 기능)
    print(f"\n=== 방법 3: 순차적 객체 처리 (VLM이 감지한 모든 객체) ===")
    
    # D2SA 샘플 이미지로 순차적 처리 테스트
    try:
        print("📸 D2SA 샘플 이미지로 순차적 객체 처리 테스트...")
        d2sa_image_tensor, d2sa_bbox_tensor, d2sa_pil_image, d2sa_category_name, d2sa_ann_info = model.get_d2sa_sample(index=1)
        
        print(f"  - 이미지 ID: {d2sa_ann_info['image_id']}")
        print(f"  - 카테고리: {d2sa_category_name}")
        print(f"  - 파일명: {d2sa_ann_info['file_name']}")
        print(f"  - 이미지 크기: {d2sa_pil_image.size}")
        
        # 순차적 객체 처리 실행
        sequential_result = model.process_all_objects_sequentially(
            image_pil=d2sa_pil_image,
            save_visualization=True,
            output_dir="./outputs/d2sa_vlm_sam_250909"
        )
        
        if sequential_result['status'] == 'success':
            print(f"✅ 순차적 객체 처리 성공!")
            print(f"  - 총 객체 수: {sequential_result['total_objects']}개")
            print(f"  - 성공한 객체: {sequential_result['successful_objects']}개")
            print(f"  - 실패한 객체: {sequential_result['failed_objects']}개")
            print(f"  - 성공률: {sequential_result['success_rate']:.1f}%")
            
            # VLM이 감지한 객체들 출력
            occlusion_info = sequential_result['occlusion_info']
            print(f"  - 보이는 객체: {occlusion_info.get('visible_objects', [])}")
            print(f"  - 가려진 객체: {occlusion_info.get('occluded_objects', [])}")
            
            print(f"  - 시각화 저장: ./outputs/sequential_d2sa_test/")
        else:
            print(f"❌ 순차적 객체 처리 실패: {sequential_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ 순차적 객체 처리 테스트 실패: {e}")
    
    # 방법 4: 기존 방식 (더미 데이터) - 호환성 유지
    print(f"\n=== 방법 4: 기존 더미 데이터 방식 (호환성 테스트) ===")
    dummy_image = torch.randn(1, 3, 1024, 1024)
    dummy_box = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    dummy_pil = Image.new('RGB', (1024, 1024), color='lightgray')
    
    print("🚀 더미 데이터로 추론 실행 중...")
    amodal_mask, amodal_iou, visible_mask, visible_iou, occlusion_info, attention_maps, aggregated_attention, sam_points, sam_labels = model(
        dummy_image, dummy_box, dummy_pil, "a test object"
    )
    
    print(f"✅ 더미 데이터 추론 완료!")
    print(f"  - Amodal 마스크 shape: {amodal_mask.shape}")
    print(f"  - Visible 마스크 shape: {visible_mask.shape}")
    print(f"  - Attention Maps: {len(attention_maps)}개 layer")
    print(f"  - SAM Points: {sam_points.shape}")
    print(f"  - Positive Points: {np.sum(sam_labels)}개")
    
    print(f"\n🎯 사용법 요약:")
    print(f"1. 단일 D2SA 샘플: model.process_d2sa_sample()")
    print(f"2. 배치 D2SA 샘플: model.process_multiple_d2sa_samples()")
    print(f"3. 순차적 객체 처리: model.process_all_objects_sequentially()")
    print(f"4. 특정 D2SA 샘플: model.get_d2sa_sample(index=N)")
    print(f"5. 기존 방식 호환: model(image, box, pil_image, text)")
    
    print("\n=== VLM-SAM 모델 with D2SA 테스트 완료 ===")
