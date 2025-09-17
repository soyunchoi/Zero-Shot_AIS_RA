'''''
Amodal / Visible decoder 따로 학습
'''''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# EfficientSAM 모듈 임포트
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

class BaselineModel(nn.Module):
    """
    모델 A (Baseline): Amodal과 Visible 마스크를 별도로 예측하는 모델.
    하나의 공유된 인코더와 두 개의 분리된 마스크 디코더(헤드)를 가집니다.
    태스크별 특화 학습과 일관성 제약을 포함합니다.
    """
    def __init__(self, training_stage='joint'):
        super().__init__()
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

        # 학습 단계 설정
        self.training_stage = training_stage
        self._setup_training_stage(training_stage)
        
        print("태스크별 특화 학습 설정 완료.")

    def _create_amodal_decoder(self):
        """Amodal 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Amodal 특화: 더 넓은 영역을 예측하도록 초기화
        # 마스크 임베딩 레이어의 가중치를 약간 확장
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                # 마스크 관련 레이어의 가중치를 약간 확장
                param.data *= 1.1  # 10% 확장
                print(f"Amodal 디코더 {name} 가중치 확장 적용")
        
        return decoder
    
    def _create_visible_decoder(self):
        """Visible 마스크에 특화된 디코더"""
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # Visible 특화: 더 정확한 경계를 예측하도록 초기화
        # 마스크 임베딩 레이어의 가중치를 약간 축소
        for name, param in decoder.named_parameters():
            if 'mask_tokens' in name or 'output_upscaling' in name:
                # 마스크 관련 레이어의 가중치를 약간 축소
                param.data *= 0.9  # 10% 축소
                print(f"Visible 디코더 {name} 가중치 축소 적용")
        
        return decoder

    def _setup_training_stage(self, stage):
        """학습 단계별 파라미터 설정"""
        if stage == 'visible_only':
            # 1단계: Visible 디코더만 학습
            self._freeze_encoder()
            self._freeze_decoder(self.amodal_mask_decoder)
            self._unfreeze_decoder(self.visible_mask_decoder)
            print("1단계: Visible 디코더만 학습 (Amodal 고정)")
            
        elif stage == 'amodal_only':
            # 2단계: Amodal 디코더만 학습 (visible 고정)
            self._freeze_encoder()
            self._freeze_decoder(self.visible_mask_decoder)
            self._unfreeze_decoder(self.amodal_mask_decoder)
            print("2단계: Amodal 디코더만 학습 (Visible 고정)")
            
        elif stage == 'joint':
            # 3단계: 전체 미세조정
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
        
        # 프롬프트 인코더는 학습 가능하도록 설정
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
        # amodal_mask >= visible_mask 여야 함
        # sigmoid 적용 후 계산
        amodal_probs = torch.sigmoid(amodal_mask)
        visible_probs = torch.sigmoid(visible_mask)
        
        # amodal이 visible보다 작은 경우에만 페널티
        violation = torch.clamp(visible_probs - amodal_probs, min=0)
        consistency_loss = torch.mean(violation)
        
        return consistency_loss

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
        # 바운딩 박스를 EfficientSAM 형태의 포인트로 변환
        # EfficientSAM 예제에서 사용하는 형태: [[[x1, y1], [x2, y2]]]
        points = torch.stack([
            box[:, :2],  # top-left (x1, y1)
            box[:, 2:]   # bottom-right (x2, y2)
        ], dim=1)  # shape: (B, 2, 2)
        
        # EfficientSAM 형태로 변환: (B, 1, 2, 2)
        points = points.unsqueeze(1)
        
        # 라벨: corner points를 위한 라벨
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
            
            # --- Amodal 마스크 예측 (태스크별 특화 디코더 사용) ---
            amodal_logits, amodal_iou = self.amodal_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # --- Visible 마스크 예측 (태스크별 특화 디코더 사용) ---
            visible_logits, visible_iou = self.visible_mask_decoder(
                image_embeddings=image_embeddings,
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
                amodal_mask = torch.nn.functional.interpolate(
                    amodal_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )
            if visible_mask.shape[-2:] != (img_h, img_w):
                visible_mask = torch.nn.functional.interpolate(
                    visible_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                )

            return amodal_mask, amodal_iou_best, visible_mask, visible_iou_best

        except Exception as e:
            print(f"모델 추론 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 빈 마스크 반환
            empty_mask = torch.zeros(batch_size, 1, img_h, img_w, device=image.device)
            empty_iou = torch.zeros(batch_size, 1, device=image.device)
            return empty_mask, empty_iou, empty_mask.clone(), empty_iou.clone()


# 사용 예시
if __name__ == '__main__':
    # 더미 입력 생성
    dummy_image = torch.randn(1, 3, 1024, 1024)
    dummy_box = torch.tensor([[100.0, 200.0, 300.0, 400.0]]) # (x1, y1, x2, y2)

    # 모델 초기화 (기본: joint 학습)
    model_a = BaselineModel(training_stage='joint')

    # 순전파 실행
    amodal_mask, amodal_iou, visible_mask, visible_iou = model_a(dummy_image, dummy_box)

    print(f"""
모델 A (Baseline)이 성공적으로 초기화되었습니다.""")
    print(f"입력 이미지 shape: {dummy_image.shape}")
    print(f"입력 박스 shape: {dummy_box.shape}")
    print(f"Amodal 마스크 shape: {amodal_mask.shape}")
    print(f"Amodal IoU shape: {amodal_iou.shape}")
    print(f"Visible 마스크 shape: {visible_mask.shape}")
    print(f"Visible IoU shape: {visible_iou.shape}")

    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_a.parameters())
    print(f"학습 가능한 파라미터 수: {trainable_params}")
    print(f"전체 파라미터 수: {total_params}")

    # 학습 단계별 테스트
    print("\n=== 학습 단계별 테스트 ===")
    
    # 1단계: Visible만 학습
    model_a.set_training_stage('visible_only')
    visible_trainable = sum(p.numel() for p in model_a.visible_mask_decoder.parameters() if p.requires_grad)
    amodal_trainable = sum(p.numel() for p in model_a.amodal_mask_decoder.parameters() if p.requires_grad)
    print(f"1단계 - Visible 학습 가능: {visible_trainable}, Amodal 학습 가능: {amodal_trainable}")
    
    # 2단계: Amodal만 학습
    model_a.set_training_stage('amodal_only')
    visible_trainable = sum(p.numel() for p in model_a.visible_mask_decoder.parameters() if p.requires_grad)
    amodal_trainable = sum(p.numel() for p in model_a.amodal_mask_decoder.parameters() if p.requires_grad)
    print(f"2단계 - Visible 학습 가능: {visible_trainable}, Amodal 학습 가능: {amodal_trainable}")
    
    # 3단계: 전체 학습
    model_a.set_training_stage('joint')
    visible_trainable = sum(p.numel() for p in model_a.visible_mask_decoder.parameters() if p.requires_grad)
    amodal_trainable = sum(p.numel() for p in model_a.amodal_mask_decoder.parameters() if p.requires_grad)
    print(f"3단계 - Visible 학습 가능: {visible_trainable}, Amodal 학습 가능: {amodal_trainable}")

