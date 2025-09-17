'''''
Amodal / Visible decoder 따로 학습
Naive 버전 (Retrieval 없음)
'''''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# EfficientSAM 모듈 임포트
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

class BaselineModelNaive(nn.Module):
    """
    모델 A (Baseline): Amodal과 Visible 마스크를 별도로 예측하는 모델.
    Retrieval 없이 기본 구조만 사용.
    """
    def __init__(self, training_stage='joint'):
        super().__init__()
        # EfficientSAM 모델 로드
        self.efficient_sam = build_efficient_sam_vitt().eval()

        # --- 인코더와 디코더 분리 ---
        self.image_encoder = self.efficient_sam.image_encoder
        self.prompt_encoder = self.efficient_sam.prompt_encoder
        
        # 태스크별 특화된 디코더 생성
        self.amodal_mask_decoder = self._create_amodal_decoder()
        self.visible_mask_decoder = self._create_visible_decoder()

        # 학습 단계 설정
        self.training_stage = training_stage
        self._setup_training_stage(training_stage)

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

    def _setup_training_stage(self, stage):
        """학습 단계별 파라미터 설정"""
        if stage == 'visible_only':
            self._freeze_encoder()
            self._freeze_decoder(self.amodal_mask_decoder)
            self._unfreeze_decoder(self.visible_mask_decoder)
            
        elif stage == 'amodal_only':
            self._freeze_encoder()
            self._freeze_decoder(self.visible_mask_decoder)
            self._unfreeze_decoder(self.amodal_mask_decoder)
            
        elif stage == 'joint':
            self._freeze_encoder()
            self._unfreeze_decoder(self.visible_mask_decoder)
            self._unfreeze_decoder(self.amodal_mask_decoder)
            
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
