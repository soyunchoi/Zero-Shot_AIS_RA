'''''
LoRA 어댑터 기반 경량 미세 조정
- visible_mask_decoder와 amodal_mask_decoder 모두 LoRA 어댑터 적용
- 전체 파라미터의 1% 미만만 학습하여 과적합 방지
- Text RA + Cross-Attention 제거 버전
- 임베딩 공간 OT (Optimal Transport) 손실 함수 추가
  - 픽셀 공간 (N×N) 대신 임베딩 공간 (Q×Q)에서 OT 계산 → 계산 효율성 극대화
  - Visible과 Amodal 디코더의 쿼리 임베딩 간 구조적 매칭
  - 시맨틱 정보를 포함한 의미적 유사성 측정
  - Q×Q 매트릭스 (예: 4×4)로 계산하여 메모리 및 계산 비용 최소화
'''''

import torch
import torch.nn as nn
import copy

# EfficientSAM 모듈 임포트
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

# LoRA 어댑터 임포트
from lora_adapter import LoRALinear, LoRAMultiHeadAttention, apply_lora_to_linear_layers, count_parameters

# OT (Optimal Transport) 손실 함수를 위한 라이브러리
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("경고: geomloss가 설치되지 않았습니다. OT 손실 함수를 사용할 수 없습니다.")
    print("설치 방법: pip install geomloss")

class BaselineModel(nn.Module):
    """
    LoRA 어댑터 기반 모델 (Text RA + Cross-Attention 제거):
    - visible_mask_decoder와 amodal_mask_decoder 모두 LoRA 어댑터 적용
    - 인코더는 동결하여 기존 SAM 지식 보존
    - Joint 학습으로 Visible과 Amodal 동시 학습
    """
    def __init__(self, lora_rank=16, lora_alpha=16.0):
        super().__init__()
        # EfficientSAM 모델 로드
        self.efficient_sam = build_efficient_sam_vitt().eval()
        print("EfficientSAM (ViT-Tiny) 모델 로드 완료.")

        # --- 인코더와 디코더 분리 ---
        self.image_encoder = self.efficient_sam.image_encoder
        self.prompt_encoder = self.efficient_sam.prompt_encoder
        
        # LoRA 설정
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 디코더 생성
        self.visible_mask_decoder = self._create_visible_decoder_with_lora()  # LoRA 적용 디코더
        self.amodal_mask_decoder = self._create_amodal_decoder_with_lora()  # LoRA 적용 디코더
        print("LoRA 기반 Visible 및 Amodal 디코더 생성 완료.")

        # Joint 학습 설정
        self._setup_training_stage()
        
        # OT 손실 함수 초기화 (한 번만 생성하여 재사용)
        if GEOMLOSS_AVAILABLE:
            self.ot_loss_fn = SamplesLoss(
                loss="sinkhorn",  # Sinkhorn 알고리즘 사용
                p=2,  # L2 거리 사용
                blur=0.05,  # 기본 엔트로피 정규화 (나중에 파라미터로 조정 가능)
                reach=None,  # 무한대 도달
                scaling=0.9,  # 스케일링 파라미터 (안정성)
                debias=False,  # 편향 제거 비활성화 (더 빠름)
                potentials=False,  # 포텐셜 반환 비활성화
                backend="tensorized",  # keops 대신 tensorized 백엔드 사용 (호환성 문제 해결)
                verbose=False
            )
        else:
            self.ot_loss_fn = None
        
        # 파라미터 수 출력
        self._print_parameter_counts()
        
        print("LoRA 어댑터 기반 학습 설정 완료 (Text RA + Cross-Attention 없음, OT 손실 포함).")

    def _create_amodal_decoder_with_lora(self):
        """LoRA 어댑터가 적용된 Amodal 마스크 디코더"""
        # 기존 디코더 복사
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # 모든 Linear 레이어를 LoRA로 교체
        decoder = apply_lora_to_linear_layers(
            decoder, 
            rank=self.lora_rank, 
            alpha=self.lora_alpha
        )
        
        print(f"LoRA 어댑터 적용 완료 (rank={self.lora_rank}, alpha={self.lora_alpha})")
        return decoder
    
    def _create_visible_decoder_with_lora(self):
        """LoRA 어댑터가 적용된 Visible 마스크 디코더"""
        # 기존 디코더 복사
        decoder = copy.deepcopy(self.efficient_sam.mask_decoder)
        
        # 모든 Linear 레이어를 LoRA로 교체
        decoder = apply_lora_to_linear_layers(
            decoder, 
            rank=self.lora_rank, 
            alpha=self.lora_alpha
        )
        
        print(f"Visible 디코더에 LoRA 어댑터 적용 완료 (rank={self.lora_rank}, alpha={self.lora_alpha})")
        return decoder

    def _setup_training_stage(self):
        """Joint 학습 전략 - Visible과 Amodal LoRA 모두 학습"""
        # 인코더는 동결 (기존 SAM 지식 보존)
        self._freeze_encoder()
        
        # Visible과 Amodal LoRA 모두 학습
        self._unfreeze_lora_decoder(self.visible_mask_decoder)
        self._unfreeze_lora_decoder(self.amodal_mask_decoder)
        print("Joint 학습: Visible + Amodal LoRA")

    def _freeze_encoder(self):
        """이미지 인코더 동결, 프롬프트 인코더는 학습 가능"""
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
    
    def _unfreeze_lora_decoder(self, decoder):
        """LoRA 디코더의 LoRA 파라미터만 학습 가능하도록 설정"""
        for name, param in decoder.named_parameters():
            if 'lora_' in name:  # LoRA 파라미터만 학습
                param.requires_grad = True
            else:  # 기존 파라미터는 동결
                param.requires_grad = False
    
    def _print_parameter_counts(self):
        """파라미터 수 출력"""
        print("\n=== 파라미터 수 분석 ===")
        
        # 전체 모델 파라미터
        total_stats = count_parameters(self, trainable_only=False)
        print(f"전체 파라미터: {total_stats['total_params']:,}")
        print(f"학습 가능한 파라미터: {total_stats['trainable_params']:,}")
        print(f"동결된 파라미터: {total_stats['frozen_params']:,}")
        print(f"학습 가능한 비율: {total_stats['trainable_ratio']:.2%}")
        
        # Prompt Encoder 파라미터
        prompt_stats = count_parameters(self.prompt_encoder, trainable_only=False)
        print(f"\nPrompt Encoder:")
        print(f"  전체: {prompt_stats['total_params']:,}")
        print(f"  학습 가능: {prompt_stats['trainable_params']:,}")
        print(f"  학습 비율: {prompt_stats['trainable_ratio']:.2%}")
        
        # Amodal 디코더 파라미터 (LoRA)
        amodal_stats = count_parameters(self.amodal_mask_decoder, trainable_only=False)
        print(f"\nAmodal 디코더 (LoRA):")
        print(f"  전체: {amodal_stats['total_params']:,}")
        print(f"  학습 가능: {amodal_stats['trainable_params']:,}")
        print(f"  학습 비율: {amodal_stats['trainable_ratio']:.2%}")
        
        # Visible 디코더 파라미터 (LoRA)
        visible_stats = count_parameters(self.visible_mask_decoder, trainable_only=False)
        print(f"\nVisible 디코더 (LoRA):")
        print(f"  전체: {visible_stats['total_params']:,}")
        print(f"  학습 가능: {visible_stats['trainable_params']:,}")
        print(f"  학습 비율: {visible_stats['trainable_ratio']:.2%}")
        
        print("=" * 50)

    def compute_amodal_visible_consistency_loss(self, amodal_mask, visible_mask):
        """
        Joint 학습 시 Amodal과 Visible 예측 간의 일관성 제약
        - Amodal이 Visible을 포함해야 함
        - 두 마스크의 IoU가 높아야 함
        - Amodal이 과도하게 확장되지 않도록 제약
        """
        # sigmoid 적용
        amodal_probs = torch.sigmoid(amodal_mask)
        visible_probs = torch.sigmoid(visible_mask)
        
        # 1. Amodal이 Visible을 포함해야 함 (amodal >= visible)
        inclusion_loss = torch.mean(torch.clamp(visible_probs - amodal_probs, min=0))
        
        # 2. 두 마스크의 IoU가 높아야 함 (겹치는 영역이 많아야 함)
        intersection = torch.sum(amodal_probs * visible_probs, dim=[2, 3])
        union = torch.sum(amodal_probs + visible_probs - amodal_probs * visible_probs, dim=[2, 3])
        iou = intersection / (union + 1e-8)
        iou_loss = 1.0 - torch.mean(iou)
        
        # 3. Amodal이 너무 크지 않도록 제약 (과도한 확장 방지)
        amodal_area = torch.mean(amodal_probs)
        visible_area = torch.mean(visible_probs)
        expansion_penalty = torch.clamp(amodal_area - visible_area * 1.5, min=0)  # 50% 확장까지 허용
        
        # 가중합
        total_consistency_loss = inclusion_loss + 0.5 * iou_loss + 0.1 * expansion_penalty
        
        return total_consistency_loss, {
            'inclusion_loss': inclusion_loss.item(),
            'iou_loss': iou_loss.item(),
            'expansion_penalty': expansion_penalty.item()
        }
    
    def compute_optimal_transport_loss_embedding(self, visible_embeddings, amodal_embeddings, sinkhorn_blur=0.05, distance_type='l2'):
        """
        임베딩 공간에서의 OT (Optimal Transport) 기반 손실 함수
        - Visible과 Amodal 디코더의 쿼리 임베딩 간의 구조적 매칭을 측정
        - Q×Q 매트릭스로 계산하여 픽셀 공간 대비 계산 효율성 극대화
        - 시맨틱 정보를 포함한 의미적 유사성 측정
        
        Args:
            visible_embeddings (torch.Tensor): Visible 디코더의 쿼리 임베딩. Shape: (B, Q, D)
            amodal_embeddings (torch.Tensor): Amodal 디코더의 쿼리 임베딩. Shape: (B, Q, D)
            sinkhorn_blur (float): Sinkhorn 알고리즘의 엔트로피 정규화 계수
            distance_type (str): 거리 계산 방식 ('l2', 'cosine', 'l1')
        
        Returns:
            torch.Tensor: OT 손실 값 (스칼라)
        """
        if not GEOMLOSS_AVAILABLE:
            return torch.tensor(0.0, device=visible_embeddings.device, requires_grad=True)
        
        batch_size = visible_embeddings.shape[0]
        Q = visible_embeddings.shape[1]  # 쿼리 수 (예: 4)
        
        # 배치별로 OT 손실 계산
        ot_losses = []
        
        for b in range(batch_size):
            vis_emb = visible_embeddings[b]  # (Q, D)
            amodal_emb = amodal_embeddings[b]  # (Q, D)
            
            # 거리 매트릭스 계산
            if distance_type == 'l2':
                # L2 거리: ||x - y||^2
                distances = torch.cdist(vis_emb, amodal_emb, p=2) ** 2  # (Q, Q)
            elif distance_type == 'l1':
                # L1 거리: ||x - y||_1
                distances = torch.cdist(vis_emb, amodal_emb, p=1)  # (Q, Q)
            elif distance_type == 'cosine':
                # Cosine 거리: 1 - cosine_similarity
                vis_norm = torch.nn.functional.normalize(vis_emb, p=2, dim=1)
                amodal_norm = torch.nn.functional.normalize(amodal_emb, p=2, dim=1)
                cosine_sim = torch.mm(vis_norm, amodal_norm.t())  # (Q, Q)
                distances = 1 - cosine_sim  # (Q, Q)
            else:
                raise ValueError(f"지원하지 않는 거리 타입: {distance_type}")
            
            # 균일한 가중치 (각 쿼리가 동일한 중요도)
            vis_weights = torch.ones(Q, device=vis_emb.device) / Q  # (Q,)
            amodal_weights = torch.ones(Q, device=amodal_emb.device) / Q  # (Q,)
            
            # SamplesLoss를 사용한 Sinkhorn-Knopp 알고리즘
            if sinkhorn_blur != 0.05 or self.ot_loss_fn is None:
                ot_loss_fn = SamplesLoss(
                    loss="sinkhorn",
                    p=2,
                    blur=sinkhorn_blur,
                    reach=None,
                    scaling=0.9,
                    debias=False,
                    potentials=False,
                    backend="tensorized",
                    verbose=False
                )
            else:
                ot_loss_fn = self.ot_loss_fn
            
            # 배치 차원 추가: (1, Q) 형태로 변환
            vis_weights_batch = vis_weights.unsqueeze(0)  # (1, Q)
            amodal_weights_batch = amodal_weights.unsqueeze(0)  # (1, Q)
            vis_emb_batch = vis_emb.unsqueeze(0)  # (1, Q, D)
            amodal_emb_batch = amodal_emb.unsqueeze(0)  # (1, Q, D)
            
            # OT 손실 계산
            ot_loss = ot_loss_fn(vis_weights_batch, vis_emb_batch, amodal_weights_batch, amodal_emb_batch)
            
            # 스칼라로 변환
            if ot_loss.dim() > 0:
                ot_loss = ot_loss.squeeze()
            if ot_loss.dim() > 0:
                ot_loss = ot_loss.flatten()[0]
            
            ot_losses.append(ot_loss)
        
        # 배치 평균 반환
        if len(ot_losses) == 0:
            return torch.tensor(0.0, device=visible_embeddings.device, requires_grad=True)
        
        # 모든 손실을 스칼라 텐서로 변환하여 스택
        ot_losses_scalar = []
        for loss in ot_losses:
            if loss.dim() == 0:
                ot_losses_scalar.append(loss)
            else:
                loss_scalar = loss.squeeze() if loss.dim() > 0 else loss
                if loss_scalar.dim() > 0:
                    loss_scalar = loss_scalar.flatten()[0]
                ot_losses_scalar.append(loss_scalar)
        
        stacked = torch.stack(ot_losses_scalar)
        return stacked.mean()
    
    def compute_optimal_transport_loss(self, pred_mask_logits, gt_mask, sinkhorn_blur=0.05, downscale_factor=8, max_resolution=128):
        """
        [DEPRECATED] 픽셀 공간에서의 OT 손실 함수 (이전 버전)
        이 함수는 임베딩 공간 OT Loss로 대체되었습니다.
        호환성을 위해 유지하지만, compute_optimal_transport_loss_embedding 사용을 권장합니다.
        """
        if not GEOMLOSS_AVAILABLE:
            # geomloss가 없으면 0 반환 (경고는 이미 출력됨)
            return torch.tensor(0.0, device=pred_mask_logits.device, requires_grad=True)
        
        batch_size = pred_mask_logits.shape[0]
        
        # sigmoid 적용하여 확률 분포로 변환
        pred_probs = torch.sigmoid(pred_mask_logits)  # (B, 1, H, W)
        gt_probs = gt_mask  # (B, 1, H, W) - 이미 [0, 1] 범위
        
        # 배치별로 OT 손실 계산
        ot_losses = []
        
        for b in range(batch_size):
            pred_mask = pred_probs[b, 0]  # (H, W)
            gt_mask_b = gt_probs[b, 0]  # (H, W)
            
            # 다운스케일링 (계산 효율성 및 메모리 절약)
            H, W = pred_mask.shape
            
            # 먼저 downscale_factor로 다운스케일
            if downscale_factor > 1:
                new_H, new_W = H // downscale_factor, W // downscale_factor
                if new_H > 0 and new_W > 0:
                    pred_mask = torch.nn.functional.interpolate(
                        pred_mask.unsqueeze(0).unsqueeze(0),
                        size=(new_H, new_W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    gt_mask_b = torch.nn.functional.interpolate(
                        gt_mask_b.unsqueeze(0).unsqueeze(0),
                        size=(new_H, new_W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    H, W = new_H, new_W
                else:
                    # 다운스케일이 너무 크면 스킵
                    continue
            
            # 최대 해상도 제한 (메모리 절약)
            if max_resolution > 0 and (H > max_resolution or W > max_resolution):
                scale = min(max_resolution / H, max_resolution / W)
                new_H = max(int(H * scale), 1)
                new_W = max(int(W * scale), 1)
                pred_mask = torch.nn.functional.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0),
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                gt_mask_b = torch.nn.functional.interpolate(
                    gt_mask_b.unsqueeze(0).unsqueeze(0),
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                H, W = new_H, new_W
            
            H, W = pred_mask.shape
            
            # 픽셀 좌표 생성 (2D 그리드) - 0~1 범위로 정규화
            # 정규화하지 않으면 거리가 너무 커서 OT Loss가 비정상적으로 큼 (예: 200~260)
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=pred_mask.device, dtype=torch.float32),
                torch.arange(W, device=pred_mask.device, dtype=torch.float32),
                indexing='ij'
            )
            # 좌표를 0~1 범위로 정규화 (OT Loss 값이 적절한 범위로 줄어듦)
            x_coords_norm = x_coords / (W - 1) if W > 1 else x_coords
            y_coords_norm = y_coords / (H - 1) if H > 1 else y_coords
            coords = torch.stack([x_coords_norm.flatten(), y_coords_norm.flatten()], dim=1)  # (H*W, 2)
            
            # 확률 분포로 정규화 (총합이 1이 되도록)
            pred_flat = pred_mask.flatten()  # (H*W,)
            gt_flat = gt_mask_b.flatten()  # (H*W,)
            
            # 총합이 0인 경우 처리 (빈 마스크)
            pred_sum = pred_flat.sum()
            gt_sum = gt_flat.sum()
            
            if pred_sum < 1e-8 or gt_sum < 1e-8:
                # 빈 마스크인 경우 0 반환 (스칼라 텐서로 통일)
                ot_losses.append(torch.tensor(0.0, device=pred_mask.device, requires_grad=True).squeeze())
                continue
            
            # 확률 분포로 정규화
            pred_dist = pred_flat / pred_sum  # (H*W,)
            gt_dist = gt_flat / gt_sum  # (H*W,)
            
            # 배치 차원 추가: (1, H*W) 형태로 변환
            pred_dist_batch = pred_dist.unsqueeze(0)  # (1, H*W)
            gt_dist_batch = gt_dist.unsqueeze(0)  # (1, H*W)
            coords_batch = coords.unsqueeze(0)  # (1, H*W, 2)
            
            # OT 손실 계산 (SamplesLoss는 (weights, positions) 형태로 입력받음)
            # blur 파라미터를 동적으로 조정하려면 새로운 SamplesLoss 객체 생성 필요
            if sinkhorn_blur != 0.05 or self.ot_loss_fn is None:
                # blur 파라미터가 기본값과 다르면 새로운 객체 생성
                ot_loss_fn = SamplesLoss(
                    loss="sinkhorn",
                    p=2,
                    blur=sinkhorn_blur,
                    reach=None,
                    scaling=0.9,
                    debias=False,
                    potentials=False,
                    backend="tensorized",  # keops 대신 tensorized 백엔드 사용 (호환성 문제 해결)
                    verbose=False
                )
            else:
                ot_loss_fn = self.ot_loss_fn
            
            ot_loss = ot_loss_fn(pred_dist_batch, coords_batch, gt_dist_batch, coords_batch)
            
            # OT 손실을 스칼라 텐서로 변환 (shape 통일)
            # SamplesLoss는 보통 스칼라를 반환하지만, 안전하게 처리
            if ot_loss.dim() > 0:
                ot_loss = ot_loss.squeeze()
            if ot_loss.dim() > 0:
                # 여전히 텐서인 경우 첫 번째 요소 사용
                ot_loss = ot_loss.flatten()[0]
            
            ot_losses.append(ot_loss)
        
        # 배치 평균 반환
        if len(ot_losses) == 0:
            return torch.tensor(0.0, device=pred_mask_logits.device, requires_grad=True)
        
        # 모든 손실을 스칼라 텐서로 변환하여 스택 (shape 통일 보장)
        ot_losses_scalar = []
        for loss in ot_losses:
            # 스칼라로 변환
            if loss.dim() == 0:
                # 이미 스칼라
                ot_losses_scalar.append(loss)
            else:
                # 텐서인 경우 스칼라로 변환
                loss_scalar = loss.squeeze() if loss.dim() > 0 else loss
                if loss_scalar.dim() > 0:
                    loss_scalar = loss_scalar.flatten()[0]
                ot_losses_scalar.append(loss_scalar)
        
        # 스칼라 텐서들을 스택
        stacked = torch.stack(ot_losses_scalar)
        return stacked.mean()

    def forward(self, image: torch.Tensor, box: torch.Tensor, return_embeddings=False):
        """
        Args:
            image (torch.Tensor): 입력 이미지. Shape: (B, 3, H, W)
            box (torch.Tensor): 입력 바운딩 박스. Shape: (B, 4) (x1, y1, x2, y2)
            return_embeddings (bool): 임베딩도 함께 반환할지 여부

        Returns:
            Tuple[torch.Tensor, ...]: 
                - amodal_mask (B, 1, H, W)
                - amodal_iou (B, 1)
                - visible_mask (B, 1, H, W)
                - visible_iou (B, 1)
                - (optional) amodal_embeddings (B, Q, D): Amodal 디코더의 쿼리 임베딩
                - (optional) visible_embeddings (B, Q, D): Visible 디코더의 쿼리 임베딩
        """
        batch_size, _, img_h, img_w = image.shape

        # --- 입력 전처리 ---
        # 바운딩 박스를 EfficientSAM 형태의 포인트로 변환
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
            
            # 프롬프트 임베딩 가져오기
            rescaled_points = self.efficient_sam.get_rescaled_pts(points, img_h, img_w)
            sparse_embeddings = self.prompt_encoder(
                rescaled_points.reshape(batch_size * 1, 2, 2),
                labels.reshape(batch_size * 1, 2),
            )
            sparse_embeddings = sparse_embeddings.view(batch_size, 1, sparse_embeddings.shape[1], sparse_embeddings.shape[2])
            
            # --- Amodal 마스크 예측 및 임베딩 추출 ---
            amodal_logits, amodal_iou = self.amodal_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # Amodal 디코더의 쿼리 임베딩 추출 (transformer 출력)
            amodal_embeddings = self._extract_decoder_embeddings(
                self.amodal_mask_decoder,
                image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
            )
            
            # --- Visible 마스크 예측 및 임베딩 추출 ---
            visible_logits, visible_iou = self.visible_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                multimask_output=True,
            )
            
            # Visible 디코더의 쿼리 임베딩 추출 (transformer 출력)
            visible_embeddings = self._extract_decoder_embeddings(
                self.visible_mask_decoder,
                image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
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

            if return_embeddings:
                return amodal_mask, amodal_iou_best, visible_mask, visible_iou_best, amodal_embeddings, visible_embeddings
            else:
                return amodal_mask, amodal_iou_best, visible_mask, visible_iou_best

        except Exception as e:
            print(f"모델 추론 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 빈 마스크 반환
            empty_mask = torch.zeros(batch_size, 1, img_h, img_w, device=image.device)
            empty_iou = torch.zeros(batch_size, 1, device=image.device)
            if return_embeddings:
                # 임베딩도 빈 텐서로 반환
                empty_emb = torch.zeros(batch_size, 1, self.amodal_mask_decoder.transformer_dim, device=image.device)
                return empty_mask, empty_iou, empty_mask.clone(), empty_iou.clone(), empty_emb, empty_emb.clone()
            else:
                return empty_mask, empty_iou, empty_mask.clone(), empty_iou.clone()
    
    def _extract_decoder_embeddings(self, decoder, image_embeddings, image_pe, sparse_prompt_embeddings):
        """
        디코더의 transformer 출력에서 쿼리 임베딩 추출
        
        Returns:
            torch.Tensor: 쿼리 임베딩. Shape: (B, Q, D) where Q는 쿼리 수, D는 임베딩 차원
        """
        batch_size, max_num_queries, _, _ = sparse_prompt_embeddings.shape
        
        # 이미지 임베딩 타일링
        image_embeddings_tiled = torch.tile(
            image_embeddings[:, None, :, :, :], [1, max_num_queries, 1, 1, 1]
        ).view(
            batch_size * max_num_queries,
            image_embeddings.shape[1],
            image_embeddings.shape[2],
            image_embeddings.shape[3],
        )
        sparse_prompt_embeddings_reshaped = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_prompt_embeddings.shape[2], sparse_prompt_embeddings.shape[3]
        )
        
        # Transformer를 통과하여 임베딩 추출
        output_tokens = torch.cat(
            [decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings_reshaped.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings_reshaped), dim=1)
        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings_tiled.shape
        hs, src = decoder.transformer(image_embeddings_tiled, pos_src, tokens)
        
        # mask_tokens_out 추출 (iou_token 제외한 쿼리 임베딩)
        mask_tokens_out = hs[:, 1 : (1 + decoder.num_mask_tokens), :]  # (B*Q, num_mask_tokens, D)
        
        # 배치 차원 복원
        mask_tokens_out = mask_tokens_out.view(batch_size, max_num_queries, decoder.num_mask_tokens, -1)
        
        # multimask_output=True일 때 첫 번째 쿼리만 반환하거나 모든 쿼리 반환
        # 여기서는 모든 쿼리 반환 (Q=1이면 첫 번째만)
        if decoder.num_multimask_outputs > 1:
            # 첫 번째 쿼리만 사용 (multimask_output=True일 때 첫 번째 마스크만 선택하는 것과 일치)
            return mask_tokens_out[:, 0, 1:, :]  # (B, num_multimask_outputs, D)
        else:
            return mask_tokens_out[:, 0, :, :]  # (B, 1, D)


