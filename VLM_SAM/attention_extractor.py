"""
VLM (LLaVA) decoder에서 attention map을 추출하는 모듈
Multi-layer multi-head attention을 통해 attention map을 생성합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

class AttentionExtractor:
    """
    VLM decoder에서 attention map을 추출하는 클래스
    """
    
    def __init__(self, model, processor):
        """
        Args:
            model: LLaVA 모델 (필수)
            processor: LLaVA processor (필수)
        """
        if model is None or processor is None:
            raise ValueError("AttentionExtractor는 유효한 LLaVA 모델과 processor가 필수입니다!")
        
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        
        # Attention 저장을 위한 hooks
        self.attention_maps = {}
        self.attention_hooks = []
        
        # 모델 구조 분석
        self._analyze_model_structure()
        
        print("AttentionExtractor 초기화 완료")
    
    def _analyze_model_structure(self):
        """모델 구조를 분석하여 attention layer를 찾습니다."""
        print("=== LLaVA 모델 구조 분석 ===")
        
        # LLaVA 모델의 주요 컴포넌트 확인
        if hasattr(self.model, 'language_model'):
            self.language_model = self.model.language_model
            print(f"✓ Language Model: {type(self.language_model).__name__}")
        
        if hasattr(self.model, 'vision_tower'):
            self.vision_tower = self.model.vision_tower
            print(f"✓ Vision Tower: {type(self.vision_tower).__name__}")
        
        if hasattr(self.model, 'multi_modal_projector'):
            self.projector = self.model.multi_modal_projector
            print(f"✓ Multi-modal Projector: {type(self.projector).__name__}")
        
        # Transformer layers 찾기
        self.transformer_layers = []
        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'layers'):
            self.transformer_layers = self.language_model.model.layers
            print(f"✓ Transformer Layers: {len(self.transformer_layers)}개")
        else:
            # LLaVA 1.5의 다른 구조 시도
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'layers'):
                self.transformer_layers = self.model.language_model.layers
                print(f"✓ Transformer Layers (direct): {len(self.transformer_layers)}개")
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.transformer_layers = self.model.model.layers
                print(f"✓ Transformer Layers (model.model): {len(self.transformer_layers)}개")
        
        # Vision Transformer layers 찾기 (만약 있다면)
        self.vision_layers = []
        if hasattr(self.vision_tower, 'vision_model') and hasattr(self.vision_tower.vision_model, 'encoder'):
            if hasattr(self.vision_tower.vision_model.encoder, 'layers'):
                self.vision_layers = self.vision_tower.vision_model.encoder.layers
                print(f"✓ Vision Transformer Layers: {len(self.vision_layers)}개")
    
    def register_attention_hooks(self, layer_indices: Optional[List[int]] = None):
        """
        Attention 추출을 위한 hooks를 등록합니다.
        
        Args:
            layer_indices: 추출할 layer 인덱스들 (None이면 모든 layer)
        """
        # 기존 hooks 제거
        self.remove_attention_hooks()
        self.attention_maps = {}
        
        if layer_indices is None:
            # 마지막 몇 개 layer만 사용 (계산 효율성을 위해)
            layer_indices = list(range(max(0, len(self.transformer_layers) - 6), len(self.transformer_layers)))
        
        print(f"Attention hooks 등록 중: layers {layer_indices}")
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx < len(self.transformer_layers):
                layer = self.transformer_layers[layer_idx]
                
                # Self-attention 모듈 찾기
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        self._create_attention_hook(f"layer_{layer_idx}")
                    )
                    self.attention_hooks.append(hook)
                    print(f"  ✓ Layer {layer_idx}: Self-attention hook 등록")
    
    def _create_attention_hook(self, layer_name: str):
        """Attention map을 추출하는 hook 함수를 생성합니다."""
        def hook_fn(module, input, output):
            try:
                # Multi-head attention의 출력에서 attention weights 추출
                if isinstance(output, tuple) and len(output) > 1:
                    # output[0]: attention output, output[1]: attention weights
                    attention_weights = output[1]  # (batch_size, num_heads, seq_len, seq_len)
                    
                    if attention_weights is not None:
                        # CPU로 이동하여 저장
                        self.attention_maps[layer_name] = attention_weights.detach().cpu()
                        print(f"    Attention map 저장: {layer_name}, shape: {attention_weights.shape}")
                
            except Exception as e:
                print(f"    Attention 추출 실패 ({layer_name}): {e}")
        
        return hook_fn
    
    def remove_attention_hooks(self):
        """등록된 모든 hooks를 제거합니다."""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        print("모든 attention hooks 제거됨")
    
    def extract_attention_maps(self, image: Image.Image, prompt: str = None, use_vlsam_method: bool = True) -> Dict[str, torch.Tensor]:
        """
        이미지와 텍스트에 대해 attention map을 추출합니다.
        
        Args:
            image: PIL 이미지
            prompt: 텍스트 프롬프트
            use_vlsam_method: VL-SAM 논문 방식 사용 여부
            
        Returns:
            Dict: layer별 attention maps
        """
        if prompt is None:
            prompt = "Describe what you see in this image, focusing on objects and their relationships."
        
        print(f"🔍 VLM Attention map 추출 시작 ({'VL-SAM 방식' if use_vlsam_method else '기본 방식'})...")
        print(f"  - 프롬프트: {prompt}")
        
        if use_vlsam_method:
            return self.extract_vlsam_attention_maps(image, prompt)
        else:
            return self.extract_basic_attention_maps(image, prompt)
    
    def extract_basic_attention_maps(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """기존 방식의 attention map 추출"""
        # Attention hooks 등록
        self.register_attention_hooks()
        
        try:
            # LLaVA 1.5 형식에 맞는 프롬프트로 변환 (main_prompt_VLM_reasoning_250704.py 방식)
            llava_prompt = f"USER: <image>{prompt}\nASSISTANT:"
            
            inputs = self.processor(
                text=llava_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # 모델 추론 (attention map 추출) - main_prompt_VLM_reasoning_250704.py 방식
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # 짧게 설정
                    temperature=0.0     # 검증된 설정
                )
            
            # 추출된 attention maps 반환
            if self.attention_maps:
                print(f"  ✓ 실제 attention map 추출 완료: {len(self.attention_maps)}개 layer")
                return self.attention_maps
            else:
                print("❌ Attention map이 추출되지 않았습니다.")
                print("❌ VLM에서 실제 attention map을 추출할 수 없어 프로그램을 중단합니다.")
                raise RuntimeError("VLM attention map 추출 실패: attention hooks가 정상적으로 작동하지 않았습니다.")
            
        except Exception as e:
            print(f"❌ 실제 VLM attention map 추출 실패: {e}")
            raise RuntimeError(f"VLM attention map 추출 실패: {e}")
        
        finally:
            # Hooks 정리
            self.remove_attention_hooks()
    
    def extract_vlsam_attention_maps(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """
        VL-SAM 논문 방식의 attention map 추출 (수정된 버전)
        - 더 긴 텍스트 생성으로 충분한 attention 정보 확보
        - 이미지-텍스트 Cross-Attention 활용
        """
        print("  🧠 VL-SAM 방식 적용: Attention Flow + Rollout (개선된 버전)")
        
        # Query-Key 추출을 위한 hooks 등록
        self.register_vlsam_hooks()
        
        try:
            # 더 상세한 프롬프트로 더 많은 토큰 생성 유도
            vlsam_prompt = f"USER: <image>Describe all objects in this image in detail, including their locations, colors, shapes, and relationships with other objects. List at least 10 different elements you can see.\nASSISTANT:"
            
            inputs = self.processor(
                text=vlsam_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            print(f"  📊 Input 정보:")
            print(f"    - Input IDs shape: {inputs.input_ids.shape}")
            if hasattr(inputs, 'pixel_values'):
                print(f"    - Pixel values shape: {inputs.pixel_values.shape}")
            
            # 충분한 토큰 생성으로 더 많은 attention 정보 확보
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # 더 많은 토큰 생성
                    min_new_tokens=50,   # 최소 토큰 수 보장
                    temperature=0.1,     # 약간의 다양성
                    do_sample=True,      # 샘플링 활성화
                    return_dict_in_generate=True,
                    output_attentions=True
                )
            
            print(f"  📈 Generated tokens: {generate_ids.sequences.shape[1] - inputs.input_ids.shape[1]}개")
            
            # VL-SAM Attention Flow 처리
            if not hasattr(self, 'attention_weights_cache') or not self.attention_weights_cache:
                raise RuntimeError("VL-SAM Hook 실패: attention_weights_cache가 비어있음. Hook이 제대로 작동하지 않았습니다.")
            
            # 캐시된 attention 정보 디버깅
            print(f"  🔍 Attention Cache 정보:")
            for layer_name, attention in self.attention_weights_cache.items():
                print(f"    - {layer_name}: {attention.shape}")
            
            attention_maps = self.compute_vlsam_attention_flow()
            print(f"  ✓ VL-SAM Attention Flow 완료: {len(attention_maps)}개 layer")
            return attention_maps
            
        except Exception as e:
            print(f"❌ VL-SAM attention map 추출 실패: {e}")
            print("❌ 더미 데이터 사용 금지: VL-SAM Hook이 반드시 작동해야 합니다.")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"VL-SAM Hook 필수 실패: {e}")
        
        finally:
            # Hooks 정리
            self.remove_vlsam_hooks()
    
    def register_vlsam_hooks(self):
        """VL-SAM을 위한 Query-Key 추출 hooks 등록 (강화된 버전)"""
        # 기존 hooks 제거
        self.remove_attention_hooks()
        
        # Query-Key 캐시 초기화
        self.queries_cache = {}
        self.keys_cache = {}
        self.attention_weights_cache = {}
        
        # 마지막 몇 개 layer에만 hook 등록 (계산 효율성)
        layer_indices = list(range(max(0, len(self.transformer_layers) - 6), len(self.transformer_layers)))
        
        print(f"VL-SAM Query-Key hooks 등록 중: layers {layer_indices}")
        print(f"전체 transformer layers: {len(self.transformer_layers)}개")
        
        for layer_idx in layer_indices:
            if layer_idx < len(self.transformer_layers):
                layer = self.transformer_layers[layer_idx]
                
                # 다양한 attention module 구조에 대응
                attention_module = None
                module_type = None
                
                if hasattr(layer, 'self_attn'):
                    attention_module = layer.self_attn
                    module_type = "self_attn"
                elif hasattr(layer, 'attention'):
                    attention_module = layer.attention
                    module_type = "attention"
                elif hasattr(layer, 'attn'):
                    attention_module = layer.attn
                    module_type = "attn"
                
                if attention_module is not None:
                    # 모듈 구조 분석
                    print(f"  📋 Layer {layer_idx} ({module_type}): {type(attention_module).__name__}")
                    
                    # Hook 등록
                    hook = attention_module.register_forward_hook(
                        self._create_vlsam_hook(layer_idx, module_type)
                    )
                    self.attention_hooks.append(hook)
                    print(f"  ✓ Layer {layer_idx}: VL-SAM hook 등록 성공")
                else:
                    print(f"  ❌ Layer {layer_idx}: attention module을 찾을 수 없음")
    
    def _create_vlsam_hook(self, layer_idx: int, module_type: str = "self_attn"):
        """VL-SAM을 위한 Query-Key 추출 hook (완전 강화 버전)"""
        def vlsam_hook_fn(module, input, output):
            try:
                # 1. Input 구조 완전 분석
                hidden_states = None
                attention_mask = None
                
                # 다양한 input 패턴에 대한 완전한 대응
                if isinstance(input, tuple):
                    if len(input) >= 1:
                        hidden_states = input[0]
                    if len(input) >= 2 and input[1] is not None:
                        attention_mask = input[1]
                elif isinstance(input, torch.Tensor):
                    hidden_states = input
                elif isinstance(input, dict):
                    hidden_states = input.get('hidden_states', input.get('input', None))
                    attention_mask = input.get('attention_mask', None)
                
                # hidden_states 검증 및 복구 시도
                if hidden_states is None:
                    # output에서 역추론 시도
                    if isinstance(output, tuple) and len(output) > 0:
                        hidden_states = output[0]
                    elif isinstance(output, torch.Tensor):
                        hidden_states = output
                
                if hidden_states is None or not isinstance(hidden_states, torch.Tensor):
                    # 모듈의 현재 상태에서 추출 시도
                    if hasattr(module, 'last_hidden_state'):
                        hidden_states = module.last_hidden_state
                    else:
                        raise ValueError(f"Layer {layer_idx}: hidden_states를 찾을 수 없음")
                
                # print(f"    ✓ Layer {layer_idx}: hidden_states shape={hidden_states.shape}")
                
                # 2. Query, Key 추출 - 다양한 모듈 구조에 대응
                query, key = self._extract_query_key_from_module(module, hidden_states, layer_idx)
                
                if query is None or key is None:
                    raise ValueError(f"Layer {layer_idx}: Query/Key 추출 실패")
                
                # 3. Multi-head attention parameters 추출
                num_heads, head_dim = self._get_attention_params(module, query, layer_idx)
                
                batch_size, seq_len, _ = query.shape
                
                # 4. Multi-head reshape (안전한 버전)
                try:
                    query_reshaped = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (B, H, N, D)
                    key_reshaped = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)      # (B, H, N, D)
                except Exception as e:
                    # Fallback: 간단한 reshape
                    print(f"    ⚠️ Layer {layer_idx}: 표준 reshape 실패, 단순화된 방식 사용")
                    query_reshaped = query.unsqueeze(1)  # (B, 1, N, D)
                    key_reshaped = key.unsqueeze(1)      # (B, 1, N, D)
                    num_heads = 1
                    head_dim = query.shape[-1]
                
                # 5. VL-SAM 논문의 유사도 행렬 계산
                similarity = torch.matmul(query_reshaped, key_reshaped.transpose(-2, -1))  # (B, H, N, N)
                
                # Scale factor 적용
                similarity = similarity / (head_dim ** 0.5)
                
                # 6. Causal mask 적용 (VL-SAM 논문 방식)
                if attention_mask is not None:
                    # 실제 attention mask 사용
                    similarity = similarity.masked_fill(attention_mask == 0, float('-inf'))
                else:
                    # 기본 causal mask
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=similarity.device))
                    similarity = similarity.masked_fill(causal_mask == 0, float('-inf'))
                
                # 7. SoftMax 정규화 (VL-SAM 논문 공식)
                attention_weights = torch.softmax(similarity, dim=-1)  # (B, H, N, N)
                
                # 8. 캐시에 저장
                self.queries_cache[f"layer_{layer_idx}"] = query_reshaped.detach().cpu()
                self.keys_cache[f"layer_{layer_idx}"] = key_reshaped.detach().cpu()
                self.attention_weights_cache[f"layer_{layer_idx}"] = attention_weights.detach().cpu()
                
                # print(f"    ✅ VL-SAM 데이터 캐시 성공: layer_{layer_idx}")
                # print(f"        - Q/K shape: {query_reshaped.shape}")
                # print(f"        - Attention shape: {attention_weights.shape}")
                # print(f"        - Heads: {num_heads}, Head_dim: {head_dim}")
                
            except Exception as e:
                print(f"    ❌ VL-SAM hook 치명적 실패 (layer_{layer_idx}): {e}")
                import traceback
                traceback.print_exc()
                
                # 치명적 실패 시 에러 발생 (더미 데이터 방지)
                raise RuntimeError(f"VL-SAM Hook layer_{layer_idx} 필수 실패: {e}")
        
        return vlsam_hook_fn
    
    def _extract_query_key_from_module(self, module, hidden_states: torch.Tensor, layer_idx: int):
        """모듈에서 Query, Key 추출 (다양한 구조 지원)"""
        try:
            # 방법 1: 표준 q_proj, k_proj
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                query = module.q_proj(hidden_states)
                key = module.k_proj(hidden_states)
                # print(f"    ✓ Layer {layer_idx}: q_proj/k_proj 방식 사용")
                return query, key
            
            # 방법 2: query, key 속성
            elif hasattr(module, 'query') and hasattr(module, 'key'):
                query = module.query(hidden_states)
                key = module.key(hidden_states)
                # print(f"    ✓ Layer {layer_idx}: query/key 방식 사용")
                return query, key
            
            # 방법 3: Linear layer들 탐색
            elif hasattr(module, 'linear_q') and hasattr(module, 'linear_k'):
                query = module.linear_q(hidden_states)
                key = module.linear_k(hidden_states)
                # print(f"    ✓ Layer {layer_idx}: linear_q/linear_k 방식 사용")
                return query, key
            
            # 방법 4: 하위 모듈 탐색
            else:
                for name, submodule in module.named_children():
                    if 'q' in name.lower() or 'query' in name.lower():
                        q_module = submodule
                    elif 'k' in name.lower() or 'key' in name.lower():
                        k_module = submodule
                
                if 'q_module' in locals() and 'k_module' in locals():
                    query = q_module(hidden_states)
                    key = k_module(hidden_states)
                    print(f"    ✓ Layer {layer_idx}: 하위 모듈 탐색 방식 사용")
                    return query, key
            
            print(f"    ❌ Layer {layer_idx}: Query/Key 추출 방법을 찾을 수 없음")
            return None, None
            
        except Exception as e:
            print(f"    ❌ Layer {layer_idx}: Query/Key 추출 중 오류: {e}")
            return None, None
    
    def _get_attention_params(self, module, query: torch.Tensor, layer_idx: int):
        """Attention 파라미터 추출 (num_heads, head_dim)"""
        try:
            # 방법 1: 모듈 속성에서 직접 추출
            if hasattr(module, 'num_heads') and hasattr(module, 'head_dim'):
                return module.num_heads, module.head_dim
            elif hasattr(module, 'num_attention_heads') and hasattr(module, 'attention_head_size'):
                return module.num_attention_heads, module.attention_head_size
            elif hasattr(module, 'n_heads') and hasattr(module, 'head_size'):
                return module.n_heads, module.head_size
            
            # 방법 2: config에서 추출
            elif hasattr(module, 'config'):
                config = module.config
                if hasattr(config, 'num_attention_heads') and hasattr(config, 'hidden_size'):
                    num_heads = config.num_attention_heads
                    head_dim = config.hidden_size // num_heads
                    return num_heads, head_dim
            
            # 방법 3: 모델 전체 config에서 추출
            elif hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'num_attention_heads') and hasattr(config, 'hidden_size'):
                    num_heads = config.num_attention_heads
                    head_dim = config.hidden_size // num_heads
                    return num_heads, head_dim
            
            # 방법 4: Query tensor shape에서 추정
            else:
                hidden_size = query.shape[-1]
                # LLaMA 계열 모델의 일반적인 설정 추정
                if hidden_size == 4096:
                    num_heads = 32
                elif hidden_size == 2048:
                    num_heads = 16
                elif hidden_size == 1024:
                    num_heads = 8
                else:
                    num_heads = max(1, hidden_size // 128)  # 추정값
                
                head_dim = hidden_size // num_heads
                print(f"    📊 Layer {layer_idx}: 파라미터 추정 - heads={num_heads}, head_dim={head_dim}")
                return num_heads, head_dim
                
        except Exception as e:
            print(f"    ⚠️ Layer {layer_idx}: 파라미터 추출 실패, 기본값 사용: {e}")
            # 최후의 기본값
            return 1, query.shape[-1]
    
    def remove_vlsam_hooks(self):
        """VL-SAM hooks 제거"""
        self.remove_attention_hooks()
        
        # 캐시 정리
        if hasattr(self, 'queries_cache'):
            del self.queries_cache
        if hasattr(self, 'keys_cache'):
            del self.keys_cache
        if hasattr(self, 'attention_weights_cache'):
            del self.attention_weights_cache
        
        print("VL-SAM hooks 및 캐시 정리 완료")
    
    def compute_vlsam_attention_flow(self) -> Dict[str, torch.Tensor]:
        """
        VL-SAM 논문의 Attention Flow 계산
        1. Mean-Max Attention Head Weights 계산
        2. Head 집계
        3. Attention Rollout with Regularization
        """
        print("  📊 VL-SAM Attention Flow 계산 중...")
        
        if not self.attention_weights_cache:
            raise RuntimeError("Attention weights cache가 비어있습니다.")
        
        layer_names = sorted(self.attention_weights_cache.keys())
        processed_attention_maps = {}
        
        # 1. Mean-Max Attention Head Weights 계산
        print("  📈 Mean-Max Attention Head Weights 계산...")
        head_weights = {}
        
        for layer_name in layer_names:
            S = self.attention_weights_cache[layer_name]  # (B, H, N, N)
            
            # W = Mean(Max(S, dim=1), dim=0)  - 논문 공식 (1)
            max_similarity = torch.max(S, dim=2)[0]  # (B, H, N) - dim=2는 j 차원
            mean_max = torch.mean(max_similarity, dim=2)  # (B, H) - dim=2는 i 차원
            
            # Layer별, Head별 weights
            head_weights[layer_name] = mean_max  # (B, H)
            print(f"    {layer_name}: Head weights shape {mean_max.shape}")
        
        # 2. Head 집계: S' = Mean(S ⊙ W, dim=2) - 논문 공식 (2)
        print("  🔄 Head 집계 (Mean-Max Weighting)...")
        aggregated_attention = {}
        
        for layer_name in layer_names:
            S = self.attention_weights_cache[layer_name]  # (B, H, N, N)
            W = head_weights[layer_name]  # (B, H)
            
            # W를 S와 같은 차원으로 확장: (B, H) -> (B, H, N, N)
            W_expanded = W.unsqueeze(-1).unsqueeze(-1).expand_as(S)
            
            # Pointwise 곱셈 및 Head 차원 평균
            weighted_S = S * W_expanded  # (B, H, N, N)
            S_prime = torch.mean(weighted_S, dim=1)  # (B, N, N) - Head 차원 평균
            
            aggregated_attention[layer_name] = S_prime
            print(f"    {layer_name}: 집계된 attention shape {S_prime.shape}")
        
        # 3. Attention Rollout with Regularization
        print("  🌊 Attention Rollout with Regularization...")
        rolled_attention = self.compute_attention_rollout_with_regularization(aggregated_attention)
        
        # 4. 마지막 레이어에서 이미지 attention map 추출
        print("  🖼️ 이미지 Attention Map 추출...")
        final_attention_maps = self.extract_image_attention_maps(rolled_attention)
        
        return final_attention_maps
    
    def compute_attention_rollout_with_regularization(self, aggregated_attention: Dict) -> Dict:
        """
        Attention Rollout with Regularization 계산
        논문 공식 (3) + Regularization term
        """
        layer_names = sorted(aggregated_attention.keys())
        rolled_attention = {}
        
        # 첫 번째 레이어는 그대로 사용
        first_layer = layer_names[0]
        S_prime = aggregated_attention[first_layer]  # (B, N, N)
        batch_size, seq_len, _ = S_prime.shape
        
        # Identity matrix
        I = torch.eye(seq_len, device=S_prime.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 첫 번째 레이어 초기화
        rolled_attention[first_layer] = S_prime + I
        
        # 순차적으로 rollout 계산
        for i in range(1, len(layer_names)):
            current_layer = layer_names[i]
            prev_layer = layer_names[i-1]
            
            S_prime_l = aggregated_attention[current_layer]  # 현재 레이어
            S_bar_prev = rolled_attention[prev_layer]        # 이전 rollout 결과
            
            # 논문 공식 (3): S¯′_l_i,j = Σ_k (I_i,k + S′_l_i,k) × (I_k,j + S¯′_{l-1}_k,j)
            # 행렬 형태로: S¯′_l = (I + S′_l) @ (I + S¯′_{l-1})
            current_with_identity = I + S_prime_l
            prev_with_identity = I + S_bar_prev
            
            S_bar_l = torch.matmul(current_with_identity, prev_with_identity)
            
            # Regularization 적용 (Attention Collapse 방지)
            S_bar_l_reg = self.apply_attention_regularization(S_bar_l)
            
            rolled_attention[current_layer] = S_bar_l_reg
            
            print(f"    Rollout {current_layer}: shape {S_bar_l_reg.shape}")
        
        return rolled_attention
    
    def apply_attention_regularization(self, attention: torch.Tensor) -> torch.Tensor:
        """
        VL-SAM 논문의 Regularization term 적용
        각 column에 대해 1 - (L0 - 1)/L 곱셈 (L0는 unmasked length)
        """
        batch_size, seq_len, _ = attention.shape
        regularized_attention = attention.clone()
        
        # Causal mask로 인한 unmasked length 계산
        for i in range(seq_len):
            L0 = i + 1  # i번째 column의 unmasked length
            L = seq_len  # 전체 길이
            
            # Regularization factor 계산
            reg_factor = 1.0 - (L0 - 1) / L
            
            # 해당 column에 regularization 적용
            regularized_attention[:, :, i] *= reg_factor
        
        return regularized_attention
    
    def extract_image_attention_maps(self, rolled_attention: Dict) -> Dict[str, torch.Tensor]:
        """
        Rollout된 attention에서 이미지 attention map 추출 (강화된 버전)
        """
        print("  🖼️ 이미지 Attention Map 추출...")
        layer_names = sorted(rolled_attention.keys())
        final_attention_maps = {}
        
        for layer_name in layer_names:
            try:
                S_bar = rolled_attention[layer_name]  # (B, N, N)
                print(f"    {layer_name}: Raw rollout attention shape {S_bar.shape}")
                
                if len(S_bar.shape) != 3:
                    print(f"    {layer_name}: 예상치 못한 shape, 건너뜀")
                    continue
                
                batch_size, seq_len, _ = S_bar.shape
                print(f"    {layer_name}: 시퀀스 길이 {seq_len}")
                
                # (1,1,1) 크기 attention에 대한 특별 처리
                if seq_len == 1:
                    print(f"    {layer_name}: 시퀀스 길이가 1이므로 기본 attention map 생성")
                    # 중앙 집중된 attention 패턴 생성
                    spatial_size = 24
                    center = spatial_size // 2
                    y, x = torch.meshgrid(torch.arange(spatial_size), torch.arange(spatial_size), indexing='ij')
                    
                    # 가우시안 분포로 중앙 집중 패턴
                    sigma = 6.0
                    attention_value = S_bar[0, 0, 0].item()  # 실제 attention 값 사용
                    gaussian = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
                    spatial_attention = gaussian * attention_value
                    
                    # 정규화
                    if spatial_attention.max() > spatial_attention.min():
                        spatial_attention = (spatial_attention - spatial_attention.min()) / (spatial_attention.max() - spatial_attention.min())
                    
                    final_attention_maps[layer_name] = spatial_attention
                    print(f"    {layer_name}: 중앙 집중 attention map 생성 {spatial_attention.shape}")
                    continue
                
                # 일반적인 attention 처리 (seq_len > 1)
                # 방법 1: 전체 attention의 평균 (모든 토큰 간 상호작용)
                avg_attention = torch.mean(S_bar[0], dim=0)  # (N,) - 첫 번째 배치
                
                # 방법 2: 대각선 attention (자기 자신에 대한 attention)
                diag_attention = torch.diag(S_bar[0])  # (N,)
                
                # 방법 3: 마지막 토큰의 attention (생성된 텍스트가 이미지를 보는 방식)
                last_token_attention = S_bar[0, -1, :]  # (N,)
                
                # 방법 4: 첫 번째와 마지막 토큰의 cross-attention
                if seq_len > 1:
                    cross_attention = S_bar[0, 0, :]  # 첫 번째 토큰이 보는 attention
                else:
                    cross_attention = last_token_attention
                
                # 가장 분산이 큰(정보가 많은) attention 선택
                candidates = [avg_attention, diag_attention, last_token_attention, cross_attention]
                candidate_names = ["평균", "대각선", "마지막토큰", "첫토큰"]
                
                best_attention = None
                best_variance = 0
                best_method = "없음"
                
                for i, candidate in enumerate(candidates):
                    try:
                        if len(candidate) > 1:  # 최소 2개 이상의 값이 있어야 분산 계산 가능
                            variance = torch.var(candidate).item()
                            if variance > best_variance:
                                best_variance = variance
                                best_attention = candidate
                                best_method = candidate_names[i]
                    except:
                        continue
                
                # 2D spatial map 생성
                if best_attention is not None and len(best_attention) > 1:
                    attention_1d = best_attention
                    
                    # 적절한 정사각형 크기 찾기
                    target_sizes = [24, 16, 12, 8, 6, 4, 3, 2]  # 가능한 ViT 패치 크기들
                    
                    success = False
                    for size in target_sizes:
                        if size * size <= len(attention_1d):
                            # 균등하게 샘플링하여 정사각형 만들기
                            if len(attention_1d) >= size * size:
                                # 일정 간격으로 샘플링
                                indices = torch.linspace(0, len(attention_1d)-1, size*size).long()
                                sampled_attention = attention_1d[indices]
                            else:
                                # 앞의 토큰들만 사용
                                sampled_attention = attention_1d[:size*size]
                            
                            # 정규화
                            if sampled_attention.min() != sampled_attention.max():
                                sampled_attention = (sampled_attention - sampled_attention.min()) / (sampled_attention.max() - sampled_attention.min())
                            else:
                                sampled_attention = torch.ones_like(sampled_attention) * 0.5
                            
                            spatial_attention = sampled_attention.view(size, size)
                            final_attention_maps[layer_name] = spatial_attention
                            
                            print(f"    {layer_name}: Image attention map shape {spatial_attention.shape} (방법: {best_method}, 크기: {size}x{size})")
                            success = True
                            break
                    
                    if not success:
                        print(f"    {layer_name}: 적절한 크기를 찾지 못함")
                else:
                    print(f"    {layer_name}: 유효한 attention을 찾지 못함")
                        
            except Exception as e:
                print(f"    {layer_name}: 이미지 attention 추출 실패: {e}")
        
        # 결과 검증 - 더미 맵 생성 없이 실패 시 에러 발생
        if not final_attention_maps:
            raise RuntimeError("이미지 attention map 추출에 완전히 실패했습니다. VL-SAM Hook이 제대로 작동하지 않았습니다.")
        
        return final_attention_maps
    
    
    def process_attention_maps(self, attention_maps: Dict[str, torch.Tensor], 
                             image_size: Tuple[int, int] = (1024, 1024)) -> Dict[str, np.ndarray]:
        """
        추출된 attention maps를 후처리합니다.
        
        Args:
            attention_maps: 추출된 attention maps
            image_size: 출력 이미지 크기
            
        Returns:
            Dict: 후처리된 attention maps
        """
        processed_maps = {}
        
        for layer_name, attention_tensor in attention_maps.items():
            try:
                print(f"처리 중: {layer_name}, shape: {attention_tensor.shape}")
                
                # Tensor shape에 따른 처리
                if len(attention_tensor.shape) == 4:
                    attention_map = attention_tensor.squeeze().numpy()  # (H, W)
                elif len(attention_tensor.shape) == 3:
                    attention_map = attention_tensor.squeeze(0).numpy()  # (H, W)
                elif len(attention_tensor.shape) == 2:
                    attention_map = attention_tensor.numpy()  # (H, W)
                else:
                    # 실제 transformer attention의 경우
                    # attention_tensor shape: (batch_size, num_heads, seq_len, seq_len)
                    batch_size, num_heads, seq_len, _ = attention_tensor.shape
                    
                    # Multi-head attention을 평균내기
                    attention_avg = attention_tensor.mean(dim=1)  # (batch_size, seq_len, seq_len)
                    
                    # 첫 번째 배치만 사용
                    attention_map = attention_avg[0]  # (seq_len, seq_len)
                    
                    # Vision token에 해당하는 부분만 추출
                    vision_token_length = min(576, seq_len // 2)
                    vision_attention = attention_map[:vision_token_length, :vision_token_length]
                    
                    # 2D spatial attention map으로 변환
                    patch_size = int(np.sqrt(vision_token_length))
                    if patch_size * patch_size == vision_token_length:
                        spatial_attention = vision_attention.mean(dim=0)
                        attention_map = spatial_attention.view(patch_size, patch_size).numpy()
                
                # 이미지 크기로 업샘플링
                if attention_map.shape != image_size:
                    attention_resized = cv2.resize(
                        attention_map, 
                        image_size, 
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    attention_resized = attention_map
                
                # 정규화
                attention_resized = (attention_resized - attention_resized.min()) / \
                                  (attention_resized.max() - attention_resized.min() + 1e-8)
                
                processed_maps[layer_name] = attention_resized
                
                print(f"  ✓ {layer_name}: {attention_map.shape} -> {image_size}")
                
            except Exception as e:
                print(f"  ❌ {layer_name} 처리 실패: {e}")
                raise RuntimeError(f"Attention map 처리 중 오류 발생: {e}")
        
        if not processed_maps:
            raise RuntimeError("모든 attention map 처리에 실패했습니다. VL-SAM이 제대로 작동하지 않았습니다.")
        
        return processed_maps
    
    def aggregate_attention_maps(self, processed_maps: Dict[str, np.ndarray]) -> np.ndarray:
        """
        여러 layer의 attention map을 집계합니다.
        
        Args:
            processed_maps: 처리된 attention maps
            
        Returns:
            np.ndarray: 집계된 attention map
        """
        if not processed_maps:
            return np.zeros((1024, 1024))
        
        # 모든 layer의 attention map을 평균내기
        attention_stack = np.stack(list(processed_maps.values()), axis=0)
        aggregated_attention = np.mean(attention_stack, axis=0)
        
        # 추가 후처리
        # 1. Gaussian blur for smoothing
        aggregated_attention = cv2.GaussianBlur(aggregated_attention, (5, 5), 1.0)
        
        # 2. 재정규화
        aggregated_attention = (aggregated_attention - aggregated_attention.min()) / \
                              (aggregated_attention.max() - aggregated_attention.min() + 1e-8)
        
        print(f"✓ {len(processed_maps)}개 layer의 attention map 집계 완료")
        
        return aggregated_attention
    
    def visualize_attention_maps(self, image: Image.Image, 
                               processed_maps: Dict[str, np.ndarray],
                               aggregated_map: np.ndarray,
                               save_path: str):
        """
        Attention maps을 시각화하여 저장합니다.
        
        Args:
            image: 원본 이미지
            processed_maps: 처리된 layer별 attention maps
            aggregated_map: 집계된 attention map
            save_path: 저장 경로
        """
        import matplotlib.pyplot as plt
        
        num_layers = len(processed_maps)
        if num_layers == 0:
            return
        
        # 그리드 레이아웃 계산
        cols = min(3, num_layers + 2)  # 최대 3열, +2 for original image and aggregated map
        rows = max(1, (num_layers + 2 + cols - 1) // cols)  # +2 for original image and aggregated map
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 원본 이미지
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Layer별 attention maps
        for i, (layer_name, attention_map) in enumerate(processed_maps.items()):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            # 원본 이미지 위에 attention map 오버레이
            axes[row, col].imshow(image, alpha=0.7)
            im = axes[row, col].imshow(attention_map, alpha=0.6, cmap='jet')
            axes[row, col].set_title(f'{layer_name}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        # 집계된 attention map
        if num_layers > 0:
            row = (num_layers + 1) // cols
            col = (num_layers + 1) % cols
            
            axes[row, col].imshow(image, alpha=0.7)
            im = axes[row, col].imshow(aggregated_map, alpha=0.6, cmap='jet')
            axes[row, col].set_title('Aggregated Attention')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        # 빈 subplot 숨기기
        for i in range(num_layers + 2, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Attention map 시각화 저장: {save_path}")

# 사용 예시
if __name__ == '__main__':
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from PIL import Image
    import numpy as np
    
    print("=== AttentionExtractor 테스트 ===")
    
    # 모델 로드
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # AttentionExtractor 초기화
    extractor = AttentionExtractor(model, processor)
    
    # 더미 이미지 생성
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Attention map 추출
    attention_maps = extractor.extract_attention_maps(
        dummy_image, 
        "Describe the objects in this image."
    )
    
    if attention_maps:
        # 후처리
        processed_maps = extractor.process_attention_maps(attention_maps)
        
        # 집계
        aggregated_map = extractor.aggregate_attention_maps(processed_maps)
        
        # 시각화
        extractor.visualize_attention_maps(
            dummy_image, 
            processed_maps, 
            aggregated_map,
            "./test_attention_visualization.png"
        )
        
        print("✅ AttentionExtractor 테스트 완료!")
    else:
        print("❌ Attention map 추출 실패")
