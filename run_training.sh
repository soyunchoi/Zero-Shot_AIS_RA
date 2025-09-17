#!/bin/bash

echo "======================================================"
echo "Enhanced D2SA Amodal Instance Segmentation Training"
echo "======================================================"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# GPU 메모리 정리
nvidia-smi > /dev/null 2>&1 && echo "GPU 정보:" && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

# echo ""
# echo "1. 시스템 테스트 실행 중..."
# python test_setup.py

# if [ $? -ne 0 ]; then
#     echo "❌ 시스템 테스트 실패! 설정을 확인해주세요."
#     exit 1
# fi

# echo ""
# echo "2. 학습 시작..."
# echo "Ctrl+C로 중단할 수 있습니다."
# echo ""

# # 학습 실행 (nohup으로 백그라운드 실행 가능)
# if [ "$1" = "background" ]; then
#     echo "백그라운드에서 학습을 시작합니다..."
#     nohup python train.py > training.log 2>&1 &
#     echo "프로세스 ID: $!"
#     echo "로그 확인: tail -f training.log"
# else
python train.py
# fi

echo ""
echo "학습 완료! 결과를 확인하세요:"
echo "- 체크포인트: outputs/D2SA_amodal_visible_head_enhanced_250820/checkpoints/"
echo "- 시각화: outputs/D2SA_amodal_visible_head_enhanced_250820/visualization/"
echo "- 로그: outputs/D2SA_amodal_visible_head_enhanced_250820/logs/" 