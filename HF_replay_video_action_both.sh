#!/bin/bash

echo "🚀 Unitree G1 데이터 분석 및 비디오 재생을 시작합니다..."

# 1. 첫 번째 스크립트 실행 (백그라운드 실행을 위해 끝에 & 붙임)
python3 /home/taeung/g1_datasets_huggingface/HF_video_replay.py &

# 2. 두 번째 스크립트 실행 (이건 터미널 로그를 보기 위해 그대로 실행)
python3 /home/taeung/g1_datasets_huggingface/HF_replay_dataset.py

# 모든 작업이 끝날 때까지 대기
wait