# Climbing Simulation

LearningClimbingMovements 스타일의 클라이밍 시뮬레이션

## 구조

```
climbing_sim/
├── hold_extractor/     # 홀드 추출 (GLB/SAM3D)
├── physics_sim/         # 물리 시뮬 (MuJoCo)
├── learning/            # CMA-ES / RL 학습
├── unity/               # Unity 프로젝트
├── data/                # 데이터
└── scripts/             # 유틸리티
```

## 필요 환경

- Python 3.10+
- CUDA (NVIDIA GPU)
- Unity 2022+