"""
Device Manager - CUDA/CPU 자동 선택
"""

import torch

def get_device():
    """실행 환경에 맞는 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU 사용")
    return device

def to_device(data, device):
    """데이터를 디바이스로 이동"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data

# 전역 디바이스
DEVICE = get_device()