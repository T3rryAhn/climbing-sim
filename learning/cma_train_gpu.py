"""
Learning Module - GPU 가속 CMA-ES 학습
LearningClimbingMovements 스타일
"""

import numpy as np
import cma
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from physics_sim.kinematic_sim import KinematicClimber
from learning.device import get_device, DEVICE

print(f"\n디바이스: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 홀드 데이터
HOLDS = [
    {"id": 0, "pos": [1.5, 0.5], "type": "start"},
    {"id": 1, "pos": [2.5, 0.5], "type": "start"},
    {"id": 2, "pos": [1.8, 1.0], "type": "normal"},
    {"id": 3, "pos": [2.3, 1.2], "type": "normal"},
    {"id": 4, "pos": [1.5, 1.5], "type": "normal"},
    {"id": 5, "pos": [2.5, 1.8], "type": "normal"},
    {"id": 6, "pos": [1.8, 2.2], "type": "normal"},
    {"id": 7, "pos": [2.3, 2.5], "type": "normal"},
    {"id": 8, "pos": [1.5, 2.8], "type": "normal"},
    {"id": 9, "pos": [2.5, 3.0], "type": "normal"},
    {"id": 10, "pos": [1.8, 3.5], "type": "normal"},
    {"id": 11, "pos": [2.3, 3.8], "type": "normal"},
    {"id": 12, "pos": [2.0, 4.5], "type": "normal"},
    {"id": 13, "pos": [1.5, 5.0], "type": "normal"},
    {"id": 14, "pos": [2.5, 5.2], "type": "normal"},
    {"id": 15, "pos": [2.0, 5.8], "type": "goal"},
]

NUM_HOLDS = len(HOLDS)
HOLDS_POS = torch.tensor([h['pos'] + [0] for h in HOLDS], dtype=torch.float32, device=DEVICE)
GOAL_HEIGHT = 5.5
ARM_LENGTH = 0.7

# Cost 가중치
COST_WEIGHTS = {
    'height': 10.0,
    'stability': 5.0,
    'energy': 0.1,
}


def batch_evaluate(paths_tensor):
    """
    GPU 가속 배치 평가
    paths_tensor: (batch_size, path_length) tensor
    """
    batch_size, path_length = paths_tensor.shape
    
    # 결과를 GPU에 저장
    final_heights = torch.zeros(batch_size, device=DEVICE)
    max_heights = torch.zeros(batch_size, device=DEVICE)
    reach_rates = torch.zeros(batch_size, device=DEVICE)
    
    # 각 경로 평가 (배치 처리)
    for b in range(batch_size):
        # 클라이머 상태 초기화
        handL = torch.tensor([1.5, 0.8, 0], device=DEVICE)
        handR = torch.tensor([2.5, 0.8, 0], device=DEVICE)
        
        heights = []
        reach_count = 0
        
        for i in range(path_length):
            hold_idx = paths_tensor[b, i].item()
            if 0 <= hold_idx < NUM_HOLDS:
                target = HOLDS_POS[hold_idx]
                
                # 왼손 또는 오른손 이동
                if i % 2 == 0:
                    dist = torch.norm(target - handL)
                    if dist <= ARM_LENGTH:
                        handL = target.clone()
                        reach_count += 1
                    else:
                        direction = (target - handL) / dist
                        handL = handL + direction * ARM_LENGTH
                else:
                    dist = torch.norm(target - handR)
                    if dist <= ARM_LENGTH:
                        handR = target.clone()
                        reach_count += 1
                    else:
                        direction = (target - handR) / dist
                        handR = handR + direction * ARM_LENGTH
                
                # 토르소 위치
                torso = (handL + handR) / 2 + torch.tensor([0, -0.3, 0], device=DEVICE)
                heights.append(torso[1].item())
        
        final_heights[b] = max(heights) if heights else 0
        max_heights[b] = max(heights) if heights else 0
        reach_rates[b] = reach_count / path_length if path_length > 0 else 0
    
    # Cost 계산 (배치)
    height_reward = COST_WEIGHTS['height'] * final_heights
    energy_cost = COST_WEIGHTS['energy'] * reach_rates * path_length * 0.1
    
    total_cost = -height_reward + energy_cost
    
    # 실패한 경로 페널티
    success = final_heights >= GOAL_HEIGHT * 0.8
    total_cost += (~success).float() * (GOAL_HEIGHT - final_heights) * 20
    
    return total_cost, final_heights, max_heights, reach_rates


def train():
    """GPU 가속 CMA-ES 학습"""
    print("=" * 60)
    print("GPU 가속 CMA-ES Climbing Learning")
    print("=" * 60)
    print(f"디바이스: {DEVICE}")
    print(f"홀드: {NUM_HOLDS}개, 목표높이: {GOAL_HEIGHT}m")
    print("=" * 60)
    
    # CMA-ES 설정
    path_length = 12
    population_size = 32  # GPU를 위해 늘림
    x0 = np.random.randint(0, NUM_HOLDS, path_length)
    es = cma.CMAEvolutionStrategy(x0, 1.5, inopts={'popsize': population_size})
    
    best_path = None
    best_cost = float('inf')
    best_height = 0
    history = []
    
    start_time = time.time()
    
    for gen in range(100):
        solutions = es.ask()
        
        # 배치 텐서로 변환
        solutions_tensor = torch.tensor(solutions, dtype=torch.long, device=DEVICE)
        
        # GPU 배치 평가
        costs, heights, _, rates = batch_evaluate(solutions_tensor)
        
        costs_cpu = costs.cpu().numpy()
        heights_cpu = heights.cpu().numpy()
        rates_cpu = rates.cpu().numpy()
        
        for i, (cost, height, rate) in enumerate(zip(costs_cpu, heights_cpu, rates_cpu)):
            if cost < best_cost:
                best_cost = cost
                best_path = [int(round(x)) for x in solutions[i]]
                best_height = height
                print(f"Gen {gen:3d}: cost={best_cost:8.2f}, height={best_height:.2f}, rate={rate:.0%}")
        
        es.tell(solutions, costs_cpu.tolist())
        
        if gen % 10 == 0:
            print(f"Gen {gen:3d}: best={best_cost:.2f}")
        
        history.append({'gen': int(gen), 'best_cost': float(best_cost), 'height': float(best_height)})
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("학습 완료")
    print("=" * 60)
    print(f"시간: {total_time:.1f}초 ({total_time/100:.3f}초/세대)")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Best path: {best_path}")
    print(f"Final height: {best_height:.2f}m")
    
    # 결과 저장
    output = {
        'training': {
            'generations': 100,
            'population': population_size,
            'path_length': path_length,
            'best_cost': float(best_cost),
            'device': str(DEVICE)
        },
        'holds': HOLDS,
        'path': [int(x) for x in best_path],
        'goal_height': float(GOAL_HEIGHT),
        'cost_weights': COST_WEIGHTS,
        'history': history
    }
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "data", 
        "climbing_result.json"
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    train()