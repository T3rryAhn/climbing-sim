"""
Learning Module - CMA-ES로 등반 경로 학습
LearningClimbingMovements 스타일
"""

import numpy as np
import cma
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_sim.humanoid_sim import HumanoidSim, HUMANOID_XML

# 설정
GRID_W, GRID_H = 5, 8
SPACING = 0.8
POP_SIZE = 16
GENS = 100
SIGMA = 1.0

# 홀드 위치 (grid-based)
HOLDS = []
for y in range(GRID_H):
    for x in range(GRID_W):
        HOLDS.append([x * SPACING + 1.5, y * SPACING + 0.5, 0])
NUM_HOLDS = len(HOLDS)

# 목표 높이
GOAL_HEIGHT = (GRID_H - 1) * SPACING + 0.5


class ClimbingEnv:
    """등반 환경"""
    
    def __init__(self, render=False):
        self.sim = HumanoidSim()
        self.render = render
        
        # 홀드 위치
        self.holds = np.array(HOLDS)
        
        # 학습 결과
        self.best_path = None
        self.best_cost = float('inf')
    
    def reset(self):
        self.sim.reset()
        self.sim.data.qpos[0:3] = [2.0, 0.8, 0.5]
    
    def step(self, ctrl=None):
        self.sim.step(ctrl)
    
    def set_joint_angles(self, angles):
        """관절 각도 설정"""
        for i, angle in enumerate(angles):
            if i < self.sim.model.nq:
                self.sim.data.qpos[i] = angle
    
    def move_hand_to(self, hand_id, target_pos, max_steps=50):
        """손을 목표 위치로 이동"""
        for _ in range(max_steps):
            current = self.sim.get_position(hand_id)
            diff = target_pos - current
            dist = np.linalg.norm(diff)
            
            if dist < 0.05:
                return True  # 도달
            
            # 간단한 IK (역운동학)
            ctrl = np.zeros(8)
            
            if hand_id == self.sim.handL_id:
                ctrl[0] = diff[0] * 3  # shoulder_L x
                ctrl[1] = diff[1] * 3 - 1.0  # elbow_L
            else:
                ctrl[2] = diff[0] * 3  # shoulder_R x
                ctrl[3] = diff[1] * 3 - 1.0  # elbow_R
            
            self.step(ctrl)
        
        return False  # 실패
    
    def evaluate_path(self, path_indices):
        """경로 평가"""
        self.reset()
        total_cost = 0
        heights = []
        
        for hold_idx in path_indices:
            if hold_idx < 0 or hold_idx >= NUM_HOLDS:
                continue
            
            target = self.holds[hold_idx] + np.array([0, 0.1, 0])
            
            # 양손交互 이동
            self.move_hand_to(self.sim.handL_id, target)
            heights.append(self.sim.get_height())
            
            self.move_hand_to(self.sim.handR_id, target + np.array([0.15, 0, 0]))
            heights.append(self.sim.get_height())
        
        final_height = self.sim.get_height()
        max_height = max(heights) if heights else 0
        
        # Cost 계산
        if final_height >= GOAL_HEIGHT * 0.8:
            # 성공: 높이 최대화, 에너지 최소화
            cost = -final_height * 10
        else:
            # 실패: 남은 높이 비례 페널티
            cost = (GOAL_HEIGHT - final_height) * 10
        
        return cost, final_height, max_height


def fitness_func(chromosome, env):
    """적합도 함수"""
    path = np.clip(chromosome, 0, NUM_HOLDS - 1).astype(int).tolist()
    cost, _, _ = env.evaluate_path(path)
    return cost


def train():
    """CMA-ES 학습"""
    print("=" * 50)
    print("CMA-ES Climbing Learning")
    print("=" * 50)
    print(f"홀드: {NUM_HOLDS}개, 목표높이: {GOAL_HEIGHT:.1f}")
    print(f"Population: {POP_SIZE}, Generations: {GENS}")
    print("=" * 50)
    
    env = ClimbingEnv()
    
    # CMA-ES 초기화
    x0 = np.random.randint(0, NUM_HOLDS, 10)  # 10스텝 경로
    es = cma.CMAEvolutionStrategy(x0, SIGMA, inopts={'popsize': POP_SIZE})
    
    best_path = None
    best_cost = float('inf')
    history = []
    
    start_time = time.time()
    
    for gen in range(GENS):
        solutions = es.ask()
        costs = []
        
        gen_start = time.time()
        
        for sol in solutions:
            cost = fitness_func(sol, env)
            costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_path = np.clip(sol, 0, NUM_HOLDS - 1).astype(int).tolist()
                
                _, final_h, max_h = env.evaluate_path(best_path)
                print(f"Gen {gen:3d}: cost={cost:8.2f}, height={final_h:.2f}, max={max_h:.2f}")
        
        gen_time = time.time() - gen_start
        es.tell(solutions, costs)
        
        # 통계
        avg_cost = np.mean(costs)
        history.append({'gen': gen, 'best': best_cost, 'avg': avg_cost})
        
        if gen % 10 == 0:
            print(f"Gen {gen:3d}: best={best_cost:.2f}, avg={avg_cost:.2f}, time={gen_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("학습 완료")
    print("=" * 50)
    print(f"시간: {total_time:.1f}초")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Best path: {best_path}")
    
    # 경로 평가
    _, final_h, max_h = env.evaluate_path(best_path)
    print(f"Final height: {final_h:.2f}m (max: {max_h:.2f}m)")
    
    return best_path, best_cost, history


def save_results(path, cost, history):
    """결과 저장"""
    output = {
        'training': {
            'generations': GENS,
            'population': POP_SIZE,
            'best_cost': float(cost),
            'history': history
        },
        'holds': HOLDS,
        'path': path,
        'goal_height': GOAL_HEIGHT,
        'grid': {
            'width': GRID_W,
            'height': GRID_H,
            'spacing': SPACING
        }
    }
    
    output_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "data", 
        f"climbing_result_{int(time.time())}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n결과 저장: {output_path}")
    return output_path


if __name__ == "__main__":
    path, cost, history = train()
    save_results(path, cost, history)