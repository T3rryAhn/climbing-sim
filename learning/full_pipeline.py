"""
LearningClimbingMovements 통합 학습 파이프라인
论文 순서: Stance Graph → A*prune → Low-level 시뮬 → 재탐색
"""

import numpy as np
import torch
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from physics_sim.kinematic_sim import KinematicClimber
from learning.stance_graph import StanceGraph, Stance
from learning.astar_prune import AStarPrune
from learning.device import DEVICE

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
GOAL_HEIGHT = 5.5
ARM_LENGTH = 1.2  # 팔 길이 증가


class LowLevelController:
    """
    Low-level 컨트롤러 -论文 Section 5
    Stance 전환 시뮬레이션
    """
    
    def __init__(self, arm_length=1.2):
        self.arm_length = arm_length
        self.climber = KinematicClimber(arm_length=arm_length)
        self.reach_threshold = arm_length
        self.reach_angle_threshold = 0.5
        self.reach_angle_threshold = 0.5
    
    def reset(self):
        self.climber.reset()
    
    def simulate_transition(self, from_stance: Stance, to_stance: Stance) -> dict:
        """
        from_stance에서 to_stance로 전환 시뮬
        성공: True, 실패: False
        """
        self.reset()
        
        # 시작 스턴스의 홀드에 손/발 배치
        self._set_stance(from_stance)
        
        # 어떤 홀이 다른지 확인
        changes = self._get_changes(from_stance, to_stance)
        
        if not changes:
            return {'success': True, 'height': self.climber.get_height()}
        
        # 변경된 홀드만 이동
        heights = [self.climber.get_height()]
        all_reached = True
        
        for limb, new_hold in changes.items():
            if new_hold < 0:
                continue  # 공중
            
            target = np.array(HOLDS[new_hold]['pos'] + [0])
            hand = 'L' if limb in ['left_hand', 'left_foot'] else 'R'
            
            if limb in ['left_hand', 'right_hand']:
                reached = self.climber.move_hand(target, hand)
            else:
                # 발은 아래로
                reached = True
            
            if not reached:
                all_reached = False
            
            self.climber.update_torso()
            self.climber.update_legs()
            heights.append(self.climber.get_height())
        
        # 스턴스 업데이트
        self._set_stance(to_stance)
        
        return {
            'success': all_reached,
            'height': self.climber.get_height(),
            'max_height': max(heights),
            'reached_limbs': sum(1 for c in changes.values() if c >= 0)
        }
    
    def _set_stance(self, stance: Stance):
        """스턴스에서 손/발 위치 설정"""
        if stance.left_hand >= 0:
            self.climber.handL = np.array(HOLDS[stance.left_hand]['pos'] + [0])
        if stance.right_hand >= 0:
            self.climber.handR = np.array(HOLDS[stance.right_hand]['pos'] + [0])
        
        self.climber.update_torso()
        self.climber.update_legs()
    
    def _get_changes(self, from_s: Stance, to_s: Stance) -> dict:
        """어떤 사지가 변경되었는���"""
        changes = {}
        
        if from_s.left_hand != to_s.left_hand:
            changes['left_hand'] = to_s.left_hand
        if from_s.right_hand != to_s.right_hand:
            changes['right_hand'] = to_s.right_hand
        if from_s.left_foot != to_s.left_foot:
            changes['left_foot'] = to_s.left_foot
        if from_s.right_foot != to_s.right_foot:
            changes['right_foot'] = to_s.right_foot
        
        return changes


class FullLearningPipeline:
    """
    전체 학습 파이프라인 -论文 Section 5
    1. Stance Graph 구축
    2. A*prune으로 경로 탐색
    3. Low-level 시뮬으로 검증
    4. 실패 시 edge cost 증가 → 재탐색
    """
    
    def __init__(self):
        print("1. Stance Graph 구축 중...")
        self.graph = StanceGraph(HOLDS)
        
        print("2. A*prune 초기화 중...")
        self.astar = AStarPrune(self.graph)
        
        print("3. Low-level 컨트롤러 초기화 중...")
        self.controller = LowLevelController(arm_length=ARM_LENGTH)
        
        # Dynamic edge costs
        self.edge_costs = dict(self.graph.edges)
    
    def run(self, max_iterations=10):
        """
        전체 학습 실행
        """
        start_stance = self.graph.get_start_stance()
        goal_stance = self.graph.get_goal_stance()
        
        print(f"\n시작: {start_stance}")
        print(f"목표: {goal_stance}")
        
        self.best_path = None
        self.best_cost = 0
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"반복 {iteration + 1}")
            print(f"{'='*50}")
            
            # Step 1: A*prune으로 경로 탐색
            print("\n[A*prune 경로 탐색]")
            path_ids = self.astar._astar_search(
                start_stance.id, 
                goal_stance.id, 
                self.edge_costs
            )
            
            if path_ids is None:
                print("경로 없음!")
                break
            
            path_stances = [self.graph.get_stance_by_id(sid) for sid in path_ids]
            print(f"탐색된 경로: {len(path_stances)} 스텝")
            
            # Step 2: Low-level 시뮬
            print("\n[Low-level 시뮬]")
            total_height = 0
            max_height = 0
            failures = []
            
            for i in range(len(path_stances) - 1):
                from_s = path_stances[i]
                to_s = path_stances[i + 1]
                
                result = self.controller.simulate_transition(from_s, to_s)
                
                total_height += result['height']
                max_height = max(max_height, result['max_height'])
                
                if not result['success']:
                    failures.append(i)
                    print(f"  스텝 {i}: 실패 (높이={result['height']:.2f})")
                    
                    # 실패한 엣지의 비용 증가
                    edge_key = (from_s.id, to_s.id)
                    if edge_key in self.edge_costs:
                        old_cost = self.edge_costs[edge_key]
                        self.edge_costs[edge_key] *= 2
                        print(f"    Edge cost: {old_cost:.2f} → {self.edge_costs[edge_key]:.2f}")
                else:
                    print(f"  스텝 {i}: 성공 (높이={result['height']:.2f})")
            
            # 결과 평가
            success_rate = 1 - len(failures) / (len(path_stances) - 1) if len(path_stances) > 1 else 0
            
            print(f"\n결과:")
            print(f"  최종 높이: {total_height:.2f}m")
            print(f"  최대 높이: {max_height:.2f}m")
            print(f"  성공률: {success_rate*100:.0f}%")
            
            if success_rate > 0.8 and max_height >= GOAL_HEIGHT * 0.8:
                print(f"\n성공!")
                self.best_path = path_stances
                self.best_cost = 1 - success_rate
                break
            elif len(failures) == 0 and max_height > self.best_cost:
                self.best_path = path_stances
                self.best_cost = max_height
        
        return self.best_path, self.best_cost
    
    def save_results(self):
        """결과 저장"""
        if self.best_path is None:
            return
        
        output = {
            'pipeline': 'LearningClimbingMovements',
            'holds': HOLDS,
            'path': [s.id for s in self.best_path],
            'path_stances': [str(s) for s in self.best_path],
            'final_height': self.best_cost,
            'goal_height': GOAL_HEIGHT
        }
        
        output_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "data", 
            "climbing_result_stance.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("LearningClimbingMovements 통합 학습")
    print("=" * 60)
    
    pipeline = FullLearningPipeline()
    path, cost = pipeline.run(max_iterations=5)
    
    if path:
        pipeline.save_results()
        print(f"\n최적 경로 ({len(path)} 스텝):")
        for i, stance in enumerate(path):
            print(f"  {i}: {stance}")
    else:
        print("경로 찾기 실패")