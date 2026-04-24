"""
Physics Simulation - 순수 Python Kinematic 모델
LearningClimbingMovements 핵심 구현
중력, 관절 물리 없이 Kinematic으로 먼저 구현
"""

import numpy as np

# 팔/다리 관절 제한
ARM_LENGTH = 1.2  # 팔 길이
LEG_LENGTH = 0.6  # 다리 길이
TORSO_HEIGHT = 0.4  # 몸통 높이


class KinematicClimber:
    """Kinematic 등반 시뮬레이션"""
    
    def __init__(self, arm_length=1.2):
        self.ARM_LENGTH = arm_length  # 인스턴스 변수로
        self.torso = np.array([2.0, 0.5, 0])
        self.handL = np.array([1.5, 0.8, 0])
        self.handR = np.array([2.5, 0.8, 0])
        self.footL = np.array([1.8, 0.2, 0])
        self.footR = np.array([2.2, 0.2, 0])
    
    def reset(self):
        self.torso = np.array([2.0, 0.5, 0])
        self.handL = np.array([1.5, 0.8, 0])
        self.handR = np.array([2.5, 0.8, 0])
        self.footL = np.array([1.8, 0.2, 0])
        self.footR = np.array([2.2, 0.2, 0])
    
    def move_hand(self, target, hand='L'):
        """손을 목표 위치로 이동 (근거리 한계)"""
        arm_len = getattr(self, 'ARM_LENGTH', 1.2)  # 기본값 1.2
        
        if hand == 'L':
            dist = np.linalg.norm(target - self.handL)
            if dist <= arm_len:
                self.handL = target.copy()
                return True
            else:
                direction = (target - self.handL) / dist
                self.handL = self.handL + direction * arm_len
                return False
        else:
            dist = np.linalg.norm(target - self.handR)
            if dist <= arm_len:
                self.handR = target.copy()
                return True
            else:
                direction = (target - self.handR) / dist
                self.handR = self.handR + direction * arm_len
                return False
    
    def update_torso(self):
        """토르소 위치 업데이트 (손 사이)"""
        # 토르소는 두 손 사이의 아래쪽
        self.torso = (self.handL + self.handR) / 2 + np.array([0, -0.3, 0])
    
    def update_legs(self):
        """다리 위치 업데이트 (토르소 아래)"""
        self.footL = self.torso + np.array([-0.1, -0.4, 0])
        self.footR = self.torso + np.array([0.1, -0.4, 0])
    
    def get_height(self):
        return self.torso[1]
    
    def get_state(self):
        return {
            'torso': self.torso,
            'handL': self.handL,
            'handR': self.handR,
            'footL': self.footL,
            'footR': self.footR,
            'height': self.get_height()
        }
    
    def execute_path(self, path, hold_positions):
        """경로 실행"""
        self.reset()
        heights = []
        success_count = 0
        
        for i, hold_idx in enumerate(path):
            if hold_idx < 0 or hold_idx >= len(hold_positions):
                continue
            
            target = np.array(hold_positions[hold_idx] + [0])
            
            hand = 'L' if i % 2 == 0 else 'R'
            reached = self.move_hand(target, hand)
            
            if reached:
                success_count += 1
            
            self.update_torso()
            self.update_legs()
            heights.append(self.get_height())
        
        final_height = self.get_height()
        max_height = max(heights) if heights else 0
        
        return {
            'final_height': final_height,
            'max_height': max_height,
            'reach_rate': success_count / len(path) if len(path) > 0 else 0,
            'heights': heights
        }


def test():
    print("=== Kinematic Climber 테스트 ===")
    climber = KinematicClimber()
    
    # 예시 홀드들 (grid-based)
    hold_positions = [
        [1.5, 0.5], [2.5, 0.5],  # 시작
        [1.8, 1.0], [2.3, 1.2],
        [1.5, 1.5], [2.5, 1.8],
        [1.8, 2.2], [2.3, 2.5],
        [1.5, 2.8], [2.5, 3.0],
        [1.8, 3.5], [2.3, 3.8],
        [2.0, 4.5], [1.5, 5.0],
        [2.5, 5.2], [2.0, 5.8],  # 목표
    ]
    
    # CMA-ES가 찾은 경로 대신 간단한 경로
    path = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    print("\n경로 실행:")
    climber.reset()
    
    for i, hold_idx in enumerate(path):
        target = np.array(hold_positions[hold_idx] + [0])  # 3D로
        hand = 'L' if i % 2 == 0 else 'R'
        
        reached = climber.move_hand(target, hand)
        climber.update_torso()
        climber.update_legs()
        
        print(f"  홀드 {hold_idx} ({target[0]:.1f}, {target[1]:.1f}): "
              f"{hand}손 {'도달' if reached else '최대'}, 높이={climber.get_height():.2f}")
    
    result = climber.execute_path(path, hold_positions)
    
    print(f"\n결과:")
    print(f"  최종 높이: {result['final_height']:.2f}m")
    print(f"  최대 높이: {result['max_height']:.2f}m")
    print(f"  도달률: {result['reach_rate']*100:.0f}%")


if __name__ == "__main__":
    test()