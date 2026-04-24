"""
Stance Graph - LearningClimbingMovements 스타일
论文 Section 4-5 기반
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq

# 홀드 (임시)
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
ARM_REACH = 0.8  # 팔 도달 거리
MAX_REACH_FROM_TORSO = 1.2  # 토르소에서 최대 도달


class Stance:
    """Climber stance - 4-hold 설정"""
    
    def __init__(self, left_hand=-1, right_hand=-1, left_foot=-1, right_foot=-1):
        self.left_hand = left_hand   # -1 = 공중
        self.right_hand = right_hand
        self.left_foot = left_foot
        self.right_foot = right_foot
        
        # 노드 ID는 해시로
        self.id = self._hash()
    
    def _hash(self):
        return (self.left_hand + 1) * 1000000 + \
               (self.right_hand + 1) * 100000 + \
               (self.left_foot + 1) * 10000 + \
               (self.right_foot + 1) * 1000
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return self.id
    
    def __repr__(self):
        return f"Stance(LH={self.left_hand},RH={self.right_hand},LF={self.left_foot},RF={self.right_foot})"
    
    def is_valid(self):
        """유효한 스턴스인지"""
        # 최소 2개 홀드 사용
        holds = [self.left_hand, self.right_hand, self.left_foot, self.right_foot]
        active = [h for h in holds if h >= 0]
        return len(active) >= 2
    
    def is_goal(self):
        """목표 도달?"""
        return self.left_hand == 15 or self.right_hand == 15


class StanceGraph:
    """Stance Graph 구현 - 논문 Section 4"""
    
    def __init__(self, holds=HOLDS):
        self.holds = holds
        self.nodes = {}  # id -> Stance
        self.edges = {}  # (node_id1, node_id2) -> edge_cost
        self.adjacency = {}  # node_id -> [neighbor_ids]
        
        self.build_graph()
    
    def build_graph(self):
        """그래프 구축"""
        print("Building stance graph...")
        
        # 모든 가능한 스턴스 생성
        for lh in range(-1, NUM_HOLDS):
            for rh in range(-1, NUM_HOLDS):
                for lf in range(-1, NUM_HOLDS):
                    for rf in range(-1, NUM_HOLDS):
                        stance = Stance(lh, rh, lf, rf)
                        
                        if stance.is_valid():
                            self.nodes[stance.id] = stance
        
        print(f"  Nodes: {len(self.nodes)}")
        
        # 엣지 연결 (한 홀드만 다른 인접 스턴스)
        edge_count = 0
        for stance in self.nodes.values():
            neighbors = self.find_neighbors(stance)
            
            if stance.id not in self.adjacency:
                self.adjacency[stance.id] = []
            
            for neighbor in neighbors:
                cost = self.calculate_edge_cost(stance, neighbor)
                
                edge_key = (stance.id, neighbor.id)
                self.edges[edge_key] = cost
                
                if neighbor.id not in self.adjacency:
                    self.adjacency[neighbor.id] = []
                
                self.adjacency[stance.id].append(neighbor.id)
                edge_count += 1
        
        print(f"  Edges: {edge_count}")
    
    def find_neighbors(self, stance: Stance) -> List[Stance]:
        """인접 스턴스 찾기 (论文 Section 4)"""
        neighbors = []
        
        # 한 홀드씩만 다른 스턴스로 이동
        for i, (cur_h, next_h) in enumerate([
            (stance.left_hand, range(-1, NUM_HOLDS)),
            (stance.right_hand, range(-1, NUM_HOLDS)),
            (stance.left_foot, range(-1, NUM_HOLDS)),
            (stance.right_foot, range(-1, NUM_HOLDS)),
        ]):
            for new_hold in next_h:
                if new_hold == cur_h:
                    continue
                
                # 새 스턴스 생성
                if i == 0:
                    new_stance = Stance(new_hold, stance.right_hand, stance.left_foot, stance.right_foot)
                elif i == 1:
                    new_stance = Stance(stance.left_hand, new_hold, stance.left_foot, stance.right_foot)
                elif i == 2:
                    new_stance = Stance(stance.left_hand, stance.right_hand, new_hold, stance.right_foot)
                else:
                    new_stance = Stance(stance.left_hand, stance.right_hand, stance.left_foot, new_hold)
                
                if new_stance.is_valid() and new_stance.id in self.nodes:
                    neighbors.append(new_stance)
        
        return neighbors
    
    def calculate_edge_cost(self, from_stance: Stance, to_stance: Stance) -> float:
        """엣지 비용 계산 - Cost 함수"""
        # 어떤 홀이 다른지
        from_holds = [from_stance.left_hand, from_stance.right_hand, 
                      from_stance.left_foot, from_stance.right_foot]
        to_holds = [to_stance.left_hand, to_stance.right_hand,
                    to_stance.left_foot, to_stance.right_foot]
        
        # 높이 변화
        height_from = self.get_avg_height(from_holds)
        height_to = self.get_avg_height(to_holds)
        height_diff = height_to - height_from
        
        # 비용 = -높이 상승 + 이동 거리
        cost = -height_diff * 10  # 높이 상승은 비용 감소
        
        # 홀드 간 거리
        for i in range(4):
            if from_holds[i] >= 0 and to_holds[i] >= 0:
                from_pos = self.holds[from_holds[i]]['pos']
                to_pos = self.holds[to_holds[i]]['pos']
                dist = np.linalg.norm(np.array(from_pos) - np.array(to_pos))
                cost += dist * 2  # 이동 거리에 따른 비용
        
        return max(0.1, cost)  # 최소 비용
    
    def get_avg_height(self, hold_indices):
        """홀드들의 평균 높이"""
        heights = []
        for h in hold_indices:
            if h >= 0:
                heights.append(self.holds[h]['pos'][1])
        return np.mean(heights) if heights else 0
    
    def get_start_stance(self) -> Stance:
        """시작 스턴스 (0번과 1번 홀드)"""
        return Stance(0, 1, -1, -1)
    
    def get_goal_stance(self) -> Stance:
        """목표 스턴스 (15번 홀드)"""
        return Stance(15, -1, -1, -1)
    
    def get_stance_by_id(self, id) -> Optional[Stance]:
        return self.nodes.get(id)
    
    def get_neighbors(self, stance_id) -> List[Tuple[int, float]]:
        """인접 노드와 비용 반환"""
        if stance_id not in self.adjacency:
            return []
        return [(n_id, self.edges.get((stance_id, n_id), 1.0)) 
                for n_id in self.adjacency[stance_id]]


def test():
    """테스트"""
    print("=== Stance Graph 테스트 ===")
    
    graph = StanceGraph()
    
    start = graph.get_start_stance()
    goal = graph.get_goal_stance()
    
    print(f"\n시작 스턴스: {start}")
    print(f"목표 스턴스: {goal}")
    print(f"그래프 크기: {len(graph.nodes)} 노드, {len(graph.edges)} 엣지")
    
    # 이웃 확인
    neighbors = graph.get_neighbors(start.id)[:5]
    print(f"\n시작 스턴스 이웃 (상위 5개):")
    for n_id, cost in neighbors:
        stance = graph.get_stance_by_id(n_id)
        if stance:
            print(f"  {stance}, cost={cost:.2f}")


if __name__ == "__main__":
    test()