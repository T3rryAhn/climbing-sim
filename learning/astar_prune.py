"""
A*prune Algorithm - Dynamic Graph용 K-최단 경로
论文 Section 2.3, 5.2 기반
"""

import numpy as np
from typing import List, Tuple, Optional
import heapq
from .stance_graph import StanceGraph, Stance


class AStarPrune:
    """A*prune - k-shortest paths in dynamic graph"""
    
    def __init__(self, graph: StanceGraph):
        self.graph = graph
    
    def heuristic(self, stance_id: int, goal_id: int) -> float:
        """A* 휴리스틱: 목표까지 남은 높이"""
        stance = self.graph.get_stance_by_id(stance_id)
        goal = self.graph.get_stance_by_id(goal_id)
        
        if stance is None or goal is None:
            return 0
        
        # 현재 높이와 목표 높이 차이
        stance_holds = [stance.left_hand, stance.right_hand, stance.left_foot, stance.right_foot]
        goal_holds = [goal.left_hand, goal.right_hand, goal.left_foot, goal.right_foot]
        
        current_height = self.graph.get_avg_height(stance_holds)
        goal_height = self.graph.get_avg_height(goal_holds)
        
        # 남은 높이 (높을수록 가까운 척)
        return max(0, (goal_height - current_height) * 0.5)
    
    def find_k_shortest_paths(self, start_id: int, goal_id: int, k: int = 3) -> List[List[int]]:
        """
        K-최단 경로 탐색 - A*prune 알고리즘
        
        - 첫 번째 경로 실패 시 에지 비용 증가
        - 두 번째 경로 탐색
        - 반복
        """
        all_paths = []
        edge_costs = dict(self.graph.edges)  # 복사
        
        for path_num in range(k):
            path = self._astar_search(start_id, goal_id, edge_costs)
            
            if path is None:
                break
            
            all_paths.append(path)
            
            # 실패한 엣지 비용 증가 (dynamic graph)
            if path_num < k - 1:
                self._increase_failed_edge_costs(path, edge_costs)
        
        return all_paths
    
    def _astar_search(self, start_id: int, goal_id: int, edge_costs: dict) -> Optional[List[int]]:
        """A* 탐색"""
        open_set = [(0, start_id, [start_id])]  # (f_score, node_id, path)
        visited = set()
        g_scores = {start_id: 0}
        
        iterations = 0
        max_iterations = 50000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            f_score, current_id, path = heapq.heappop(open_set)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 목표 도달
            if current_id == goal_id:
                return path
            
            # 이웃 탐색
            neighbors = self.graph.get_neighbors(current_id)
            
            for neighbor_id, base_cost in neighbors:
                if neighbor_id in visited:
                    continue
                
                edge_key = (current_id, neighbor_id)
                actual_cost = edge_costs.get(edge_key, base_cost)
                
                g_new = g_scores[current_id] + actual_cost
                
                if neighbor_id not in g_scores or g_new < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = g_new
                    f_new = g_new + self.heuristic(neighbor_id, goal_id)
                    
                    heapq.heappush(open_set, (f_new, neighbor_id, path + [neighbor_id]))
        
        return None  # 경로 없음
    
    def _increase_failed_edge_costs(self, path: List[int], edge_costs: dict):
        """실패한 경로의 에지 비용 증가"""
        # 경로의 마지막 2개 스턴스 사이 비용 증가
        if len(path) >= 2:
            last_edge = (path[-2], path[-1])
            if last_edge in edge_costs:
                edge_costs[last_edge] *= 2  # 비용 2배
                print(f"  Edge cost increased: {last_edge} -> {edge_costs[last_edge]:.2f}")


def test():
    """테스트"""
    print("=== A*prune 테스트 ===")
    
    from .stance_graph import StanceGraph
    
    graph = StanceGraph()
    
    start = graph.get_start_stance()
    goal = graph.get_goal_stance()
    
    print(f"\n시작: {start.id}")
    print(f"목표: {goal.id}")
    
    astar = AStarPrune(graph)
    
    print("\nK-최단 경로 탐색 (k=3):")
    paths = astar.find_k_shortest_paths(start.id, goal.id, k=3)
    
    for i, path in enumerate(paths):
        print(f"\n경로 {i+1}:")
        total_cost = 0
        for j, stance_id in enumerate(path):
            stance = graph.get_stance_by_id(stance_id)
            if j > 0:
                prev_id = path[j-1]
                edge_key = (prev_id, stance_id)
                cost = graph.edges.get(edge_key, 0)
                total_cost += cost
            print(f"  {j}: {stance}")
        print(f"  총 비용: {total_cost:.2f}")


if __name__ == "__main__":
    test()