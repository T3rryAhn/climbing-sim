"""
Physics Simulation - MuJoCo 휴머노이드 모델
LearningClimbingMovements 스타일
"""

import mujoco
from mujoco import MjModel, MjData
import numpy as np

# 휴머노이드 XML 모델
HUMANOID_XML = """
<mujoco model="humanoid">
    <option timestep="0.001" integrator="Euler" gravity="0 0 -9.81"/>
    
    <worldbody>
        <!-- 바닥 -->
        <geom type="plane" size="10 10 0.01" rgba="0.5 0.5 0.5 1" friction="1 0.1 0.1"/>
        
        <!-- 벽 -->
        <geom type="box" size="2 4 0.1" pos="2 3.2 -0.05" rgba="0.8 0.8 0.8 1"/>
        
        <!-- 홀드들 - 예시 -->
        <body name="wall">
            <geom type="sphere" size="0.08" pos="1.5 1.0 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.5 1.0 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.0 1.8 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="1.5 2.6 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.5 2.6 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.0 3.4 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="1.5 4.2 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.5 4.2 0" rgba="0.9 0.2 0.2 1" friction="2"/>
            <geom type="sphere" size="0.08" pos="2.0 5.0 0" rgba="0.2 0.9 0.2 1" friction="2"/>
        </body>
        
        <!-- Humanoid - 관절 연결 -->
        <body name="torso" pos="2.0 0.5 0.5">
            <freejoint/>
            <inertial pos="0 0.15 0" mass="15" diaginertia="0.1 0.1 0.1"/>
            
            <!-- 몸통 -->
            <geom type="capsule" size="0.08" fromto="0 -0.1 0 0 0.35 0" rgba="1 0.6 0.4 1" friction="1"/>
            
            <!-- 머리 -->
            <body name="head" pos="0 0.45 0">
                <inertial pos="0 0.05 0" mass="5" diaginertia="0.02 0.02 0.02"/>
                <geom type="sphere" size="0.1" rgba="1 0.6 0.4 1" friction="1"/>
            </body>
            
            <!-- 왼팔 -->
            <body name="upper_arm_L" pos="-0.12 0.3 0">
                <inertial pos="0 -0.1 0" mass="2" diaginertia="0.01 0.01 0.01"/>
                <joint name="shoulder_L" type="ball"/>
                <geom type="capsule" size="0.035" fromto="0 0 0 0 -0.2 0" rgba="1 0.6 0.4 1" friction="1"/>
                <body name="lower_arm_L" pos="0 -0.2 0">
                    <inertial pos="0 -0.1 0" mass="1.5" diaginertia="0.005 0.005 0.005"/>
                    <joint name="elbow_L" type="hinge" range="-2.5 0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 -0.2 0" rgba="1 0.6 0.4 1" friction="1"/>
                    <body name="hand_L" pos="0 -0.2 0">
                        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                        <geom type="sphere" size="0.05" rgba="1 0 0 1" friction="3"/>
                    </body>
                </body>
            </body>
            
            <!-- 오른팔 -->
            <body name="upper_arm_R" pos="0.12 0.3 0">
                <inertial pos="0 -0.1 0" mass="2" diaginertia="0.01 0.01 0.01"/>
                <joint name="shoulder_R" type="ball"/>
                <geom type="capsule" size="0.035" fromto="0 0 0 0 -0.2 0" rgba="1 0.6 0.4 1" friction="1"/>
                <body name="lower_arm_R" pos="0 -0.2 0">
                    <inertial pos="0 -0.1 0" mass="1.5" diaginertia="0.005 0.005 0.005"/>
                    <joint name="elbow_R" type="hinge" range="-2.5 0"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 -0.2 0" rgba="1 0.6 0.4 1" friction="1"/>
                    <body name="hand_R" pos="0 -0.2 0">
                        <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                        <geom type="sphere" size="0.05" rgba="1 0 0 1" friction="3"/>
                    </body>
                </body>
            </body>
            
            <!-- 왼다리 -->
            <body name="thigh_L" pos="-0.08 -0.15 0">
                <inertial pos="0 -0.15 0" mass="3" diaginertia="0.02 0.02 0.02"/>
                <joint name="hip_L" type="ball"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.6 1 1" friction="1"/>
                <body name="shin_L" pos="0 -0.3 0">
                    <inertial pos="0 -0.15 0" mass="2" diaginertia="0.01 0.01 0.01"/>
                    <joint name="knee_L" type="hinge" range="-2.5 0"/>
                    <geom type="capsule" size="0.035" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.6 1 1" friction="1"/>
                    <body name="foot_L" pos="0 -0.3 0">
                        <inertial pos="0 0 0" mass="0.8" diaginertia="0.005 0.005 0.005"/>
                        <geom type="box" size="0.06 0.025 0.1" rgba="0 1 0 1" friction="3"/>
                    </body>
                </body>
            </body>
            
            <!-- 오른다리 -->
            <body name="thigh_R" pos="0.08 -0.15 0">
                <inertial pos="0 -0.15 0" mass="3" diaginertia="0.02 0.02 0.02"/>
                <joint name="hip_R" type="ball"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.6 1 1" friction="1"/>
                <body name="shin_R" pos="0 -0.3 0">
                    <inertial pos="0 -0.15 0" mass="2" diaginertia="0.01 0.01 0.01"/>
                    <joint name="knee_R" type="hinge" range="-2.5 0"/>
                    <geom type="capsule" size="0.035" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.6 1 1" friction="1"/>
                    <body name="foot_R" pos="0 -0.3 0">
                        <inertial pos="0 0 0" mass="0.8" diaginertia="0.005 0.005 0.005"/>
                        <geom type="box" size="0.06 0.025 0.1" rgba="0 1 0 1" friction="3"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- 팔 -->
        <motor joint="shoulder_L" ctrllimited="true" ctrlrange="-50 50" gear="100"/>
        <motor joint="elbow_L" ctrllimited="true" ctrlrange="-30 30" gear="50"/>
        <motor joint="shoulder_R" ctrllimited="true" ctrlrange="-50 50" gear="100"/>
        <motor joint="elbow_R" ctrllimited="true" ctrlrange="-30 30" gear="50"/>
        <!-- 다리 -->
        <motor joint="hip_L" ctrllimited="true" ctrlrange="-50 50" gear="100"/>
        <motor joint="knee_L" ctrllimited="true" ctrlrange="-30 30" gear="50"/>
        <motor joint="hip_R" ctrllimited="true" ctrlrange="-50 50" gear="100"/>
        <motor joint="knee_R" ctrllimited="true" ctrlrange="-30 30" gear="50"/>
    </actuator>
</mujoco>
"""


class HumanoidSim:
    """휴머노이드 물리 시뮬"""
    
    def __init__(self, xml=None):
        self.model = MjModel.from_xml_string(xml or HUMANOID_XML)
        self.data = MjData(self.model)
        
        # body IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.handL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_L")
        self.handR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_R")
        self.footL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
        self.footR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
        
        # joint IDs
        self.shoulderL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_L")
        self.shoulderR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_R")
        self.elbowL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_L")
        self.elbowR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_R")
        self.hipL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")
        self.hipR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
        self.kneeL_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_L")
        self.kneeR_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_R")
        
        print(f"모델 로드: {self.model.nq} DOF, {self.model.nu} actuator")
        print(f"관절: shoulder_L={self.shoulderL_id}, elbow_L={self.elbowL_id}")
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [2.0, 0.8, 0.5]  # torso 위치
    
    def step(self, ctrl=None):
        if ctrl is not None:
            self.data.ctrl[:] = np.clip(ctrl, -1, 1)
        mujoco.mj_step(self.model, self.data)
    
    def get_position(self, body_id):
        return self.data.body(body_id).xpos.copy()
    
    def get_joint_angle(self, joint_id):
        return self.data.qpos[joint_id]
    
    def set_joint_angle(self, joint_id, angle):
        self.data.qpos[joint_id] = angle
    
    def get_height(self):
        return self.data.body(self.torso_id).xpos[1]
    
    def apply_control(self, ctrl):
        """제어 신호 적용"""
        self.data.ctrl[:] = np.clip(ctrl, -1, 1)
    
    def get_state(self):
        """상태 반환"""
        return {
            'torso': self.get_position(self.torso_id),
            'head': self.get_position(self.head_id),
            'handL': self.get_position(self.handL_id),
            'handR': self.get_position(self.handR_id),
            'footL': self.get_position(self.footL_id),
            'footR': self.get_position(self.footR_id),
            'height': self.get_height()
        }


def test_simulation():
    """시뮬레이션 테스트"""
    print("=== 휴머노이드 시뮬 테스트 ===")
    
    sim = HumanoidSim()
    sim.reset()
    
    print("\n초기 상태:")
    state = sim.get_state()
    print(f"  토르소 위치: {state['torso']}")
    print(f"  높이: {state['height']:.2f}")
    
    # 중력만으로落下
    print("\n중력 테스트 (100 steps):")
    for i in range(100):
        sim.step()
        if i % 20 == 0:
            print(f"  step {i}: height={sim.get_height():.2f}")
    
    # 팔 제어 테스트
    print("\n팔 제어 테스트:")
    sim.reset()
    
    for i in range(50):
        ctrl = np.zeros(8)
        ctrl[0] = 0.5  # 왼쪽 어깨
        ctrl[1] = -1.0  # 왼쪽 팔굽힘
        sim.apply_control(ctrl)
        sim.step()
    
    print(f"  손 위치: {sim.get_position(sim.handL_id)}")
    
    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    test_simulation()