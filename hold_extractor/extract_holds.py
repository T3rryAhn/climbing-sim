"""
Hold Extractor - GLB 파일에서 홀드 추출
pygltflib 사용
"""

import pygltflib
import numpy as np
import json
import os

def load_glb(path):
    """GLB 파일 로드"""
    print(f"GLB 로드: {path}")
    gltf = pygltflib.GLTF2().load(path)
    return gltf

def extract_holds(gltf):
    """홀드 추출"""
    holds = []
    
    # 메쉬 정보
    for i, mesh in enumerate(gltf.meshes):
        print(f"Mesh {i}: {mesh.name if mesh.name else 'unnamed'}")
        
        # 프리미티브 정보
        for j, primitive in enumerate(mesh.primitives):
            if primitive.attributes.POSITION is not None:
                accessor = gltf.accessors[primitive.attributes.POSITION]
                print(f"  Primitive {j}: {accessor.count} vertices")
    
    return holds

def main():
    glb_path = os.path.join(os.path.dirname(__file__), "climbing-wall-00.glb")
    
    if not os.path.exists(glb_path):
        print(f"GLB 파일 없음: {glb_path}")
        return
    
    try:
        gltf = load_glb(glb_path)
        holds = extract_holds(gltf)
        print(f"\n추출된 홀드: {len(holds)}개")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()