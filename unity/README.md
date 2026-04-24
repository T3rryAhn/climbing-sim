# Unity 경로 재생

## 파일 구조

```
unity/
├── ClimbingPathPlayer.cs    # 경로 재생 스크립트
├── StreamingAssets/          # JSON 데이터
│   └── climbing_result_stance.json
└── README.md                # 이 파일
```

## Unity 설정

### 1. 프로젝트 생성
- Unity 2022 이상
- Build Platform: Windows

### 2. 스크립트 추가
1. 빈 GameObject 생성 (이름: `ClimbingPlayer`)
2. `ClimbingPathPlayer.cs` 컴포넌트 추가

### 3. JSON 파일 복사
- `StreamingAssets/` 폴더에 `climbing_result_stance.json` 복사

### 4. 실행
- Play 버튼 클릭
- 아바타가 경로 따라 이동

## 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| Move Speed | 이동 속도 | 1 |
| Hold Size | 홀드 크기 | 0.1 |

## 홀드 색상

- 빨강: 시작 홀드 (start)
- 주황: 일반 홀드 (normal)
- 초록: 목표 홀드 (goal)