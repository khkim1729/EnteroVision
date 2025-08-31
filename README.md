# EnteroVision v2

**Advanced CT-based Intestinal Analysis and Visualization Platform**

EnteroVision v2는 CT 데이터 기반 장협착증, 장유착 및 장폐쇄 조기 진단을 위한 차세대 시각화 플랫폼입니다. TotalSegmentator AI 기술을 활용하여 자동 장기 분할과 고급 시각화 기능을 제공합니다.

## 🌟 주요 기능

### 1. 소장(Small Bowel) 전용 분석 - `app_small_bowel.py`
- **TotalSegmentator 자동 분할**: AI 기반 소장 및 주변 장기 자동 검출
- **3D 시각화**: 소장과 주변 장기의 실시간 3D 렌더링
- **Straightened View**: 구불구불한 소장을 일자로 펼친 2D 뷰
- **다중 축 CT 슬라이스**: Axial, Sagittal, Coronal 뷰 지원
- **상세 분석 리포트**: 장기별 통계 및 HU 값 분석

### 2. 대장(Colon) 전용 분석 - `app_colon_analysis.py`
- **CT Colonography**: 대장 내시경술을 위한 전문 시각화
- **Curved Planar Reformation (CPR)**: 대장을 따라 펼친 고급 뷰
- **중심선 추출**: 대장의 3D 중심선 자동 추출 및 시각화
- **곡률 분석**: 대장 구조의 정량적 분석
- **멀티뷰 시각화**: 3D와 CPR을 동시에 표시

## 🔬 기술적 장점

### 소장 vs 대장 분석 비교

| 특성 | 소장 (Small Bowel) | 대장 (Colon) |
|------|-------------------|--------------|
| **TotalSegmentator 정확도** | 제한적 | 높음 ⭐ |
| **중심선 추출** | 어려움 | 안정적 ⭐ |
| **시각화 품질** | 보통 | 우수 ⭐ |
| **임상 적용** | 연구 단계 | 검증됨 ⭐ |
| **CPR 지원** | 기본 | 고급 ⭐ |

**권장사항**: 안정적인 결과가 필요한 경우 대장 분석 애플리케이션 사용을 권장합니다.

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# Python 3.9+ 환경에서 설치
pip install -r requirements.txt

# TotalSegmentator 설치 (GPU 권장)
pip install TotalSegmentator

# GPU 사용을 위한 PyTorch 설치 (선택사항)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 데이터 준비

CT 이미지 파일(.nii.gz)을 다음 경로에 배치:
```
EnteroVision_v002/
├── datasets/              # EnteroVision_v001과 공유
│   ├── ct_images/         # CT 원본 이미지 (.nii.gz)
│   └── ct_labels/         # TotalSegmentator 분할 결과 (자동 생성)
```

### 3. 애플리케이션 실행

#### 소장 분석 애플리케이션
```bash
cd EnteroVision_v002
streamlit run app_small_bowel.py --server.port 8501
```

#### 대장 분석 애플리케이션  
```bash
cd EnteroVision_v002
streamlit run app_colon_analysis.py --server.port 8502
```

브라우저에서 각각 `http://localhost:8501`, `http://localhost:8502`로 접속

## 🚀 사용 방법

### 소장 분석 워크플로우

1. **데이터 선택**: 사이드바에서 CT 이미지 파일 선택
2. **처리 옵션 설정**: TotalSegmentator 자동 분할 활성화
3. **분석 시작**: "🚀 분석 시작" 버튼 클릭 (2-5분 소요)
4. **결과 확인**:
   - **3D Visualization**: 소장 및 주변 장기의 3D 뷰
   - **CT Slices**: 다축 CT 슬라이스 뷰어
   - **Straightened View**: 펼친 소장 구조
   - **Analysis**: 상세 통계 및 보고서

### 대장 분석 워크플로우

1. **데이터 선택**: CT 이미지 및 CPR 설정 조정
2. **대장 분석 시작**: TotalSegmentator로 대장 자동 검출
3. **결과 분석**:
   - **3D Colon View**: 대장 표면과 중심선 3D 시각화
   - **CPR Analysis**: Curved Planar Reformation 뷰
   - **CT Slices**: 대장이 강조된 슬라이스 뷰
   - **Centerline Analysis**: 중심선 통계 및 곡률 분석
   - **Report**: 종합 분석 보고서

## 📊 결과 해석

### 3D 시각화
- **장기별 색상 구분**: 각 장기는 고유 색상으로 표시
- **투명도 조절**: 내부 구조 확인을 위한 투명도 설정
- **중심선**: 빨간색 선으로 장기의 중심 경로 표시

### CPR (Curved Planar Reformation)
- **Y축**: 중심선을 따른 위치 (근위부→원위부)
- **X축**: 중심선에 수직인 너비
- **색상**: HU 값에 따른 그레이스케일 (뼈: 밝음, 공기: 어둠)

### 통계 정보
- **Voxel Count**: 검출된 장기의 복셀 개수
- **Volume**: 추정 부피 (0.5mm³/voxel 가정)
- **HU Statistics**: 장기 내 CT 값 분포 통계

## ⚠️ 주의사항

### 소장 분석 한계
- TotalSegmentator의 소장 검출 정확도가 제한적
- 복잡한 소장 구조로 인한 중심선 추출 어려움
- 장협착증 검출 기능은 연구 단계

### 대장 분석 권장사항
- 대조제 사용 CT에서 최적 성능
- 대장 정결도가 결과에 영향
- CPR 분석은 중심선 품질에 의존

### 일반적 주의사항
- **연구 목적 전용**: 임상 진단 목적 사용 금지
- **데이터 보안**: 환자 정보 보호 준수
- **GPU 권장**: TotalSegmentator는 GPU 사용 시 빠른 처리

## 🔧 기술 스택

- **AI 분할**: TotalSegmentator v2 (104개 해부학 구조)
- **3D 시각화**: Plotly + Marching Cubes
- **UI 프레임워크**: Streamlit
- **영상 처리**: SimpleITK, scikit-image
- **수치 연산**: NumPy, SciPy

## 📁 프로젝트 구조

```
EnteroVision_v002/
├── README.md                    # 이 파일
├── requirements.txt             # Python 의존성
├── app_small_bowel.py          # 소장 분석 메인 앱
├── app_colon_analysis.py       # 대장 분석 메인 앱
├── src/                        # 핵심 모듈들
│   ├── totalsegmentator_wrapper.py   # TotalSegmentator 래퍼
│   ├── volume_renderer.py            # 3D 볼륨 렌더링
│   └── colon_cpr_visualizer.py       # CPR 시각화
├── static/                     # 정적 파일 (이미지 등)
├── templates/                  # 템플릿 파일
└── datasets/                   # 데이터셋 (v001과 공유)
    ├── ct_images/              # CT 원본 파일
    └── ct_labels/              # 분할 결과
```

## 🔬 연구 배경

### TotalSegmentator 성능 (2024년 연구 기준)
- **대장 분할**: 높은 Dice coefficient (>0.9)
- **소장 분할**: 제한적 성능, 작은 구조로 인한 어려움
- **CT colonography**: 기존 연구에서 검증된 시각화 방법

### CPR 기술
- **Curved Planar Reformation**: 구부러진 구조를 평면으로 전개
- **CT colonography 표준**: 대장내시경 전처치로 널리 사용
- **폴립 검출**: CPR 뷰에서 폴립 검출률 향상

## 🤝 기여 및 개발

### 개발 환경 설정
```bash
git clone <repository>
cd EnteroVision_v002
pip install -r requirements.txt
streamlit run app_small_bowel.py
```

### 향후 개발 계획
- [ ] DICOM 파일 직접 지원
- [ ] AI 기반 이상 소견 자동 검출
- [ ] 다중 환자 배치 처리
- [ ] 웹 기반 PACS 연동
- [ ] 모바일 뷰어 지원

## 📜 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다. 상업적 사용 시 별도 라이선스가 필요할 수 있습니다.

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.

---

**EnteroVision v2** - Advanced CT-based Intestinal Analysis Platform
*Powered by TotalSegmentator AI & Modern Visualization Technologies*