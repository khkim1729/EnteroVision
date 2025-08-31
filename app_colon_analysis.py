"""
EnteroVision v2 - Colon Analysis Application
CT colonography를 위한 대장 전문 시각화 및 분석 도구
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys
from pathlib import Path
import traceback

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.totalsegmentator_wrapper import process_ct_for_visualization
from src.colon_cpr_visualizer import ColonCPRVisualizer
from src.volume_renderer import CTSliceViewer

# 페이지 설정
st.set_page_config(
    page_title="EnteroVision v2 - Colon Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 데이터 경로 설정
DATASETS_PATH = Path(__file__).parent.parent / 'datasets'
CT_IMAGES_PATH = DATASETS_PATH / 'ct_images'

def get_available_ct_files():
    """사용 가능한 CT 파일 목록 반환"""
    ct_files = []
    if CT_IMAGES_PATH.exists():
        for file in CT_IMAGES_PATH.iterdir():
            if file.suffix in ['.nii', '.gz'] and 'image' in file.name:
                ct_files.append(file)
    return sorted(ct_files)

def main():
    st.title("🔍 EnteroVision v2 - Colon Analysis & CT Colonography")
    st.markdown("**Advanced colon visualization with Curved Planar Reformation (CPR)**")
    st.markdown("---")
    
    # 사이드바 - 파일 선택 및 설정
    with st.sidebar:
        st.header("📁 Data Selection")
        
        # CT 파일 선택
        available_files = get_available_ct_files()
        
        if not available_files:
            st.error("CT 이미지 파일을 찾을 수 없습니다.")
            st.info(f"다음 경로에 CT 파일을 추가하세요: {CT_IMAGES_PATH}")
            return
        
        file_names = [f.name for f in available_files]
        selected_file_idx = st.selectbox(
            "CT Image File",
            range(len(available_files)),
            format_func=lambda x: file_names[x],
            help="분석할 CT 영상을 선택하세요"
        )
        
        selected_file = available_files[selected_file_idx]
        
        st.info(f"선택된 파일: {selected_file.name}")
        
        # 분석 옵션
        st.header("⚙️ Analysis Options")
        
        use_totalsegmentator = st.checkbox(
            "TotalSegmentator 자동 분할", 
            value=True,
            help="대장 및 주변 장기를 자동으로 분할합니다"
        )
        
        # CPR 설정
        st.header("🌀 CPR Settings")
        
        cpr_width = st.slider(
            "CPR 너비 (픽셀)",
            min_value=50,
            max_value=200,
            value=100,
            help="Curved Planar Reformation의 너비를 설정합니다"
        )
        
        # 시각화 옵션
        st.header("🎨 Visualization Options")
        
        show_centerline = st.checkbox(
            "중심선 표시",
            value=True,
            help="대장 중심선을 3D 뷰에 표시합니다"
        )
        
        colon_opacity = st.slider(
            "대장 투명도",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            help="3D 대장 표면의 투명도를 조절합니다"
        )
        
        # 처리 시작 버튼
        process_button = st.button("🚀 대장 분석 시작", type="primary")
    
    # 메인 영역
    if process_button:
        with st.spinner("CT 데이터 처리 중... 대장 분할 및 중심선 추출 중입니다."):
            try:
                # CT 데이터 처리
                results = process_ct_for_visualization(str(selected_file))
                
                if results is None:
                    st.error("CT 데이터 처리에 실패했습니다.")
                    return
                
                # 대장 검출 확인
                if 'colon' not in results['visualization_data']:
                    st.warning("⚠️ 대장이 검출되지 않았습니다. 다른 CT 파일을 시도해보세요.")
                    st.info("사용 가능한 장기: " + ", ".join(results['visualization_data'].keys()))
                    return
                
                # 세션 상태에 결과 저장
                st.session_state['results'] = results
                st.session_state['selected_file'] = selected_file
                st.session_state['cpr_width'] = cpr_width
                
                st.success("✅ CT 데이터 처리 완료! 대장이 성공적으로 검출되었습니다.")
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {str(e)}")
                st.code(traceback.format_exc())
                return
    
    # 처리된 데이터가 있는 경우 시각화 표시
    if 'results' in st.session_state:
        results = st.session_state['results']
        cpr_width = st.session_state.get('cpr_width', 100)
        display_colon_analysis(results, cpr_width, show_centerline, colon_opacity)

def display_colon_analysis(results, cpr_width, show_centerline, colon_opacity):
    """대장 분석 결과 표시"""
    
    st.header("📊 Colon Analysis Results")
    
    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "CT Volume", 
            f"{results['original_ct'].shape[0]}×{results['original_ct'].shape[1]}×{results['original_ct'].shape[2]}"
        )
    
    with col2:
        colon_voxels = np.sum(results['visualization_data']['colon']['mask'])
        st.metric("Colon Voxels", f"{colon_voxels:,}")
    
    with col3:
        colon_volume = colon_voxels * 0.5  # 0.5mm³ per voxel 가정
        st.metric("Estimated Volume", f"{colon_volume:.1f} ml")
    
    with col4:
        detected_organs = len(results['visualization_data'])
        st.metric("Total Organs", detected_organs)
    
    # 탭으로 다양한 뷰 제공
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Colon View", 
        "CPR Analysis", 
        "CT Slices", 
        "Centerline Analysis",
        "Report"
    ])
    
    with tab1:
        st.subheader("🎯 3D Colon Visualization")
        
        # CPR 시각화 객체 생성
        cpr_visualizer = ColonCPRVisualizer()
        cpr_visualizer.load_data(
            results['original_ct'],
            results['visualization_data']['colon']['mask']
        )
        
        # 중심선 추출
        with st.spinner("대장 중심선 추출 중..."):
            centerline = cpr_visualizer.extract_colon_centerline()
        
        if centerline is not None and len(centerline) > 0:
            st.session_state['cpr_visualizer'] = cpr_visualizer
            
            # 3D 플롯 생성
            fig_3d = cpr_visualizer.create_colon_3d_plot()
            
            # 투명도 조정
            for trace in fig_3d.data:
                if hasattr(trace, 'opacity'):
                    trace.opacity = colon_opacity
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 중심선 정보
            st.info(f"✅ 중심선 추출 성공: {len(centerline)}개 점")
            st.info(f"📏 추정 대장 길이: {len(centerline) * 0.5:.1f} mm")
            
        else:
            st.error("❌ 중심선 추출에 실패했습니다. 대장 구조가 복잡하거나 분할 품질이 낮을 수 있습니다.")
    
    with tab2:
        st.subheader("🌀 Curved Planar Reformation (CPR)")
        
        if 'cpr_visualizer' in st.session_state:
            cpr_visualizer = st.session_state['cpr_visualizer']
            
            # CPR 이미지 생성
            with st.spinner(f"CPR 이미지 생성 중 (너비: {cpr_width}픽셀)..."):
                cpr_image = cpr_visualizer.create_cpr_image(width=cpr_width)
            
            if cpr_image is not None:
                # CPR 플롯 표시
                fig_cpr = cpr_visualizer.create_interactive_cpr_plot()
                st.plotly_chart(fig_cpr, use_container_width=True)
                
                # CPR 통계
                st.subheader("📈 CPR Image Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Image Size", f"{cpr_image.shape[0]}×{cpr_image.shape[1]}")
                with col2:
                    st.metric("HU Range", f"{np.min(cpr_image):.0f} to {np.max(cpr_image):.0f}")
                with col3:
                    st.metric("Mean HU", f"{np.mean(cpr_image):.1f}")
                
                # 다중 뷰
                st.subheader("📊 Combined 3D and CPR View")
                fig_multi = cpr_visualizer.create_multi_view_plot()
                st.plotly_chart(fig_multi, use_container_width=True)
                
            else:
                st.error("❌ CPR 이미지 생성에 실패했습니다.")
        else:
            st.warning("먼저 3D Colon View 탭에서 중심선을 추출하세요.")
    
    with tab3:
        st.subheader("🔍 CT Slice Viewer")
        
        # 슬라이스 뷰어
        slice_viewer = CTSliceViewer(
            results['original_ct'],
            {'colon': results['visualization_data']['colon']}
        )
        
        # 축 선택
        axis = st.selectbox(
            "View Axis",
            ['axial', 'sagittal', 'coronal'],
            help="볼 축을 선택하세요"
        )
        
        # 슬라이스 인덱스 선택
        max_idx = results['original_ct'].shape[0 if axis == 'axial' else 1]
        slice_idx = st.slider(
            f"Slice Index ({axis})",
            0, max_idx - 1,
            max_idx // 2
        )
        
        # 슬라이스 플롯 생성
        fig_slice = slice_viewer.create_slice_plot(axis, slice_idx)
        st.plotly_chart(fig_slice, use_container_width=True)
        
        # 해당 슬라이스의 대장 정보
        if axis == 'axial':
            colon_slice = results['visualization_data']['colon']['mask'][slice_idx, :, :]
        elif axis == 'sagittal':
            colon_slice = results['visualization_data']['colon']['mask'][:, slice_idx, :]
        else:  # coronal
            colon_slice = results['visualization_data']['colon']['mask'][:, :, slice_idx]
        
        colon_pixels = np.sum(colon_slice)
        st.info(f"이 슬라이스의 대장 픽셀 수: {colon_pixels}")
    
    with tab4:
        st.subheader("📏 Centerline Analysis")
        
        if 'cpr_visualizer' in st.session_state:
            cpr_visualizer = st.session_state['cpr_visualizer']
            centerline = cpr_visualizer.centerline
            
            if centerline is not None and len(centerline) > 0:
                # 중심선 통계
                st.write("**Centerline Statistics:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"- 총 점 개수: {len(centerline)}")
                    st.write(f"- 추정 길이: {len(centerline) * 0.5:.1f} mm")
                    
                    # 중심선 곡률 분석
                    if len(centerline) > 2:
                        curvatures = []
                        for i in range(1, len(centerline)-1):
                            p1 = np.array(centerline[i-1])
                            p2 = np.array(centerline[i])
                            p3 = np.array(centerline[i+1])
                            
                            # 간단한 곡률 계산
                            v1 = p2 - p1
                            v2 = p3 - p2
                            
                            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                                angle = np.arccos(np.clip(
                                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 
                                    -1, 1
                                ))
                                curvatures.append(angle)
                        
                        if curvatures:
                            st.write(f"- 평균 곡률: {np.mean(curvatures):.3f} rad")
                            st.write(f"- 최대 곡률: {np.max(curvatures):.3f} rad")
                
                with col2:
                    # 중심선 3D 좌표 분포
                    centerline_array = np.array(centerline)
                    st.write("**좌표 범위:**")
                    st.write(f"- Z: {np.min(centerline_array[:,0]):.1f} ~ {np.max(centerline_array[:,0]):.1f}")
                    st.write(f"- Y: {np.min(centerline_array[:,1]):.1f} ~ {np.max(centerline_array[:,1]):.1f}")
                    st.write(f"- X: {np.min(centerline_array[:,2]):.1f} ~ {np.max(centerline_array[:,2]):.1f}")
                
                # 중심선 점들의 플롯
                fig_centerline = go.Figure()
                
                fig_centerline.add_trace(go.Scatter3d(
                    x=centerline_array[:, 2],
                    y=centerline_array[:, 1],
                    z=centerline_array[:, 0],
                    mode='lines+markers',
                    line=dict(color='red', width=6),
                    marker=dict(size=2),
                    name='Centerline'
                ))
                
                fig_centerline.update_layout(
                    title="Colon Centerline in 3D Space",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode='data'
                    ),
                    height=500
                )
                
                st.plotly_chart(fig_centerline, use_container_width=True)
                
            else:
                st.warning("중심선 데이터가 없습니다.")
        else:
            st.warning("먼저 3D Colon View 탭에서 중심선을 추출하세요.")
    
    with tab5:
        st.subheader("📋 Analysis Report")
        
        # 상세 보고서
        st.write("**File Information:**")
        st.write(f"- Original CT: {st.session_state['selected_file'].name}")
        st.write(f"- Segmentation: {Path(results['segmentation_path']).name}")
        st.write(f"- Processing date: {st.session_state.get('processing_date', 'Unknown')}")
        
        st.write("**Colon Analysis Summary:**")
        colon_data = results['visualization_data']['colon']
        colon_mask = colon_data['mask']
        colon_voxels = np.sum(colon_mask)
        
        st.write(f"- Colon detected: ✅ Yes")
        st.write(f"- Voxel count: {colon_voxels:,}")
        st.write(f"- Estimated volume: {colon_voxels * 0.5:.1f} ml")
        st.write(f"- Label ID: {colon_data['label_id']}")
        
        # 대장 영역의 HU 값 분석
        colon_hu_values = results['original_ct'][colon_mask]
        st.write(f"- HU statistics: {np.min(colon_hu_values):.0f} / {np.mean(colon_hu_values):.0f} / {np.max(colon_hu_values):.0f} (min/mean/max)")
        
        if 'cpr_visualizer' in st.session_state:
            cpr_visualizer = st.session_state['cpr_visualizer']
            if cpr_visualizer.centerline is not None:
                st.write(f"- Centerline points: {len(cpr_visualizer.centerline)}")
                st.write(f"- Estimated colon length: {len(cpr_visualizer.centerline) * 0.5:.1f} mm")
        
        # 다른 검출된 장기들
        other_organs = [name for name in results['visualization_data'].keys() if name != 'colon']
        if other_organs:
            st.write(f"- Other detected organs: {', '.join(other_organs)}")
        
        # 보고서 다운로드
        if st.button("Generate Detailed Report"):
            report = generate_colon_report(results, st.session_state.get('cpr_visualizer'))
            st.download_button(
                label="📥 Download Colon Analysis Report",
                data=report,
                file_name=f"colon_analysis_report_{st.session_state['selected_file'].stem}.txt",
                mime="text/plain"
            )

def generate_colon_report(results, cpr_visualizer=None):
    """대장 분석 보고서 생성"""
    report = []
    report.append("=" * 60)
    report.append("EnteroVision v2 - Colon Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    # 기본 정보
    report.append("## File Information")
    report.append(f"- Segmentation file: {Path(results['segmentation_path']).name}")
    report.append(f"- CT volume shape: {results['original_ct'].shape}")
    report.append(f"- HU value range: {np.min(results['original_ct']):.0f} to {np.max(results['original_ct']):.0f}")
    report.append("")
    
    # 대장 분석
    report.append("## Colon Analysis")
    colon_data = results['visualization_data']['colon']
    colon_mask = colon_data['mask']
    colon_voxels = np.sum(colon_mask)
    colon_hu_values = results['original_ct'][colon_mask]
    
    report.append(f"- Colon detection: SUCCESS")
    report.append(f"- Voxel count: {colon_voxels:,}")
    report.append(f"- Estimated volume: {colon_voxels * 0.5:.1f} ml")
    report.append(f"- TotalSegmentator label ID: {colon_data['label_id']}")
    report.append(f"- HU statistics:")
    report.append(f"  * Mean: {np.mean(colon_hu_values):.1f}")
    report.append(f"  * Standard deviation: {np.std(colon_hu_values):.1f}")
    report.append(f"  * Range: {np.min(colon_hu_values):.0f} to {np.max(colon_hu_values):.0f}")
    report.append("")
    
    # 중심선 분석
    if cpr_visualizer and cpr_visualizer.centerline is not None:
        centerline = cpr_visualizer.centerline
        report.append("## Centerline Analysis")
        report.append(f"- Centerline extraction: SUCCESS")
        report.append(f"- Number of points: {len(centerline)}")
        report.append(f"- Estimated colon length: {len(centerline) * 0.5:.1f} mm")
        
        # 곡률 분석
        if len(centerline) > 2:
            curvatures = []
            for i in range(1, len(centerline)-1):
                p1 = np.array(centerline[i-1])
                p2 = np.array(centerline[i])
                p3 = np.array(centerline[i+1])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 
                        -1, 1
                    ))
                    curvatures.append(angle)
            
            if curvatures:
                report.append(f"- Average curvature: {np.mean(curvatures):.3f} radians")
                report.append(f"- Maximum curvature: {np.max(curvatures):.3f} radians")
        
        # CPR 정보
        if cpr_visualizer.cpr_image is not None:
            cpr_image = cpr_visualizer.cpr_image
            report.append("## CPR Analysis")
            report.append(f"- CPR image generation: SUCCESS")
            report.append(f"- CPR image size: {cpr_image.shape[0]} x {cpr_image.shape[1]} pixels")
            report.append(f"- CPR HU range: {np.min(cpr_image):.0f} to {np.max(cpr_image):.0f}")
            report.append(f"- CPR mean HU: {np.mean(cpr_image):.1f}")
        
        report.append("")
    else:
        report.append("## Centerline Analysis")
        report.append("- Centerline extraction: FAILED")
        report.append("- Possible reasons: Complex colon structure, poor segmentation quality")
        report.append("")
    
    # 기타 검출된 장기
    other_organs = [name for name in results['visualization_data'].keys() if name != 'colon']
    if other_organs:
        report.append("## Other Detected Organs")
        for organ_name in other_organs:
            organ_data = results['visualization_data'][organ_name]
            organ_voxels = np.sum(organ_data['mask'])
            report.append(f"- {organ_name}: {organ_voxels:,} voxels ({organ_voxels * 0.5:.1f} ml)")
        report.append("")
    
    # 요약 및 권장사항
    report.append("## Summary and Recommendations")
    if colon_voxels > 10000:  # 충분한 대장 볼륨
        report.append("✓ Good colon segmentation quality - sufficient volume detected")
    else:
        report.append("⚠ Limited colon segmentation - consider manual review")
    
    if cpr_visualizer and cpr_visualizer.centerline is not None:
        report.append("✓ Centerline extraction successful - CPR analysis available")
    else:
        report.append("⚠ Centerline extraction failed - CPR analysis not available")
    
    report.append("")
    report.append("## Technical Notes")
    report.append("- Segmentation performed using TotalSegmentator v2")
    report.append("- CPR generated using custom curved planar reformation algorithm")
    report.append("- Volume estimation based on 0.5mm³ per voxel assumption")
    report.append("- This report is for research purposes only")
    
    return '\n'.join(report)

if __name__ == "__main__":
    main()