"""
EnteroVision v2 - Small Bowel Analysis Application
CT 데이터 기반 소장 시각화 및 분석 도구 (Streamlit 버전)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys
import time
from pathlib import Path
import traceback

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.totalsegmentator_wrapper import process_ct_for_visualization
from src.volume_renderer import VolumeRenderer3D, CTSliceViewer
from src.ui_logger import ui_logger

# 페이지 설정
st.set_page_config(
    page_title="EnteroVision v2 - Small Bowel Analysis",
    page_icon="🫁",
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
    st.title("🫁 EnteroVision v2 - Small Bowel Analysis")
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
        
        # 처리 옵션
        st.header("⚙️ Processing Options")
        
        auto_segment = st.checkbox(
            "TotalSegmentator 자동 분할 사용", 
            value=True,
            help="소장 및 주변 장기를 자동으로 분할합니다"
        )
        
        # 시각화 옵션
        st.header("🎨 Visualization Options")
        
        selected_organs = st.multiselect(
            "표시할 장기 선택",
            ['small_bowel', 'colon', 'stomach', 'liver', 'pancreas', 'spleen'],
            default=['small_bowel', 'colon'],
            help="3D 뷰에 표시할 장기들을 선택하세요"
        )
        
        show_ct_slices = st.checkbox(
            "CT 슬라이스 표시",
            value=False,
            help="3D 뷰에 CT 슬라이스를 반투명하게 표시합니다"
        )
        
        # 처리 시작 버튼
        process_button = st.button("🚀 분석 시작", type="primary")
        
        # 실시간 로그 섹션
        st.header("📊 실시간 처리 로그")
        
        # 로그 토글
        show_realtime_logs = st.checkbox(
            "실시간 로그 보기", 
            value=True,
            help="처리 중 실시간으로 로그를 확인할 수 있습니다"
        )
        
        if show_realtime_logs:
            # 로그 요약
            log_summary = ui_logger.get_log_summary()
            st.write(f"**로그 상태:** {log_summary}")
            
            # 실시간 로그 컨테이너
            log_container = st.container()
            
            with log_container:
                if ui_logger.get_logs():
                    # 스크롤 가능한 로그 영역
                    ui_logger.display_realtime_logs(log_container, max_lines=15)
                else:
                    st.info("📋 분석을 시작하면 여기에 실시간 로그가 표시됩니다.")
        else:
            st.info("실시간 로그가 비활성화되었습니다.")
    
    # 메인 영역
    if process_button:
        # 로그 초기화
        ui_logger.clear()
        
        # 실시간 로그 컨테이너 생성 (사이드바용)
        sidebar_log_container = st.sidebar.empty()
        
        # 진행상황 표시용 컨테이너
        progress_container = st.empty()
        
        def update_sidebar_logs():
            """사이드바 로그 실시간 업데이트"""
            if show_realtime_logs:
                with sidebar_log_container.container():
                    if ui_logger.get_logs():
                        st.write("**📋 실시간 처리 상황**")
                        # 최근 5개 로그만 표시 (사이드바 공간 절약)
                        recent_logs = ui_logger.get_logs()[-5:]
                        for log in reversed(recent_logs):
                            emoji = ui_logger._get_emoji(log['level'])
                            if log['level'] == "ERROR":
                                st.error(f"`{log['timestamp']}` {emoji} {log['message']}")
                            elif log['level'] == "WARNING":
                                st.warning(f"`{log['timestamp']}` {emoji} {log['message']}")
                            elif log['level'] == "SUCCESS":
                                st.success(f"`{log['timestamp']}` {emoji} {log['message']}")
                            else:
                                st.info(f"`{log['timestamp']}` {emoji} {log['message']}")
        
        try:
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # CT 데이터 처리 시작
            ui_logger.log("🚀 CT 데이터 처리 시작", "INFO")
            ui_logger.log(f"📁 선택된 파일: {selected_file.name}", "INFO")
            update_sidebar_logs()
            time.sleep(0.1)  # UI 업데이트 대기
            
            with st.spinner("CT 데이터 처리 중... 몇 분 소요될 수 있습니다."):
                status_text.text("🚀 TotalSegmentator 실행 중...")
                progress_bar.progress(20)
                
                ui_logger.log("🧠 TotalSegmentator AI 모델 로딩 중...", "INFO")
                update_sidebar_logs()
                time.sleep(0.1)  # UI 업데이트 대기
                
                results = process_ct_for_visualization(str(selected_file))
                
                progress_bar.progress(80)
                status_text.text("🎯 시각화 데이터 준비 중...")
                
                ui_logger.log("🎨 3D 시각화 데이터 생성 중...", "INFO")
                update_sidebar_logs()
                time.sleep(0.1)  # UI 업데이트 대기
                
                if results is None:
                    ui_logger.log("❌ CT 데이터 처리 실패", "ERROR")
                    update_sidebar_logs()
                    st.error("CT 데이터 처리에 실패했습니다.")
                    
                    # 에러 로그 표시
                    with st.expander("🚨 에러 로그", expanded=True):
                        ui_logger.display_logs(show_all=True)
                    return
                
                # 세션 상태에 결과 저장
                st.session_state['results'] = results
                st.session_state['selected_file'] = selected_file
                
                progress_bar.progress(100)
                status_text.text("✅ 처리 완료!")
                
                ui_logger.log("🎉 CT 데이터 처리 완료!", "SUCCESS")
                update_sidebar_logs()
                st.success("✅ CT 데이터 처리 완료!")
                
        except Exception as e:
            ui_logger.log(f"❌ 처리 중 오류: {str(e)}", "ERROR")
            ui_logger.log(f"🔍 상세 오류: {traceback.format_exc()}", "DEBUG")
            
            st.error(f"처리 중 오류 발생: {str(e)}")
            
            # 에러 로그 표시
            with st.expander("🚨 에러 로그 및 디버그 정보", expanded=True):
                ui_logger.display_logs(show_all=True)
            return
        
        # 처리 로그 표시
        if ui_logger.get_logs():
            with st.expander("📋 처리 로그 보기", expanded=False):
                ui_logger.display_logs(show_all=True)
                
                # 로그 다운로드 버튼
                log_text = "\n".join([f"[{log['timestamp']}] {log['level']}: {log['message']}" 
                                     for log in ui_logger.get_logs()])
                st.download_button(
                    label="📥 로그 다운로드",
                    data=log_text,
                    file_name=f"enterovision_log_{selected_file.stem}.txt",
                    mime="text/plain"
                )
    
    # 처리된 데이터가 있는 경우 시각화 표시
    if 'results' in st.session_state:
        results = st.session_state['results']
        display_visualization(results, selected_organs, show_ct_slices)

def display_visualization(results, selected_organs, show_ct_slices):
    """시각화 결과 표시"""
    
    st.header("📊 Analysis Results")
    
    # 분석 요약
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "CT Volume Shape", 
            f"{results['original_ct'].shape[0]}×{results['original_ct'].shape[1]}×{results['original_ct'].shape[2]}"
        )
    
    with col2:
        detected_organs = len(results['visualization_data'])
        st.metric("Detected Organs", detected_organs)
    
    with col3:
        file_size = os.path.getsize(results['segmentation_path']) / (1024 * 1024)
        st.metric("Segmentation Size", f"{file_size:.1f} MB")
    
    # 검출된 장기 목록 표시
    st.info(f"🔍 검출된 장기: {', '.join(results['visualization_data'].keys())}")
    
    # 탭으로 다양한 뷰 제공
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["3D Visualization", "All Organs 3D", "CT Slices", "Straightened View", "Analysis"])
    
    with tab1:
        st.subheader("🎯 3D Volume Rendering")
        
        # 장기 선택 관리
        available_organs = list(results['visualization_data'].keys())
        known_organs = [name for name in available_organs if not name.startswith('unknown_label_')]
        unknown_organs = [name for name in available_organs if name.startswith('unknown_label_')]
        
        # 선택 도구
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🔍 전체 선택", help="모든 장기 선택"):
                st.session_state['display_organs_tab1'] = available_organs
        
        with col2:
            if st.button("❌ 전체 해제", help="모든 장기 선택 해제"):
                st.session_state['display_organs_tab1'] = []
        
        with col3:
            if st.button("🔬 알려진 장기만", help="알려진 장기들만 선택"):
                st.session_state['display_organs_tab1'] = known_organs
                
        with col4:
            if st.button("❓ Unknown만", help="알 수 없는 장기들만 선택"):
                st.session_state['display_organs_tab1'] = unknown_organs
        
        # 장기 정보 표시
        st.write(f"**사용 가능한 장기:** 총 {len(available_organs)}개 (알려진: {len(known_organs)}개, Unknown: {len(unknown_organs)}개)")
        
        # Unknown 장기들이 있으면 별도 표시
        if unknown_organs:
            with st.expander(f"❓ 알 수 없는 장기들 ({len(unknown_organs)}개)", expanded=False):
                unknown_info = []
                for organ_name in unknown_organs:
                    mask = results['visualization_data'][organ_name]['mask']
                    voxel_count = np.sum(mask)
                    label_id = results['visualization_data'][organ_name]['label_id']
                    unknown_info.append(f"• **{organ_name}** (Label ID: {label_id}): {voxel_count:,} 복셀")
                
                st.markdown("  \n".join(unknown_info))
                
                if st.button("🎯 Unknown 장기들 한번에 보기", key="show_unknowns"):
                    st.session_state['display_organs_tab1'] = unknown_organs
        
        # 다중 선택 박스
        current_selection = st.session_state.get('display_organs_tab1', available_organs[:3] if len(available_organs) >= 3 else available_organs)
        
        display_organs = st.multiselect(
            "표시할 장기를 선택하세요:",
            available_organs,
            default=current_selection,
            key="display_organs_tab1",
            help=f"총 {len(available_organs)}개 장기 중에서 선택"
        )
        
        # CT 슬라이스 표시 옵션
        show_slices_tab = st.checkbox("CT 슬라이스 함께 표시", value=show_ct_slices, key="show_slices_tab1")
        
        if display_organs:
            # 선택된 장기들만 필터링
            filtered_organs = {
                k: v for k, v in results['visualization_data'].items() 
                if k in display_organs
            }
            
            # 3D 렌더러 생성
            volume_renderer = VolumeRenderer3D()
            volume_renderer.load_data(
                results['original_ct'],
                filtered_organs
            )
            
            # 3D 플롯 생성
            fig_3d = volume_renderer.create_interactive_plot(
                selected_organs=list(filtered_organs.keys()),
                show_slices=show_slices_tab
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 장기별 상세 정보
            st.subheader("📋 선택된 장기 정보")
            
            # 장기 정보를 컬럼으로 나누어 표시
            num_organs = len(filtered_organs)
            cols = st.columns(min(num_organs, 3))  # 최대 3개 컬럼
            
            for idx, (organ_name, data) in enumerate(filtered_organs.items()):
                mask = data['mask']
                voxel_count = np.sum(mask)
                volume_ml = voxel_count * 0.5  # 가정: 0.5mm³ per voxel
                
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"**{organ_name.replace('_', ' ').title()}**")
                    st.write(f"🔢 Voxel: {voxel_count:,}")
                    st.write(f"📏 Volume: {volume_ml:.1f} ml")
                    st.write(f"🏷️ Label ID: {data['label_id']}")
                    
                    # 색상 표시
                    color = data['color']
                    color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                    st.markdown(f'<div style="background-color: {color_hex}; height: 20px; border-radius: 10px;"></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ 표시할 장기를 선택하세요.")
            
            # 사용 가능한 장기 목록 표시
            st.info("사용 가능한 장기: " + ", ".join(available_organs))
    
    with tab2:
        st.subheader("🧠 TotalSegmentator - All Detected Organs 3D Visualization")
        st.markdown("**모든 검출된 장기를 한번에 보여주는 종합 3D 뷰**")
        
        # 장기 분류별로 그룹화 - TotalSegmentator 104개 구조 기준
        organ_groups = {
            "소화기계": ["small_bowel", "colon", "stomach", "duodenum", "liver", "pancreas", "gallbladder", 
                      "spleen", "esophagus", "trachea"],
            
            "비뇨생식기계": ["kidney_left", "kidney_right", "adrenal_gland_left", "adrenal_gland_right",
                        "kidney_cyst_left", "kidney_cyst_right", "urinary_bladder", "prostate"],
            
            "호흡기계": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", 
                      "lung_middle_lobe_right", "lung_lower_lobe_right"],
            
            "순환기계": ["heart", "aorta", "postcava", "portal_vein_splenic_vein", 
                      "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right"],
            
            "척추": [f"vertebrae_{level}" for level in ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                                      "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
                                                      "L1", "L2", "L3", "L4", "L5", "S1", "Coccyx"]],
            
            "늑골": [f"rib_{i}_{side}" for i in range(1, 13) for side in ["left", "right"]],
            
            "사지골격": ["humerus_left", "humerus_right", "scapula_left", "scapula_right", 
                      "clavicula_left", "clavicula_right", "femur_left", "femur_right", 
                      "hip_left", "hip_right", "sacrum"],
            
            "근육계": ["gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
                    "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right",
                    "iliopsoas_left", "iliopsoas_right"],
            
            "두경부": ["brain", "face"]
        }
        
        # 실제로 검출된 장기들만 필터링
        available_organs = list(results['visualization_data'].keys())
        known_organs = [name for name in available_organs if not name.startswith('unknown_label_')]
        unknown_organs = [name for name in available_organs if name.startswith('unknown_label_')]
        
        detected_groups = {}
        
        for group_name, organs in organ_groups.items():
            detected_organs = [organ for organ in organs if organ in available_organs]
            if detected_organs:
                detected_groups[group_name] = detected_organs
        
        # 미분류된 알려진 장기들
        categorized_organs = []
        for organs in detected_groups.values():
            categorized_organs.extend(organs)
        uncategorized_known = [organ for organ in known_organs if organ not in categorized_organs]
        if uncategorized_known:
            detected_groups["기타"] = detected_groups.get("기타", []) + uncategorized_known
        
        # Unknown 장기들은 별도 그룹으로
        if unknown_organs:
            detected_groups["❓ Unknown 장기"] = unknown_organs
        
        st.write(f"**검출된 장기 그룹:** {len(detected_groups)}개 그룹, 총 {len(available_organs)}개 장기")
        st.write(f"🔬 **알려진 장기:** {len(known_organs)}개 | ❓ **Unknown 장기:** {len(unknown_organs)}개")
        
        # Unknown 장기 상세 정보
        if unknown_organs:
            with st.expander(f"❓ Unknown 장기 상세 정보 ({len(unknown_organs)}개)", expanded=False):
                unknown_details = []
                for organ_name in unknown_organs:
                    mask = results['visualization_data'][organ_name]['mask']
                    voxel_count = np.sum(mask)
                    label_id = results['visualization_data'][organ_name]['label_id']
                    volume_ml = voxel_count * 0.5
                    unknown_details.append({
                        'name': organ_name,
                        'label_id': label_id,
                        'voxel_count': voxel_count,
                        'volume_ml': volume_ml
                    })
                
                # 복셀 수로 정렬
                unknown_details.sort(key=lambda x: x['voxel_count'], reverse=True)
                
                for detail in unknown_details:
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.write(f"**{detail['name']}**")
                    with col2:
                        st.write(f"Label: {detail['label_id']}")
                    with col3:
                        st.write(f"{detail['voxel_count']:,} 복셀")
                    with col4:
                        st.write(f"{detail['volume_ml']:.1f} ml")
                
                st.info("💡 Unknown 장기들은 TotalSegmentator의 104개 표준 라벨에 포함되지 않은 구조들입니다. 새로운 해부학적 구조이거나 분할 오류일 수 있습니다.")
        
        # 시각화 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            # 그룹별 선택
            selected_groups = st.multiselect(
                "표시할 장기 그룹 선택:",
                list(detected_groups.keys()),
                default=list(detected_groups.keys())[:3],  # 처음 3개 그룹만 기본 선택
                key="all_organs_groups"
            )
        
        with col2:
            # 전체 표시 옵션
            show_all_at_once = st.checkbox(
                "모든 장기 한번에 표시", 
                value=False,
                help="⚠️ 많은 장기를 동시에 표시하면 느려질 수 있습니다"
            )
            
            opacity_level = st.slider(
                "장기 투명도", 
                0.1, 1.0, 0.7,
                key="all_organs_opacity"
            )
        
        # 선택된 그룹의 장기들 수집
        organs_to_display = []
        if show_all_at_once:
            organs_to_display = available_organs
        else:
            for group in selected_groups:
                if group in detected_groups:
                    organs_to_display.extend(detected_groups[group])
        
        # 장기 수 제한 (성능을 위해)
        max_organs = 20
        if len(organs_to_display) > max_organs:
            st.warning(f"⚠️ 성능을 위해 처음 {max_organs}개 장기만 표시합니다.")
            organs_to_display = organs_to_display[:max_organs]
        
        if organs_to_display:
            # 선택된 장기 정보 표시
            st.info(f"🎯 표시할 장기: {len(organs_to_display)}개 - {', '.join(organs_to_display[:10])}{'...' if len(organs_to_display) > 10 else ''}")
            
            # 필터링된 데이터 준비
            filtered_all_organs = {
                k: v for k, v in results['visualization_data'].items() 
                if k in organs_to_display
            }
            
            # 3D 렌더러 생성
            with st.spinner(f"{len(organs_to_display)}개 장기의 3D 표면을 생성하는 중..."):
                volume_renderer_all = VolumeRenderer3D()
                volume_renderer_all.load_data(
                    results['original_ct'],
                    filtered_all_organs
                )
                
                # 3D 플롯 생성
                fig_all_organs = volume_renderer_all.create_interactive_plot(
                    selected_organs=list(filtered_all_organs.keys()),
                    show_slices=False  # 많은 장기 표시 시 슬라이스는 끄기
                )
                
                # 투명도 적용
                for trace in fig_all_organs.data:
                    if hasattr(trace, 'opacity'):
                        trace.opacity = opacity_level
                
                # 더 큰 화면으로 표시
                fig_all_organs.update_layout(
                    width=1000,
                    height=800,
                    title=f"TotalSegmentator - {len(organs_to_display)} Organs 3D Visualization"
                )
                
                st.plotly_chart(fig_all_organs, use_container_width=True)
            
            # 그룹별 통계 표시
            st.subheader("📊 그룹별 장기 통계")
            
            group_cols = st.columns(min(len(selected_groups), 4))  # 최대 4개 컬럼
            
            for idx, group_name in enumerate(selected_groups):
                if group_name in detected_groups:
                    col_idx = idx % 4
                    with group_cols[col_idx]:
                        st.markdown(f"**{group_name}**")
                        
                        group_organs = [organ for organ in detected_groups[group_name] 
                                      if organ in results['visualization_data']]
                        
                        total_voxels = 0
                        for organ in group_organs:
                            if organ in results['visualization_data']:
                                voxels = np.sum(results['visualization_data'][organ]['mask'])
                                total_voxels += voxels
                                st.write(f"• {organ.replace('_', ' ')}: {voxels:,}")
                        
                        st.write(f"**총합: {total_voxels:,} voxels**")
                        st.write(f"**추정 부피: {total_voxels * 0.5:.1f} ml**")
        
        else:
            st.warning("⚠️ 표시할 장기 그룹을 선택하세요.")
            
            # 사용 가능한 그룹 정보 표시
            for group_name, organs in detected_groups.items():
                with st.expander(f"🔍 {group_name} ({len(organs)}개 장기)"):
                    st.write(", ".join([organ.replace('_', ' ') for organ in organs]))
    
    with tab3:
        st.subheader("🔍 CT Slice Viewer with Organ Overlay")
        
        # 슬라이스 뷰어용 장기 선택
        slice_organs = st.multiselect(
            "슬라이스에 오버레이할 장기:",
            list(results['visualization_data'].keys()),
            default=list(results['visualization_data'].keys())[:2],
            key="slice_organs"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 축 선택
            axis = st.selectbox(
                "View Axis",
                ['axial', 'sagittal', 'coronal'],
                help="볼 축을 선택하세요"
            )
        
        with col2:
            # HU 윈도우 설정
            window_preset = st.selectbox(
                "HU Window Preset",
                ['Abdomen (-150~250)', 'Lung (-1000~0)', 'Bone (-500~1500)', 'Custom'],
                help="CT 윈도우 프리셋 선택"
            )
        
        # 슬라이스 인덱스 선택
        if axis == 'axial':
            max_idx = results['original_ct'].shape[0]
        elif axis == 'sagittal':
            max_idx = results['original_ct'].shape[1] 
        else:  # coronal
            max_idx = results['original_ct'].shape[2]
            
        slice_idx = st.slider(
            f"Slice Index ({axis})",
            0, max_idx - 1,
            max_idx // 2,
            key=f"slice_idx_{axis}"
        )
        
        # 슬라이스 뷰어 생성
        filtered_slice_data = {
            name: results['visualization_data'][name] 
            for name in slice_organs 
            if name in results['visualization_data']
        }
        
        slice_viewer = CTSliceViewer(
            results['original_ct'],
            filtered_slice_data
        )
        
        # 슬라이스 플롯 생성
        fig_slice = slice_viewer.create_slice_plot(axis, slice_idx)
        
        # HU 윈도우 적용
        if window_preset == 'Abdomen (-150~250)':
            fig_slice.update_traces(zmin=-150, zmax=250, selector=dict(type='heatmap'))
        elif window_preset == 'Lung (-1000~0)':
            fig_slice.update_traces(zmin=-1000, zmax=0, selector=dict(type='heatmap'))
        elif window_preset == 'Bone (-500~1500)':
            fig_slice.update_traces(zmin=-500, zmax=1500, selector=dict(type='heatmap'))
        
        st.plotly_chart(fig_slice, use_container_width=True)
        
        # 현재 슬라이스의 장기 정보
        st.subheader("📊 Current Slice Information")
        slice_info_cols = st.columns(len(slice_organs) if slice_organs else 1)
        
        for idx, organ_name in enumerate(slice_organs):
            if organ_name in results['visualization_data']:
                mask = results['visualization_data'][organ_name]['mask']
                
                # 해당 슬라이스에서 장기 픽셀 추출
                if axis == 'axial':
                    organ_slice = mask[slice_idx, :, :]
                elif axis == 'sagittal':
                    organ_slice = mask[:, slice_idx, :]
                else:  # coronal
                    organ_slice = mask[:, :, slice_idx]
                
                organ_pixels = np.sum(organ_slice)
                
                if idx < len(slice_info_cols):
                    with slice_info_cols[idx]:
                        st.metric(
                            f"{organ_name.replace('_', ' ').title()}", 
                            f"{organ_pixels} pixels"
                        )
                        
                        if organ_pixels > 0:
                            # 해당 슬라이스에서의 HU 값
                            organ_hu = results['original_ct'][organ_slice > 0] if axis == 'axial' else None
                            if organ_hu is not None and len(organ_hu) > 0:
                                st.write(f"Mean HU: {np.mean(organ_hu):.1f}")
        
        if not slice_organs:
            st.info("장기를 선택하면 슬라이스 오버레이를 볼 수 있습니다.")
    
    with tab4:
        st.subheader("📏 Straightened Small Bowel View")
        
        # 소장이 검출된 경우에만 펼친 뷰 표시
        if 'small_bowel' in results['visualization_data']:
            volume_renderer = VolumeRenderer3D()
            volume_renderer.load_data(
                results['original_ct'],
                results['visualization_data']
            )
            
            with st.spinner("소장을 일자로 펼치는 중..."):
                straightened = volume_renderer.create_straightened_view('small_bowel')
                
                if straightened is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(
                        z=straightened,
                        colorscale='gray',
                        showscale=True,
                        colorbar=dict(title="Intensity")
                    ))
                    
                    fig.update_layout(
                        title="Straightened Small Bowel",
                        xaxis_title="Width",
                        yaxis_title="Length along centerline",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 측정 정보
                    st.info(f"펼친 소장 길이: {straightened.shape[0]} 픽셀 (약 {straightened.shape[0] * 0.5:.1f} mm)")
                else:
                    st.warning("소장을 펼친 뷰를 생성할 수 없습니다. 소장 구조가 명확하지 않을 수 있습니다.")
        else:
            st.warning("소장이 검출되지 않았습니다. TotalSegmentator 자동 분할을 활성화해보세요.")
    
    with tab5:
        st.subheader("🔬 Detailed Analysis")
        
        # 파일 정보
        st.write("**File Information:**")
        st.write(f"- Original CT: {st.session_state['selected_file'].name}")
        st.write(f"- Segmentation: {Path(results['segmentation_path']).name}")
        st.write(f"- CT Shape: {results['original_ct'].shape}")
        st.write(f"- HU Range: {np.min(results['original_ct']):.0f} to {np.max(results['original_ct']):.0f}")
        
        # 장기별 상세 통계
        st.write("**Organ Statistics:**")
        
        for organ_name, data in results['visualization_data'].items():
            mask = data['mask']
            voxel_count = np.sum(mask)
            
            if voxel_count > 0:
                # 장기 영역의 CT 값 분석
                organ_hu_values = results['original_ct'][mask]
                
                st.write(f"**{organ_name.replace('_', ' ').title()}:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Voxels", f"{voxel_count:,}")
                with col2:
                    st.metric("Mean HU", f"{np.mean(organ_hu_values):.1f}")
                with col3:
                    st.metric("Std HU", f"{np.std(organ_hu_values):.1f}")
                with col4:
                    st.metric("Volume (ml)", f"{voxel_count * 0.5:.1f}")
        
        # 원시 데이터 다운로드
        st.write("**Data Export:**")
        
        if st.button("Generate Analysis Report"):
            report = generate_analysis_report(results)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"enterovision_report_{st.session_state['selected_file'].stem}.txt",
                mime="text/plain"
            )

def generate_analysis_report(results):
    """분석 보고서 생성"""
    report = []
    report.append("=" * 50)
    report.append("EnteroVision v2 Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # 기본 정보
    report.append("## File Information")
    report.append(f"- Segmentation file: {Path(results['segmentation_path']).name}")
    report.append(f"- CT volume shape: {results['original_ct'].shape}")
    report.append(f"- HU value range: {np.min(results['original_ct']):.0f} to {np.max(results['original_ct']):.0f}")
    report.append("")
    
    # 장기별 정보
    report.append("## Detected Organs")
    for organ_name, data in results['visualization_data'].items():
        mask = data['mask']
        voxel_count = np.sum(mask)
        
        if voxel_count > 0:
            organ_hu_values = results['original_ct'][mask]
            report.append(f"### {organ_name.replace('_', ' ').title()}")
            report.append(f"- Voxel count: {voxel_count:,}")
            report.append(f"- Estimated volume: {voxel_count * 0.5:.1f} ml")
            report.append(f"- Mean HU: {np.mean(organ_hu_values):.1f}")
            report.append(f"- HU standard deviation: {np.std(organ_hu_values):.1f}")
            report.append(f"- Label ID: {data['label_id']}")
            report.append("")
    
    # 요약
    report.append("## Summary")
    report.append(f"- Total detected organs: {len(results['visualization_data'])}")
    detected_organs = [name for name, data in results['visualization_data'].items() if np.sum(data['mask']) > 0]
    report.append(f"- Organs with segmentation: {', '.join(detected_organs)}")
    
    return '\n'.join(report)

if __name__ == "__main__":
    main()