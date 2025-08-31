"""
3D Volume Rendering module for EnteroVision v2
CT 데이터와 TotalSegmentator 분할 결과를 이용한 3D 시각화
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from skimage import measure
from scipy import ndimage
import SimpleITK as sitk


class VolumeRenderer3D:
    """3D 볼륨 렌더링 클래스"""
    
    def __init__(self):
        self.ct_volume = None
        self.segmentation_data = None
        self.organ_surfaces = {}
        
    def load_data(self, ct_volume, segmentation_data):
        """CT 볼륨과 분할 데이터 로드"""
        self.ct_volume = ct_volume
        self.segmentation_data = segmentation_data
        
        print(f"CT 볼륨 크기: {ct_volume.shape}")
        print(f"분할된 장기 수: {len(segmentation_data)}")
        
    def create_organ_surface(self, mask, organ_name, step_size=2):
        """장기 마스크에서 3D 표면 생성"""
        try:
            # 다운샘플링으로 성능 향상
            if step_size > 1:
                mask_downsampled = mask[::step_size, ::step_size, ::step_size]
            else:
                mask_downsampled = mask
            
            # 스무딩 필터 적용
            mask_smoothed = ndimage.gaussian_filter(mask_downsampled.astype(float), sigma=1.0)
            mask_smoothed = mask_smoothed > 0.5
            
            if np.sum(mask_smoothed) == 0:
                print(f"{organ_name}: 마스크가 비어있음")
                return None
            
            # Marching cubes로 3D 표면 추출
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    mask_smoothed, 
                    level=0.5,
                    step_size=1,
                    allow_degenerate=False
                )
                
                # 좌표 스케일링 (다운샘플링 보정)
                if step_size > 1:
                    verts = verts * step_size
                
                print(f"{organ_name}: {len(verts)}개 정점, {len(faces)}개 면")
                
                return {
                    'vertices': verts,
                    'faces': faces,
                    'normals': normals
                }
                
            except ValueError as e:
                print(f"{organ_name} 표면 추출 실패: {e}")
                return None
                
        except Exception as e:
            print(f"{organ_name} 표면 생성 중 오류: {e}")
            return None
    
    def create_interactive_plot(self, selected_organs=None, show_slices=True):
        """대화형 3D 플롯 생성"""
        if self.ct_volume is None or self.segmentation_data is None:
            return go.Figure().add_annotation(
                text="데이터를 먼저 로드하세요",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # 장기별 3D 표면 추가 (먼저 추가하여 메인으로 표시)
        if selected_organs is None:
            selected_organs = list(self.segmentation_data.keys())
        
        organs_added = 0
        for organ_name in selected_organs:
            if organ_name in self.segmentation_data:
                try:
                    self._add_organ_surface(fig, organ_name)
                    organs_added += 1
                    print(f"✅ {organ_name} 3D 표면 추가 완료")
                except Exception as e:
                    print(f"❌ {organ_name} 3D 표면 추가 실패: {e}")
        
        # CT 볼륨 슬라이스 표시 (배경으로)
        if show_slices:
            try:
                self._add_ct_slices(fig, num_slices=2)  # 슬라이스 수 줄여서 성능 향상
                print("✅ CT 슬라이스 추가 완료")
            except Exception as e:
                print(f"❌ CT 슬라이스 추가 실패: {e}")
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"EnteroVision 3D CT Visualization ({organs_added} organs)",
            scene=dict(
                xaxis_title="X (voxels)",
                yaxis_title="Y (voxels)", 
                zaxis_title="Z (voxels)",
                aspectmode='cube',  # 정육면체 비율로 표시
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(0,0,0,0.1)'  # 약간 어두운 배경
            ),
            width=900,
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        # 데이터가 없으면 안내 메시지 추가
        if organs_added == 0:
            fig.add_annotation(
                text="선택한 장기의 3D 표면을 생성할 수 없습니다.<br>다른 장기를 선택하거나 분할 품질을 확인하세요.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        return fig
    
    def _add_ct_slices(self, fig, num_slices=3):
        """CT 슬라이스를 3D 플롯에 추가"""
        try:
            z_shape, y_shape, x_shape = self.ct_volume.shape
            print(f"🖼️ CT 슬라이스 추가 중... Volume shape: {self.ct_volume.shape}")
            
            # 다운샘플링으로 성능 향상
            step = max(1, min(x_shape, y_shape) // 100)  # 최대 100x100 해상도
            
            # Z축 슬라이스들 (균등하게 분포)
            slice_indices = np.linspace(z_shape//6, 5*z_shape//6, num_slices, dtype=int)
            
            for i, z_idx in enumerate(slice_indices):
                slice_data = self.ct_volume[z_idx, ::step, ::step]
                
                # HU값 윈도우 적용 (복부 CT 기준)
                windowed = np.clip(slice_data, -200, 200)  # 복부 윈도우
                slice_normalized = (windowed + 200) / 400  # 0-1로 정규화
                
                # 다운샘플링된 좌표 그리드
                y_coords, x_coords = np.meshgrid(
                    np.arange(0, y_shape, step), 
                    np.arange(0, x_shape, step), 
                    indexing='ij'
                )
                z_coords = np.full_like(x_coords, z_idx)
                
                # 투명도를 낮춰서 장기가 잘 보이도록
                opacity = 0.2 - (i * 0.05)  # 뒤쪽 슬라이스일수록 더 투명
                opacity = max(0.1, opacity)
                
                fig.add_trace(go.Surface(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    surfacecolor=slice_normalized,
                    colorscale='gray',
                    showscale=False,
                    opacity=opacity,
                    name=f'CT Slice {z_idx}',
                    hovertemplate=f'CT Slice Z={z_idx}<br>HU: %{{surfacecolor:.0f}}<extra></extra>',
                    contours=dict(
                        x=dict(show=False),
                        y=dict(show=False),
                        z=dict(show=False)
                    )
                ))
            
            print(f"✅ {num_slices}개 CT 슬라이스 추가 완료")
            
        except Exception as e:
            print(f"❌ CT 슬라이스 추가 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_organ_surface(self, fig, organ_name):
        """장기 3D 표면을 플롯에 추가"""
        try:
            mask = self.segmentation_data[organ_name]['mask']
            voxel_count = np.sum(mask)
            
            print(f"🔍 {organ_name} 처리 중... ({voxel_count:,} voxels)")
            
            # 충분한 복셀이 있는지 확인
            if voxel_count < 100:
                print(f"⚠️ {organ_name}: 복셀 수가 너무 적음 ({voxel_count})")
                return
            
            if organ_name not in self.organ_surfaces:
                # 표면이 아직 생성되지 않았으면 생성
                step_size = 4 if voxel_count > 50000 else 2  # 큰 장기는 더 많이 다운샘플링
                surface_data = self.create_organ_surface(mask, organ_name, step_size=step_size)
                
                if surface_data is not None:
                    self.organ_surfaces[organ_name] = surface_data
                else:
                    print(f"❌ {organ_name} 표면 생성 실패")
                    return
            
            surface = self.organ_surfaces[organ_name]
            color = self.segmentation_data[organ_name]['color']
            
            # 면의 수 확인
            num_faces = len(surface['faces'])
            print(f"📐 {organ_name}: {len(surface['vertices'])} 정점, {num_faces} 면")
            
            if num_faces == 0:
                print(f"⚠️ {organ_name}: 면이 생성되지 않음")
                return
            
            # 너무 많은 면이 있으면 간소화
            if num_faces > 10000:
                print(f"⚠️ {organ_name}: 면이 너무 많음 ({num_faces}), 간소화 적용")
                # 간단한 간소화: 일정 간격으로 면 선택
                step = max(1, num_faces // 10000)
                surface['faces'] = surface['faces'][::step]
                print(f"✂️ {organ_name}: {len(surface['faces'])} 면으로 간소화")
            
            # 3D 메쉬 추가
            fig.add_trace(go.Mesh3d(
                x=surface['vertices'][:, 2],  # ITK 좌표계 조정 (Z->X)
                y=surface['vertices'][:, 1],  # Y는 그대로
                z=surface['vertices'][:, 0],  # X->Z
                i=surface['faces'][:, 0],
                j=surface['faces'][:, 1], 
                k=surface['faces'][:, 2],
                color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                opacity=0.8,
                name=organ_name.replace('_', ' ').title(),
                hovertemplate=f'<b>{organ_name}</b><br>Voxels: {voxel_count:,}<br>x: %{{x}}<br>y: %{{y}}<br>z: %{{z}}<extra></extra>',
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    specular=0.2
                )
            ))
            
            print(f"✅ {organ_name} 3D 메쉬 추가 완료")
            
        except Exception as e:
            print(f"❌ {organ_name} 표면 추가 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def create_straightened_view(self, organ_name='small_bowel'):
        """장기를 일자로 펼친 뷰 생성"""
        if organ_name not in self.segmentation_data:
            return None
        
        mask = self.segmentation_data[organ_name]['mask']
        
        # 중심선 추출 (단순화된 버전)
        centerline = self._extract_simple_centerline(mask)
        
        if centerline is None or len(centerline) < 10:
            print(f"{organ_name} 중심선 추출 실패")
            return None
        
        # 중심선을 따라 펼친 이미지 생성
        straightened_image = self._create_straightened_image(mask, centerline)
        
        return straightened_image
    
    def _extract_simple_centerline(self, mask):
        """간단한 중심선 추출 (스켈레톤화)"""
        try:
            from skimage.morphology import skeletonize_3d, remove_small_objects
            from skimage.measure import label
            
            # 작은 노이즈 제거
            cleaned_mask = remove_small_objects(mask, min_size=100)
            
            # 3D 스켈레톤화
            skeleton = skeletonize_3d(cleaned_mask)
            
            # 스켈레톤 점들을 연결된 순서로 정렬
            skeleton_points = np.where(skeleton)
            skeleton_coords = list(zip(skeleton_points[0], skeleton_points[1], skeleton_points[2]))
            
            if len(skeleton_coords) < 10:
                return None
            
            # 중심선 순서 정리 (단순화된 방법)
            centerline = self._order_centerline_points(skeleton_coords)
            
            return np.array(centerline)
            
        except ImportError:
            print("skimage.morphology 모듈이 필요합니다")
            return None
        except Exception as e:
            print(f"중심선 추출 실패: {e}")
            return None
    
    def _order_centerline_points(self, points):
        """중심선 점들을 순서대로 정렬"""
        if len(points) < 2:
            return points
        
        # 간단한 nearest neighbor 방식으로 연결
        ordered = [points[0]]
        remaining = set(points[1:])
        
        while remaining:
            current = ordered[-1]
            # 현재 점에서 가장 가까운 점 찾기
            distances = [np.linalg.norm(np.array(p) - np.array(current)) for p in remaining]
            nearest_idx = np.argmin(distances)
            nearest_point = list(remaining)[nearest_idx]
            
            ordered.append(nearest_point)
            remaining.remove(nearest_point)
        
        return ordered
    
    def _create_straightened_image(self, mask, centerline, width=50):
        """중심선을 따라 펼친 2D 이미지 생성"""
        try:
            # 중심선 길이만큼의 2D 이미지 생성
            height = len(centerline)
            straightened = np.zeros((height, width))
            
            for i, center_point in enumerate(centerline):
                z, y, x = center_point
                
                # 현재 위치에서 수직 단면 추출
                cross_section = self._extract_cross_section(mask, center_point, width)
                
                if cross_section is not None:
                    straightened[i, :] = cross_section
            
            return straightened
            
        except Exception as e:
            print(f"펼친 이미지 생성 실패: {e}")
            return None
    
    def _extract_cross_section(self, volume, center_point, width):
        """특정 점에서 수직 단면 추출"""
        try:
            z, y, x = center_point
            z_shape, y_shape, x_shape = volume.shape
            
            # 간단한 수평 단면 추출
            half_width = width // 2
            
            x_start = max(0, x - half_width)
            x_end = min(x_shape, x + half_width)
            
            if x_start >= x_end or z >= z_shape or y >= y_shape:
                return np.zeros(width)
            
            # 단면 데이터 추출
            section = volume[z, y, x_start:x_end]
            
            # 원하는 길이로 패딩/트림
            if len(section) < width:
                padded = np.zeros(width)
                padded[:len(section)] = section
                return padded
            else:
                return section[:width]
                
        except Exception as e:
            print(f"단면 추출 실패: {e}")
            return np.zeros(width)


class CTSliceViewer:
    """CT 슬라이스 뷰어"""
    
    def __init__(self, ct_volume, segmentation_data=None):
        self.ct_volume = ct_volume
        self.segmentation_data = segmentation_data
        
    def create_slice_plot(self, axis='axial', slice_idx=None):
        """축별 슬라이스 플롯 생성"""
        if slice_idx is None:
            slice_idx = self.ct_volume.shape[0] // 2 if axis == 'axial' else self.ct_volume.shape[1] // 2
        
        if axis == 'axial':
            slice_data = self.ct_volume[slice_idx, :, :]
            title = f"Axial Slice (Z={slice_idx})"
        elif axis == 'sagittal':
            slice_data = self.ct_volume[:, slice_idx, :]
            title = f"Sagittal Slice (Y={slice_idx})"
        elif axis == 'coronal':
            slice_data = self.ct_volume[:, :, slice_idx]
            title = f"Coronal Slice (X={slice_idx})"
        else:
            raise ValueError("axis는 'axial', 'sagittal', 'coronal' 중 하나여야 합니다")
        
        # HU 값 윈도우 적용
        windowed = np.clip(slice_data, -1000, 1000)
        
        fig = go.Figure()
        
        # CT 이미지 추가
        fig.add_trace(go.Heatmap(
            z=windowed,
            colorscale='gray',
            showscale=True,
            colorbar=dict(title="HU"),
            name="CT"
        ))
        
        # 분할 결과 오버레이
        if self.segmentation_data:
            self._add_segmentation_overlay(fig, axis, slice_idx)
        
        fig.update_layout(
            title=title,
            xaxis=dict(title="X"),
            yaxis=dict(title="Y", scaleanchor="x", scaleratio=1),
            width=600,
            height=600
        )
        
        return fig
    
    def _add_segmentation_overlay(self, fig, axis, slice_idx):
        """분할 결과를 슬라이스에 오버레이"""
        for organ_name, data in self.segmentation_data.items():
            mask = data['mask']
            color = data['color']
            
            if axis == 'axial':
                mask_slice = mask[slice_idx, :, :]
            elif axis == 'sagittal':
                mask_slice = mask[:, slice_idx, :]
            elif axis == 'coronal':
                mask_slice = mask[:, :, slice_idx]
            
            if np.sum(mask_slice) > 0:
                # 마스크 경계선 찾기
                from skimage import measure
                contours = measure.find_contours(mask_slice, 0.5)
                
                for contour in contours:
                    fig.add_trace(go.Scatter(
                        x=contour[:, 1],
                        y=contour[:, 0],
                        mode='lines',
                        line=dict(
                            color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})',
                            width=2
                        ),
                        name=organ_name,
                        showlegend=True
                    ))


def create_multi_view_plot(ct_volume, segmentation_data):
    """다중 뷰 플롯 생성"""
    slice_viewer = CTSliceViewer(ct_volume, segmentation_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Volume', 'Axial', 'Sagittal', 'Coronal'),
        specs=[[{'type': 'scene'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # 3D 렌더러
    volume_renderer = VolumeRenderer3D()
    volume_renderer.load_data(ct_volume, segmentation_data)
    
    # 각 뷰 추가
    fig_3d = volume_renderer.create_interactive_plot(show_slices=False)
    for trace in fig_3d.data:
        fig.add_trace(trace, row=1, col=1)
    
    # 슬라이스 뷰들 추가
    axes = ['axial', 'sagittal', 'coronal']
    positions = [(1, 2), (2, 1), (2, 2)]
    
    for axis, (row, col) in zip(axes, positions):
        slice_fig = slice_viewer.create_slice_plot(axis)
        for trace in slice_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="EnteroVision Multi-View CT Analysis"
    )
    
    return fig