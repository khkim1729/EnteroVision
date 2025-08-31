"""
Colon Curved Planar Reformation (CPR) Visualizer
CT colonography를 위한 대장 시각화 및 CPR 구현
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from skimage import measure, morphology, filters
from scipy import ndimage
from scipy.interpolate import splprep, splev
import SimpleITK as sitk


class ColonCPRVisualizer:
    """대장을 위한 Curved Planar Reformation 시각화"""
    
    def __init__(self):
        self.colon_mask = None
        self.ct_volume = None
        self.centerline = None
        self.cpr_image = None
        
    def load_data(self, ct_volume, colon_mask):
        """CT 볼륨과 대장 마스크 로드"""
        self.ct_volume = ct_volume
        self.colon_mask = colon_mask
        
        print(f"CT 볼륨 크기: {ct_volume.shape}")
        print(f"대장 복셀 수: {np.sum(colon_mask):,}")
        
        if np.sum(colon_mask) == 0:
            raise ValueError("대장 마스크가 비어있습니다")
    
    def extract_colon_centerline(self):
        """대장 중심선 추출"""
        try:
            # 1. 대장 마스크 전처리
            cleaned_mask = self._preprocess_colon_mask(self.colon_mask)
            
            # 2. 스켈레톤화를 통한 중심선 추출
            skeleton = morphology.skeletonize_3d(cleaned_mask)
            
            # 3. 스켈레톤 점들을 연결된 경로로 정렬
            centerline = self._order_skeleton_points(skeleton)
            
            # 4. 스무딩 적용
            smoothed_centerline = self._smooth_centerline(centerline)
            
            self.centerline = smoothed_centerline
            
            print(f"대장 중심선 추출 완료: {len(smoothed_centerline)}개 점")
            return smoothed_centerline
            
        except Exception as e:
            print(f"대장 중심선 추출 실패: {e}")
            return None
    
    def _preprocess_colon_mask(self, mask):
        """대장 마스크 전처리"""
        # 작은 노이즈 제거
        cleaned = morphology.remove_small_objects(mask, min_size=500)
        
        # 홀 채우기
        filled = ndimage.binary_fill_holes(cleaned)
        
        # 약간의 침식으로 경계 정리
        kernel = morphology.ball(1)
        eroded = morphology.binary_erosion(filled, kernel)
        
        return eroded
    
    def _order_skeleton_points(self, skeleton):
        """스켈레톤 점들을 연결된 순서로 정렬"""
        points = np.where(skeleton)
        coords = list(zip(points[0], points[1], points[2]))
        
        if len(coords) < 10:
            return np.array(coords)
        
        # 그래프 기반 경로 찾기
        ordered_path = self._find_longest_path(coords)
        
        return np.array(ordered_path)
    
    def _find_longest_path(self, points):
        """점들 사이의 최장 경로 찾기"""
        import networkx as nx
        
        # 그래프 생성
        G = nx.Graph()
        
        # 점들을 노드로 추가
        for i, point in enumerate(points):
            G.add_node(i, pos=point)
        
        # 인접한 점들 연결 (거리 기반)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if dist <= 2.0:  # 인접 거리 임계값
                    G.add_edge(i, j, weight=dist)
        
        # 최장 경로 찾기 (단순화된 방법)
        if len(G.nodes) < 2:
            return points
        
        # 끝점 찾기 (degree가 1인 노드들)
        endpoints = [n for n in G.nodes if G.degree[n] == 1]
        
        if len(endpoints) >= 2:
            # 끝점들 사이의 최단 경로 (실제로는 중심선)
            try:
                path = nx.shortest_path(G, endpoints[0], endpoints[-1])
                return [points[i] for i in path]
            except nx.NetworkXNoPath:
                pass
        
        # 대체 방법: DFS로 최장 경로
        try:
            start_node = list(G.nodes)[0]
            visited = set()
            longest_path = []
            
            def dfs(node, current_path):
                nonlocal longest_path
                visited.add(node)
                current_path.append(node)
                
                if len(current_path) > len(longest_path):
                    longest_path = current_path.copy()
                
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        dfs(neighbor, current_path)
                
                current_path.pop()
                visited.remove(node)
            
            dfs(start_node, [])
            
            if longest_path:
                return [points[i] for i in longest_path]
        except:
            pass
        
        # 최종 대체: 단순 정렬
        return self._simple_order_points(points)
    
    def _simple_order_points(self, points):
        """단순한 nearest neighbor 방식으로 점 정렬"""
        if len(points) < 2:
            return points
        
        ordered = [points[0]]
        remaining = set(points[1:])
        
        while remaining and len(ordered) < len(points):
            current = ordered[-1]
            distances = [np.linalg.norm(np.array(p) - np.array(current)) for p in remaining]
            
            if distances:
                nearest_idx = np.argmin(distances)
                nearest_point = list(remaining)[nearest_idx]
                ordered.append(nearest_point)
                remaining.remove(nearest_point)
            else:
                break
        
        return ordered
    
    def _smooth_centerline(self, centerline):
        """중심선 스무딩"""
        if len(centerline) < 4:
            return centerline
        
        try:
            # 3D 스플라인 피팅
            centerline_array = np.array(centerline)
            
            # 중복 점 제거
            unique_indices = []
            for i, point in enumerate(centerline_array):
                if i == 0 or np.linalg.norm(point - centerline_array[i-1]) > 0.1:
                    unique_indices.append(i)
            
            if len(unique_indices) < 4:
                return centerline
            
            unique_centerline = centerline_array[unique_indices]
            
            # 스플라인 피팅
            tck, u = splprep([unique_centerline[:, 0], unique_centerline[:, 1], unique_centerline[:, 2]], 
                           s=len(unique_centerline) * 0.1, k=3)
            
            # 더 많은 점으로 재샘플링
            u_new = np.linspace(0, 1, len(unique_centerline) * 2)
            smoothed = splev(u_new, tck)
            
            return np.column_stack(smoothed)
            
        except Exception as e:
            print(f"중심선 스무딩 실패: {e}, 원본 사용")
            return centerline
    
    def create_cpr_image(self, width=100, interpolation_method='linear'):
        """Curved Planar Reformation 이미지 생성"""
        if self.centerline is None:
            print("먼저 중심선을 추출하세요")
            return None
        
        if len(self.centerline) < 5:
            print("중심선이 너무 짧습니다")
            return None
        
        height = len(self.centerline)
        cpr_image = np.zeros((height, width))
        
        try:
            for i, center_point in enumerate(self.centerline):
                # 각 중심선 점에서 수직 단면 추출
                cross_section = self._extract_perpendicular_section(center_point, i, width)
                
                if cross_section is not None and len(cross_section) == width:
                    cpr_image[i, :] = cross_section
                else:
                    # 이전 또는 다음 단면으로 보간
                    if i > 0:
                        cpr_image[i, :] = cpr_image[i-1, :]
                    else:
                        cpr_image[i, :] = np.zeros(width)
            
            self.cpr_image = cpr_image
            print(f"CPR 이미지 생성 완료: {cpr_image.shape}")
            
            return cpr_image
            
        except Exception as e:
            print(f"CPR 이미지 생성 실패: {e}")
            return None
    
    def _extract_perpendicular_section(self, center_point, index, width):
        """중심선 점에서 수직 단면 추출"""
        try:
            z, y, x = center_point
            z, y, x = int(z), int(y), int(x)
            
            # 볼륨 경계 확인
            if not (0 <= z < self.ct_volume.shape[0] and 
                   0 <= y < self.ct_volume.shape[1] and 
                   0 <= x < self.ct_volume.shape[2]):
                return np.zeros(width)
            
            # 중심선 방향 벡터 계산
            direction = self._get_centerline_direction(index)
            
            # 수직 벡터들 계산
            perpendicular_vectors = self._get_perpendicular_vectors(direction)
            
            # 수직 단면 샘플링
            section = self._sample_perpendicular_plane(
                center_point, perpendicular_vectors[0], width
            )
            
            return section
            
        except Exception as e:
            print(f"수직 단면 추출 실패 (index {index}): {e}")
            return np.zeros(width)
    
    def _get_centerline_direction(self, index):
        """중심선 방향 벡터 계산"""
        if index == 0:
            if len(self.centerline) > 1:
                return self.centerline[1] - self.centerline[0]
            else:
                return np.array([1, 0, 0])  # 기본 방향
        elif index == len(self.centerline) - 1:
            return self.centerline[index] - self.centerline[index-1]
        else:
            # 중앙 차분
            return self.centerline[index+1] - self.centerline[index-1]
    
    def _get_perpendicular_vectors(self, direction):
        """방향 벡터에 수직인 벡터들 계산"""
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # 첫 번째 수직 벡터
        if abs(direction[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        # 그람-슈미트 과정으로 정규직교화
        v1 = v1 - np.dot(v1, direction) * direction
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        
        # 두 번째 수직 벡터 (외적)
        v2 = np.cross(direction, v1)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        return v1, v2
    
    def _sample_perpendicular_plane(self, center, direction, width):
        """수직 평면에서 샘플링"""
        half_width = width // 2
        samples = []
        
        for i in range(-half_width, width - half_width):
            sample_point = center + i * direction
            
            # 트리선형 보간으로 CT 값 추출
            value = self._trilinear_interpolation(sample_point)
            samples.append(value)
        
        return np.array(samples)
    
    def _trilinear_interpolation(self, point):
        """3D 트리선형 보간"""
        try:
            z, y, x = point
            
            # 정수 부분과 소수 부분
            z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
            z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
            
            # 경계 확인
            if (z0 < 0 or z1 >= self.ct_volume.shape[0] or
                y0 < 0 or y1 >= self.ct_volume.shape[1] or
                x0 < 0 or x1 >= self.ct_volume.shape[2]):
                return -1000  # 경계 밖은 공기로 가정
            
            # 소수 부분
            dz, dy, dx = z - z0, y - y0, x - x0
            
            # 8개 꼭짓점 값
            c000 = self.ct_volume[z0, y0, x0]
            c001 = self.ct_volume[z0, y0, x1]
            c010 = self.ct_volume[z0, y1, x0]
            c011 = self.ct_volume[z0, y1, x1]
            c100 = self.ct_volume[z1, y0, x0]
            c101 = self.ct_volume[z1, y0, x1]
            c110 = self.ct_volume[z1, y1, x0]
            c111 = self.ct_volume[z1, y1, x1]
            
            # 트리선형 보간
            c00 = c000 * (1 - dx) + c001 * dx
            c01 = c010 * (1 - dx) + c011 * dx
            c10 = c100 * (1 - dx) + c101 * dx
            c11 = c110 * (1 - dx) + c111 * dx
            
            c0 = c00 * (1 - dy) + c01 * dy
            c1 = c10 * (1 - dy) + c11 * dy
            
            result = c0 * (1 - dz) + c1 * dz
            
            return float(result)
            
        except Exception:
            return -1000  # 에러 시 공기 값
    
    def create_interactive_cpr_plot(self):
        """대화형 CPR 플롯 생성"""
        if self.cpr_image is None:
            return go.Figure().add_annotation(
                text="먼저 CPR 이미지를 생성하세요",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # CPR 이미지 표시
        fig.add_trace(go.Heatmap(
            z=self.cpr_image,
            colorscale='gray',
            showscale=True,
            colorbar=dict(title="HU Value"),
            name="CPR Image",
            hovertemplate='Position: %{y}<br>Width: %{x}<br>HU: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Colon Curved Planar Reformation (CPR)",
            xaxis_title="Width (pixels)",
            yaxis_title="Position along colon centerline",
            width=800,
            height=600
        )
        
        return fig
    
    def create_colon_3d_plot(self):
        """대장 3D 시각화 (중심선 포함)"""
        if self.colon_mask is None:
            return go.Figure()
        
        fig = go.Figure()
        
        # 대장 표면 생성
        try:
            # 다운샘플링으로 성능 향상
            downsampled_mask = self.colon_mask[::2, ::2, ::2]
            
            verts, faces, _, _ = measure.marching_cubes(
                downsampled_mask, level=0.5, step_size=1
            )
            
            # 좌표 스케일링
            verts = verts * 2
            
            # 3D 메시 추가
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2],  # ITK 좌표계 조정
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='orange',
                opacity=0.6,
                name='Colon Surface'
            ))
            
        except Exception as e:
            print(f"대장 표면 생성 실패: {e}")
        
        # 중심선 추가
        if self.centerline is not None and len(self.centerline) > 0:
            centerline_array = np.array(self.centerline)
            
            fig.add_trace(go.Scatter3d(
                x=centerline_array[:, 2],  # ITK 좌표계 조정
                y=centerline_array[:, 1],
                z=centerline_array[:, 0],
                mode='lines+markers',
                line=dict(color='red', width=8),
                marker=dict(size=3, color='red'),
                name='Centerline'
            ))
        
        fig.update_layout(
            title="3D Colon Visualization with Centerline",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_multi_view_plot(self):
        """다중 뷰 플롯 (3D + CPR)"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('3D Colon View', 'CPR View'),
            specs=[[{'type': 'scene'}, {'type': 'xy'}]]
        )
        
        # 3D 뷰 추가
        fig_3d = self.create_colon_3d_plot()
        for trace in fig_3d.data:
            fig.add_trace(trace, row=1, col=1)
        
        # CPR 뷰 추가
        if self.cpr_image is not None:
            fig.add_trace(go.Heatmap(
                z=self.cpr_image,
                colorscale='gray',
                showscale=True,
                colorbar=dict(title="HU"),
                name="CPR"
            ), row=1, col=2)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Colon Analysis: 3D View and Curved Planar Reformation"
        )
        
        return fig