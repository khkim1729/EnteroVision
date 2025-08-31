"""
Enhanced TotalSegmentator wrapper for EnteroVision v2
CT 영상에서 자동으로 소장 및 주변 장기들을 분할하여 장협착증 진단에 활용
"""

import os
import tempfile
import hashlib
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import json

class TotalSegmentatorWrapper:
    """TotalSegmentator를 이용한 자동 장기 분할"""
    
    def __init__(self):
        self.segmentation_result = None
        self.organ_labels = {}
        
        # TotalSegmentator v2 라벨 매핑 (최신 버전 기준)
        self.totalseg_v2_labels = {
            'background': 0,
            'spleen': 1,
            'kidney_right': 2,
            'kidney_left': 3,
            'gallbladder': 4,
            'liver': 5,
            'stomach': 6,
            'pancreas': 7,
            'adrenal_gland_right': 8,
            'adrenal_gland_left': 9,
            'lung_upper_lobe_left': 10,
            'lung_lower_lobe_left': 11,
            'lung_upper_lobe_right': 12,
            'lung_middle_lobe_right': 13,
            'lung_lower_lobe_right': 14,
            'vertebrae_L5': 15,
            'vertebrae_L4': 16,
            'vertebrae_L3': 17,
            'vertebrae_L2': 18,
            'vertebrae_L1': 19,
            'vertebrae_T12': 20,
            'vertebrae_T11': 21,
            'vertebrae_T10': 22,
            'vertebrae_T9': 23,
            'vertebrae_T8': 24,
            'vertebrae_T7': 25,
            'vertebrae_T6': 26,
            'vertebrae_T5': 27,
            'vertebrae_T4': 28,
            'vertebrae_T3': 29,
            'vertebrae_T2': 30,
            'vertebrae_T1': 31,
            'vertebrae_C7': 32,
            'vertebrae_C6': 33,
            'vertebrae_C5': 34,
            'vertebrae_C4': 35,
            'vertebrae_C3': 36,
            'vertebrae_C2': 37,
            'vertebrae_C1': 38,
            'heart': 39,
            'aorta': 40,
            'postcava': 41,
            'portal_vein_splenic_vein': 42,
            'brain': 43,
            'iliac_artery_left': 44,
            'iliac_artery_right': 45,
            'iliac_vena_left': 46,
            'iliac_vena_right': 47,
            'small_bowel': 48,  # 소장 라벨 ID
            'duodenum': 49,
            'colon': 50,
        }
        
        # 소장 및 장협착증 진단에 중요한 장기들의 라벨 정보
        self.bowel_related_organs = {
            'small_bowel': self.totalseg_v2_labels.get('small_bowel', 48),
            'colon': self.totalseg_v2_labels.get('colon', 50),
            'duodenum': self.totalseg_v2_labels.get('duodenum', 49),
            'stomach': self.totalseg_v2_labels.get('stomach', 6),
            'liver': self.totalseg_v2_labels.get('liver', 5),
            'pancreas': self.totalseg_v2_labels.get('pancreas', 7),
            'spleen': self.totalseg_v2_labels.get('spleen', 1),
            'kidney_left': self.totalseg_v2_labels.get('kidney_left', 3),
            'kidney_right': self.totalseg_v2_labels.get('kidney_right', 2),
            'gallbladder': self.totalseg_v2_labels.get('gallbladder', 4),
        }
        
        # 혈관계 (조영제 관련)
        self.vascular_organs = {
            'aorta': self.totalseg_v2_labels.get('aorta', 40),
            'postcava': self.totalseg_v2_labels.get('postcava', 41),  # 하대정맥
            'portal_vein_splenic_vein': self.totalseg_v2_labels.get('portal_vein_splenic_vein', 42),
        }
        
        print("TotalSegmentator 래퍼 초기화 완료")
        print(f"소장 라벨 ID: {self.bowel_related_organs['small_bowel']}")
        
    def run_segmentation(self, input_path, task='total'):
        """
        TotalSegmentator를 실행하여 장기 분할 수행
        
        Args:
            input_path: CT 영상 파일 경로
            task: 분할 태스크 ('total' 또는 'body')
        
        Returns:
            segmentation_path: 분할 결과 파일 경로
        """
        print(f"TotalSegmentator 실행 시작: {input_path}")
        
        # 입력 파일명에서 결과 파일명 생성
        input_filename = os.path.basename(input_path)
        base_name = os.path.splitext(input_filename)[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
        
        # 결과 저장 디렉토리 설정 (shared datasets 사용)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'datasets', 'ct_labels')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'{base_name}_totalseg.nii.gz')
        
        # 이미 존재하는지 확인
        if os.path.exists(output_path):
            print(f"기존 분할 결과 발견: {output_path}")
            self.segmentation_result = output_path
            return output_path
        
        try:
            # TotalSegmentator 실행 명령
            cmd = [
                'TotalSegmentator',
                '-i', input_path,
                '-o', output_path,
                '--task', task,
                '--ml',  # 다중 라벨 출력
                '--fast',  # 빠른 모드 (필요시)
            ]
            
            print(f"실행 명령: {' '.join(cmd)}")
            
            # 서브프로세스로 TotalSegmentator 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            if result.returncode != 0:
                print(f"TotalSegmentator 실행 실패:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None
            
            print("TotalSegmentator 실행 완료")
            
            # 결과 확인 및 저장
            if os.path.exists(output_path):
                print(f"분할 결과 저장 완료: {output_path}")
                self.segmentation_result = output_path
                return output_path
            else:
                print(f"분할 결과 파일을 찾을 수 없음: {output_path}")
                return None
                
        except subprocess.TimeoutExpired:
            print("TotalSegmentator 실행 시간 초과")
            return None
        except Exception as e:
            print(f"TotalSegmentator 실행 중 오류: {e}")
            return None
    
    def load_segmentation(self, segmentation_path):
        """분할 결과를 로드"""
        try:
            segmentation_image = sitk.ReadImage(segmentation_path)
            segmentation_array = sitk.GetArrayFromImage(segmentation_image)
            
            print(f"분할 결과 로드 완료: {segmentation_array.shape}")
            print(f"라벨 범위: {np.min(segmentation_array)} ~ {np.max(segmentation_array)}")
            
            # 존재하는 라벨들 확인
            unique_labels = np.unique(segmentation_array)
            print(f"존재하는 라벨 수: {len(unique_labels)}")
            print(f"처음 20개 라벨: {unique_labels[:20]}")
            
            return segmentation_array, segmentation_image
            
        except Exception as e:
            print(f"분할 결과 로드 실패: {e}")
            return None, None
    
    def extract_organ_mask(self, segmentation_array, organ_name):
        """특정 장기의 마스크 추출"""
        if organ_name in self.bowel_related_organs:
            label_id = self.bowel_related_organs[organ_name]
        elif organ_name in self.vascular_organs:
            label_id = self.vascular_organs[organ_name]
        else:
            print(f"알 수 없는 장기: {organ_name}")
            return None
        
        organ_mask = (segmentation_array == label_id)
        organ_voxels = np.sum(organ_mask)
        
        print(f"{organ_name} 복셀 수: {organ_voxels}")
        
        if organ_voxels > 0:
            print(f"{organ_name} 영역 추출 성공")
            return organ_mask
        else:
            print(f"{organ_name} 영역을 찾을 수 없음")
            return None
    
    def get_3d_visualization_data(self, segmentation_array, organ_names=None):
        """3D 시각화를 위한 데이터 준비"""
        if organ_names is None:
            organ_names = ['small_bowel', 'colon', 'stomach', 'liver']
        
        visualization_data = {}
        
        for organ_name in organ_names:
            mask = self.extract_organ_mask(segmentation_array, organ_name)
            if mask is not None and np.sum(mask) > 0:
                visualization_data[organ_name] = {
                    'mask': mask,
                    'label_id': self.bowel_related_organs.get(organ_name, 0),
                    'color': self._get_organ_color(organ_name)
                }
        
        return visualization_data
    
    def _get_organ_color(self, organ_name):
        """장기별 시각화 색상 반환"""
        color_map = {
            'small_bowel': [1.0, 0.5, 0.0],  # 주황색
            'colon': [0.8, 0.4, 0.8],        # 보라색
            'stomach': [0.0, 1.0, 0.0],      # 녹색
            'liver': [0.5, 0.0, 0.0],        # 진한 빨강
            'pancreas': [1.0, 1.0, 0.0],     # 노랑
            'spleen': [0.0, 0.0, 1.0],       # 파랑
            'kidney_left': [0.0, 0.8, 0.8],  # 시안
            'kidney_right': [0.0, 0.6, 0.8], # 진한 시안
            'duodenum': [1.0, 0.8, 0.0],     # 금색
            'gallbladder': [0.0, 0.5, 0.0]   # 진한 녹색
        }
        return color_map.get(organ_name, [0.5, 0.5, 0.5])


def process_ct_for_visualization(ct_path):
    """
    CT 영상을 TotalSegmentator로 처리하여 시각화용 데이터 준비
    """
    wrapper = TotalSegmentatorWrapper()
    
    # 1. TotalSegmentator 실행
    seg_path = wrapper.run_segmentation(ct_path, task='total')
    
    if seg_path is None:
        print("분할 실패")
        return None
    
    # 2. 분할 결과 로드
    segmentation, seg_image = wrapper.load_segmentation(seg_path)
    
    if segmentation is None:
        print("분할 결과 로드 실패")
        return None
    
    # 3. 원본 CT 로드
    original_ct_image = sitk.ReadImage(ct_path)
    original_ct = sitk.GetArrayFromImage(original_ct_image)
    
    # 4. 3D 시각화 데이터 준비
    vis_data = wrapper.get_3d_visualization_data(segmentation)
    
    results = {
        'segmentation_path': seg_path,
        'segmentation_array': segmentation,
        'segmentation_image': seg_image,
        'original_ct': original_ct,
        'original_ct_image': original_ct_image,
        'visualization_data': vis_data,
        'wrapper': wrapper
    }
    
    return results

if __name__ == "__main__":
    # 테스트용
    test_ct = "../datasets/ct_images/sample_20250727_142714_232140_image.nii.gz"
    
    if os.path.exists(test_ct):
        print(f"테스트 CT 처리: {test_ct}")
        results = process_ct_for_visualization(test_ct)
        
        if results:
            print("처리 완료!")
            print(f"시각화 데이터 준비된 장기: {list(results['visualization_data'].keys())}")
        else:
            print("처리 실패")
    else:
        print(f"테스트 파일을 찾을 수 없음: {test_ct}")