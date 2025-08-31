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
        
        # TotalSegmentator v2 라벨 매핑 (104개 해부학 구조 - 최신 버전 기준)
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
            'small_bowel': 48,
            'duodenum': 49,
            'colon': 50,
            'rib_1_left': 51,
            'rib_1_right': 52,
            'rib_2_left': 53,
            'rib_2_right': 54,
            'rib_3_left': 55,
            'rib_3_right': 56,
            'rib_4_left': 57,
            'rib_4_right': 58,
            'rib_5_left': 59,
            'rib_5_right': 60,
            'rib_6_left': 61,
            'rib_6_right': 62,
            'rib_7_left': 63,
            'rib_7_right': 64,
            'rib_8_left': 65,
            'rib_8_right': 66,
            'rib_9_left': 67,
            'rib_9_right': 68,
            'rib_10_left': 69,
            'rib_10_right': 70,
            'rib_11_left': 71,
            'rib_11_right': 72,
            'rib_12_left': 73,
            'rib_12_right': 74,
            'humerus_left': 75,
            'humerus_right': 76,
            'scapula_left': 77,
            'scapula_right': 78,
            'clavicula_left': 79,
            'clavicula_right': 80,
            'femur_left': 81,
            'femur_right': 82,
            'hip_left': 83,
            'hip_right': 84,
            'sacrum': 85,
            'face': 86,
            'gluteus_maximus_left': 87,
            'gluteus_maximus_right': 88,
            'gluteus_medius_left': 89,
            'gluteus_medius_right': 90,
            'gluteus_minimus_left': 91,
            'gluteus_minimus_right': 92,
            'autochthon_left': 93,
            'autochthon_right': 94,
            'iliopsoas_left': 95,
            'iliopsoas_right': 96,
            'urinary_bladder': 97,
            'prostate': 98,
            'kidney_cyst_left': 99,
            'kidney_cyst_right': 100,
            'vertebrae_S1': 101,
            'vertebrae_Coccyx': 102,
            'trachea': 103,
            'esophagus': 104,
        }
        
        # 모든 장기를 포괄하는 통합 매핑
        self.bowel_related_organs = {}
        self.vascular_organs = {}
        
        # 모든 장기를 하나의 통합 딕셔너리로 관리
        for organ_name, label_id in self.totalseg_v2_labels.items():
            if organ_name != 'background':
                if any(keyword in organ_name for keyword in ['artery', 'vein', 'aorta', 'postcava', 'portal']):
                    self.vascular_organs[organ_name] = label_id
                else:
                    self.bowel_related_organs[organ_name] = label_id
        
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
        """3D 시각화를 위한 데이터 준비 - 모든 검출된 장기 자동 발견"""
        
        # 먼저 실제 존재하는 라벨들 확인
        unique_labels = np.unique(segmentation_array)
        print(f"🔍 Segmentation 파일의 라벨들: {unique_labels}")
        print(f"📊 총 {len(unique_labels)}개 라벨 발견")
        
        # 라벨-장기명 역방향 매핑 생성
        label_to_organ = {}
        all_organs = {**self.bowel_related_organs, **self.vascular_organs}
        
        for organ_name, label_id in all_organs.items():
            label_to_organ[label_id] = organ_name
        
        visualization_data = {}
        
        # 사용자가 특정 장기들을 지정한 경우
        if organ_names is not None:
            print(f"🎯 사용자 지정 장기: {organ_names}")
            for organ_name in organ_names:
                mask = self.extract_organ_mask(segmentation_array, organ_name)
                if mask is not None and np.sum(mask) > 0:
                    label_id = all_organs.get(organ_name, 0)
                    visualization_data[organ_name] = {
                        'mask': mask,
                        'label_id': label_id,
                        'color': self._get_organ_color(organ_name)
                    }
        else:
            # 자동 발견: segmentation 파일에 실제 존재하는 모든 장기 찾기
            print("🔍 자동 장기 검색 모드")
            
            for label_id in unique_labels:
                if label_id == 0:  # 배경 제외
                    continue
                    
                if label_id in label_to_organ:
                    organ_name = label_to_organ[label_id]
                    organ_mask = (segmentation_array == label_id)
                    voxel_count = np.sum(organ_mask)
                    
                    if voxel_count > 50:  # 최소 복셀 수 임계치
                        print(f"✅ {organ_name} 발견: {voxel_count:,} 복셀")
                        visualization_data[organ_name] = {
                            'mask': organ_mask,
                            'label_id': int(label_id),
                            'color': self._get_organ_color(organ_name)
                        }
                    else:
                        print(f"⚠️ {organ_name} 너무 작음: {voxel_count} 복셀")
                else:
                    # 알려진 매핑에 없는 라벨도 표시
                    voxel_count = np.sum(segmentation_array == label_id)
                    if voxel_count > 100:  # 더 높은 임계치
                        organ_name = f"unknown_label_{label_id}"
                        organ_mask = (segmentation_array == label_id)
                        print(f"🆕 알 수 없는 장기 발견: label_{label_id} ({voxel_count:,} 복셀)")
                        visualization_data[organ_name] = {
                            'mask': organ_mask,
                            'label_id': int(label_id),
                            'color': self._generate_random_color(label_id)
                        }
        
        print(f"🎉 최종 검출된 장기: {len(visualization_data)}개")
        for organ_name in visualization_data.keys():
            print(f"  - {organ_name}")
        
        return visualization_data
    
    def _get_organ_color(self, organ_name):
        """장기별 시각화 색상 반환 - 확장된 색상 매핑"""
        color_map = {
            # 소화기계
            'small_bowel': [1.0, 0.5, 0.0],      # 주황색
            'colon': [0.8, 0.4, 0.8],            # 보라색
            'stomach': [0.0, 1.0, 0.0],          # 녹색
            'liver': [0.5, 0.0, 0.0],            # 진한 빨강
            'pancreas': [1.0, 1.0, 0.0],         # 노랑
            'spleen': [0.0, 0.0, 1.0],           # 파랑
            'duodenum': [1.0, 0.8, 0.0],         # 금색
            'gallbladder': [0.0, 0.5, 0.0],      # 진한 녹색
            
            # 비뇨기계
            'kidney_left': [0.0, 0.8, 0.8],      # 시안
            'kidney_right': [0.0, 0.6, 0.8],     # 진한 시안
            'adrenal_gland_left': [0.5, 0.8, 0.5],   # 연한 녹색
            'adrenal_gland_right': [0.3, 0.8, 0.3],  # 진한 연녹색
            
            # 호흡기계
            'lung_upper_lobe_left': [0.8, 0.2, 0.2],     # 밝은 빨강
            'lung_lower_lobe_left': [0.6, 0.2, 0.2],     # 어두운 빨강
            'lung_upper_lobe_right': [0.8, 0.4, 0.2],    # 주황빨강
            'lung_middle_lobe_right': [0.8, 0.6, 0.2],   # 황금색
            'lung_lower_lobe_right': [0.6, 0.4, 0.2],    # 갈색
            
            # 순환기계
            'heart': [1.0, 0.0, 0.0],            # 빨강
            'aorta': [0.8, 0.0, 0.4],            # 진한 분홍
            'postcava': [0.4, 0.0, 0.8],         # 보라
            'portal_vein_splenic_vein': [0.6, 0.0, 0.6],  # 자주색
            'iliac_artery_left': [1.0, 0.2, 0.4],         # 분홍
            'iliac_artery_right': [1.0, 0.4, 0.2],        # 연한 주황
            'iliac_vena_left': [0.2, 0.2, 1.0],           # 파랑
            'iliac_vena_right': [0.4, 0.2, 1.0],          # 연한 파랑
            
            # 신경계
            'brain': [1.0, 0.8, 0.8],            # 연한 분홍
            
            # 척추 (예시로 몇 개만)
            'vertebrae_C1': [0.9, 0.9, 0.7],     # 아이보리
            'vertebrae_C7': [0.8, 0.8, 0.6],     # 베이지
            'vertebrae_T1': [0.7, 0.7, 0.5],     # 황갈색
            'vertebrae_T12': [0.6, 0.6, 0.4],    # 갈색
            'vertebrae_L1': [0.5, 0.5, 0.3],     # 어두운 갈색
            'vertebrae_L5': [0.4, 0.4, 0.2],     # 매우 어두운 갈색
        }
        
        # 척추 기본 색상 (개별 매핑이 없는 경우)
        if organ_name.startswith('vertebrae_'):
            return [0.7, 0.6, 0.4]  # 뼈 색상
        
        return color_map.get(organ_name, [0.5, 0.5, 0.5])  # 기본 회색
    
    def _generate_random_color(self, label_id):
        """라벨 ID 기반 일관된 랜덤 색상 생성"""
        np.random.seed(int(label_id) * 42)  # 일관된 색상을 위한 시드
        return np.random.rand(3).tolist()


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
    
    # 4. 3D 시각화 데이터 준비 - 모든 장기 자동 발견
    vis_data = wrapper.get_3d_visualization_data(segmentation, organ_names=None)
    
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