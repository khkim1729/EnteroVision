"""
UI용 실시간 로깅 시스템
콘솔 출력을 캡처하여 Streamlit UI에 표시
"""

import sys
import io
from contextlib import contextmanager
import streamlit as st
from datetime import datetime


class UILogger:
    """UI용 로거 클래스"""
    
    def __init__(self):
        self.logs = []
        self.errors = []
        
    def log(self, message, level="INFO"):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': str(message)
        }
        self.logs.append(log_entry)
        
        # 에러 레벨인 경우 별도 저장
        if level in ["ERROR", "WARNING"]:
            self.errors.append(log_entry)
            
        # 콘솔에도 출력
        print(f"[{timestamp}] {level}: {message}")
    
    def clear(self):
        """로그 초기화"""
        self.logs = []
        self.errors = []
    
    def get_logs(self):
        """모든 로그 반환"""
        return self.logs
    
    def get_errors(self):
        """에러 로그만 반환"""
        return self.errors
    
    def display_logs(self, show_all=False, container=None):
        """Streamlit에서 로그 표시"""
        if not self.logs:
            return
        
        if show_all:
            # 모든 로그 표시
            log_text = []
            for log in self.logs:
                emoji = self._get_emoji(log['level'])
                log_text.append(f"[{log['timestamp']}] {emoji} {log['message']}")
            
            st.text_area(
                "처리 로그",
                value="\n".join(log_text),
                height=300,
                key=f"logs_all_{len(self.logs)}"
            )
        else:
            # 최근 로그만 표시
            recent_logs = self.logs[-10:]  # 최근 10개만
            for log in recent_logs:
                emoji = self._get_emoji(log['level'])
                if log['level'] == "ERROR":
                    st.error(f"{emoji} {log['message']}")
                elif log['level'] == "WARNING":
                    st.warning(f"{emoji} {log['message']}")
                elif log['level'] == "SUCCESS":
                    st.success(f"{emoji} {log['message']}")
                else:
                    st.info(f"{emoji} {log['message']}")
    
    def display_realtime_logs(self, container, max_lines=20):
        """실시간 로그를 채팅창 스타일로 표시"""
        if not self.logs:
            container.info("📋 로그가 없습니다.")
            return
            
        # 최근 로그들만 표시 (성능을 위해)
        recent_logs = self.logs[-max_lines:] if len(self.logs) > max_lines else self.logs
        
        with container.container():
            # 로그를 역순으로 표시 (최신이 위로)
            for log in reversed(recent_logs):
                emoji = self._get_emoji(log['level'])
                timestamp = log['timestamp']
                message = log['message']
                level = log['level']
                
                # 레벨별 스타일링
                if level == "ERROR":
                    st.error(f"`{timestamp}` {emoji} {message}")
                elif level == "WARNING":
                    st.warning(f"`{timestamp}` {emoji} {message}")
                elif level == "SUCCESS":
                    st.success(f"`{timestamp}` {emoji} {message}")
                else:
                    st.info(f"`{timestamp}` {emoji} {message}")
    
    def get_log_summary(self):
        """로그 요약 정보 반환"""
        if not self.logs:
            return "로그 없음"
            
        total = len(self.logs)
        errors = len([log for log in self.logs if log['level'] == 'ERROR'])
        warnings = len([log for log in self.logs if log['level'] == 'WARNING'])
        successes = len([log for log in self.logs if log['level'] == 'SUCCESS'])
        
        return f"총 {total}개 (✅{successes} ⚠️{warnings} ❌{errors})"
    
    def _get_emoji(self, level):
        """로그 레벨별 이모지 반환"""
        emoji_map = {
            'INFO': '🔍',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'DEBUG': '🐛'
        }
        return emoji_map.get(level, 'ℹ️')


@contextmanager
def capture_output(logger):
    """표준 출력을 캡처하여 로거로 전달"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        yield logger
    finally:
        # 캡처된 출력 처리
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            for line in stdout_content.strip().split('\n'):
                if line.strip():
                    logger.log(line.strip(), "INFO")
        
        if stderr_content:
            for line in stderr_content.strip().split('\n'):
                if line.strip():
                    logger.log(line.strip(), "ERROR")
        
        # 표준 출력 복원
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# 전역 로거 인스턴스
ui_logger = UILogger()