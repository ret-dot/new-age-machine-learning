"""Main video analysis functionality."""
import cv2
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from .models import FrameAnalysis, VideoAnalysisResult

class VideoAnalyzer:
    """Main class for video analysis using YOLO."""
    
    def __init__(self, model_name: str = 'yolov8x.pt'):
        """Initialize with a YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except ImportError as e:
            raise ImportError("Please install ultralytics: pip install ultralytics") from e
    
    def process_video(self, video_path: str, output_dir: str = 'output', 
                     frame_interval: int = 1, conf_threshold: float = 0.5) -> str:
        """Process a video and generate analysis."""
        from .processor import VideoProcessor
        return VideoProcessor(self.model).process(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=frame_interval,
            conf_threshold=conf_threshold
        )
    
    def generate_summary(self, analysis_file: str, model_name: str = "gpt-4") -> str:
        """Generate a summary from an analysis file."""
        from .summarizer import VideoSummarizer
        return VideoSummarizer().generate_summary(analysis_file, model_name)
