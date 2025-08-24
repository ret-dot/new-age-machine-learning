"""Video Analyzer Package - Analyze videos using YOLO and generate summaries."""

from .analyzer import VideoAnalyzer
from .models import FrameAnalysis

__all__ = ['VideoAnalyzer', 'FrameAnalysis']
