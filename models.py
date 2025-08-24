"""Data models for video analysis."""
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

@dataclass
class FrameAnalysis:
    """Represents the analysis of a single video frame."""
    frame_number: int
    timestamp: float
    objects: List[Dict[str, Any]]
    scene_description: Optional[str] = None

@dataclass
class VideoAnalysisResult:
    """Container for video analysis results."""
    video_path: str
    total_frames: int
    duration_seconds: float
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary."""
        return {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'duration_seconds': self.duration_seconds,
            'frame_analyses': [asdict(f) for f in self.frame_analyses]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoAnalysisResult':
        """Create an instance from a dictionary."""
        return cls(
            video_path=data['video_path'],
            total_frames=data['total_frames'],
            duration_seconds=data['duration_seconds'],
            frame_analyses=[FrameAnalysis(**fa) for fa in data['frame_analyses']]
        )
