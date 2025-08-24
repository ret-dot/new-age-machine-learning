"""Video processing functionality."""
import cv2
import jsonlines
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime
from .models import FrameAnalysis, VideoAnalysisResult

class VideoProcessor:
    """Handles video processing and frame analysis."""
    
    def __init__(self, model):
        self.model = model
    
    def process(self, video_path: str, output_dir: str, 
               frame_interval: int, conf_threshold: float) -> str:
        """Process video and save analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_path / f'video_analysis_{timestamp}.jsonl'
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        processed_count = 0
        
        with jsonlines.open(output_file, mode='w') as writer:
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.model(frame_rgb, conf=conf_threshold, verbose=False)
                        
                        # Extract detections
                        detections = []
                        for result in results:
                            for box in result.boxes:
                                detections.append({
                                    'class': self.model.names[int(box.cls)],
                                    'confidence': float(box.conf),
                                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                                })
                        
                        # Create frame analysis
                        frame_analysis = FrameAnalysis(
                            frame_number=frame_count,
                            timestamp=frame_count / fps,
                            objects=detections
                        )
                        
                        # Write to JSONL file
                        writer.write({
                            'frame_number': frame_analysis.frame_number,
                            'timestamp': frame_analysis.timestamp,
                            'objects': frame_analysis.objects
                        })
                        
                        processed_count += 1
                        
                    frame_count += 1
                    pbar.update(1)
                    
                    if frame_count >= total_frames:
                        break
                        
        cap.release()
        print(f"\nProcessed {processed_count} frames. Results saved to: {output_file}")
        return str(output_file)
