"""Video summarization functionality."""
import json
from typing import List, Dict, Any
from pathlib import Path

class VideoSummarizer:
    """Generates summaries from video analysis results."""
    
    def generate_summary(self, analysis_file: str, model_name: str = "gpt-4") -> str:
        """Generate a summary from an analysis file."""
        # Load analysis data
        analysis_data = self._load_analysis(analysis_file)
        
        # Generate logical explanation
        logical_explanation = self._generate_logical_explanation(analysis_data)
        
        # Try to enhance with LLM if available
        try:
            from openai import OpenAI
            client = OpenAI()
            
            prompt = """You are a video analysis assistant. Below is a logical analysis of the video content:
            
            {}
            
            Please provide a more natural, human-readable summary based on this analysis. 
            Focus on creating a coherent narrative of what's happening in the video.
            """.format(logical_explanation)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes video content in a natural, engaging way."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Using logical analysis (LLM not available: {str(e)})")
            return logical_explanation
    
    def _load_analysis(self, analysis_file: str) -> List[Dict]:
        """Load analysis data from JSONL file."""
        import jsonlines
        data = []
        with jsonlines.open(analysis_file) as reader:
            for obj in reader:
                data.append(obj)
        return data
    
    def _generate_logical_explanation(self, analysis_data: List[Dict]) -> str:
        """Generate a logical explanation of the video content."""
        if not analysis_data:
            return "No significant content was detected in the video."
            
        # Analyze object occurrences
        object_stats = {}
        total_frames = len(analysis_data)
        
        for frame in analysis_data:
            for obj in frame['objects']:
                obj_class = obj['class']
                if obj_class not in object_stats:
                    object_stats[obj_class] = {
                        'count': 0,
                        'first_seen': frame['timestamp'],
                        'last_seen': frame['timestamp']
                    }
                object_stats[obj_class]['count'] += 1
                object_stats[obj_class]['last_seen'] = frame['timestamp']
        
        # Sort objects by frequency
        sorted_objects = sorted(
            object_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Generate explanation
        explanation = ["Video Content Analysis:", "=" * 40]
        
        # Main objects analysis
        if sorted_objects:
            main_objects = [obj for obj, _ in sorted_objects[:3] if object_stats[obj]['count'] > 10]
            if main_objects:
                explanation.append(f"\nMain Focus: The video primarily features {', '.join(main_objects[:-1])}{' and ' if len(main_objects) > 1 else ''}{main_objects[-1]}.")
        
        # Object statistics
        explanation.append("\nObject Statistics:")
        for obj, stats in sorted_objects[:5]:  # Top 5 objects
            percentage = (stats['count'] / total_frames) * 100
            duration = stats['last_seen'] - stats['first_seen']
            explanation.append(
                f"- {obj}: Appeared {stats['count']} times, "
                f"visible for {duration:.1f}s ({percentage:.1f}% of video)"
            )
        
        return "\n".join(explanation)
