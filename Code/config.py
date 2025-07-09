"""
Configuration file for Player Re-identification Project
======================================================

This file contains all the configuration parameters and settings
for the player re-identification system.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the player re-identification project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    VIDEOS_DIR = PROJECT_ROOT / "Videos"
    MODEL_DIR = PROJECT_ROOT / "Model"
    RESULTS_DIR = PROJECT_ROOT / "Results"
    CODE_DIR = PROJECT_ROOT / "Code"
    
    # Video files
    BROADCAST_VIDEO = VIDEOS_DIR / "broadcast.mp4"
    TACTICAM_VIDEO = VIDEOS_DIR / "tacticam.mp4"
    
    # Model file
    YOLO_MODEL_PATH = MODEL_DIR / "best.pt"
    
    # Output directories
    BROADCAST_RESULTS = RESULTS_DIR / "broadcast"
    TACTICAM_RESULTS = RESULTS_DIR / "tacticam"
    MATCHED_RESULTS = RESULTS_DIR / "matched"
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # Video processing parameters
    FRAME_SKIP = 1  # Process every nth frame (1 = every frame)
    MAX_FRAMES = None  # None = process entire video
    
    # Player matching parameters
    FEATURE_EXTRACTION_METHOD = "color_histogram"  # Options: color_histogram, deep_features, hog
    MATCHING_ALGORITHM = "hungarian"  # Options: hungarian, greedy, similarity_threshold
    SIMILARITY_THRESHOLD = 0.7
    
    # Tracking parameters
    MAX_DISAPPEARED_FRAMES = 30
    MAX_DISTANCE_THRESHOLD = 100
    
    # Visualization parameters
    DRAW_BBOXES = True
    DRAW_IDS = True
    SAVE_ANNOTATED_FRAMES = True
    
    # Performance parameters
    USE_GPU = True  # Use GPU if available
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    
    @classmethod
    def validate_paths(cls):
        """Validate that all required paths exist."""
        required_paths = [
            cls.VIDEOS_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR,
            cls.BROADCAST_VIDEO,
            cls.TACTICAM_VIDEO,
            cls.YOLO_MODEL_PATH
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            raise FileNotFoundError(f"Missing required paths: {missing_paths}")
        
        return True
    
    @classmethod
    def create_output_dirs(cls):
        """Create output directories if they don't exist."""
        output_dirs = [
            cls.BROADCAST_RESULTS,
            cls.TACTICAM_RESULTS,
            cls.MATCHED_RESULTS
        ]
        
        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Get the device to use for computation (GPU/CPU)."""
        import torch
        
        if cls.USE_GPU and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("🔧 Player Re-identification Configuration")
        print("=" * 50)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Videos Directory: {cls.VIDEOS_DIR}")
        print(f"Model Directory: {cls.MODEL_DIR}")
        print(f"Results Directory: {cls.RESULTS_DIR}")
        print(f"\nVideo Files:")
        print(f"  Broadcast: {cls.BROADCAST_VIDEO}")
        print(f"  Tacticam: {cls.TACTICAM_VIDEO}")
        print(f"\nModel: {cls.YOLO_MODEL_PATH}")
        print(f"\nDetection Parameters:")
        print(f"  Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"  IOU Threshold: {cls.IOU_THRESHOLD}")
        print(f"\nProcessing Parameters:")
        print(f"  Frame Skip: {cls.FRAME_SKIP}")
        print(f"  Max Frames: {cls.MAX_FRAMES}")
        print(f"  Device: {cls.get_device()}")
        print(f"\nMatching Parameters:")
        print(f"  Feature Method: {cls.FEATURE_EXTRACTION_METHOD}")
        print(f"  Matching Algorithm: {cls.MATCHING_ALGORITHM}")
        print(f"  Similarity Threshold: {cls.SIMILARITY_THRESHOLD}")

# Create global config instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    try:
        config.validate_paths()
        config.create_output_dirs()
        config.print_config()
        print("\n✅ Configuration validated successfully!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
