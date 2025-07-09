"""
Main script for running player detection on both videos
======================================================

This script processes both broadcast and tacticam videos using the
advanced player detection system.
"""

import logging
import time
from pathlib import Path
import argparse

from player_detector import AdvancedPlayerDetector
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_video(video_path: str, output_dir: str, detector: AdvancedPlayerDetector, max_frames: int = None):
    """Process a single video file."""
    
    video_name = Path(video_path).stem
    logger.info(f"🎬 Starting detection for: {video_name}")
    
    start_time = time.time()
    stats = detector.process_video(video_path, output_dir, max_frames)
    processing_time = time.time() - start_time
    
    logger.info(f"✅ Completed {video_name} in {processing_time:.2f} seconds")
    logger.info(f"📊 Detection summary:")
    logger.info(f"   - Total detections: {stats['total_detections']}")
    logger.info(f"   - Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
    logger.info(f"   - Processing FPS: {stats['total_frames']/stats['processing_time']:.2f}")
    
    return stats

def main():
    """Main function to run player detection on both videos."""
    
    parser = argparse.ArgumentParser(description='Run player detection on videos')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='Maximum number of frames to process (default: all frames)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--videos', choices=['broadcast', 'tacticam', 'both'], default='both',
                       help='Which videos to process (default: both)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Player Detection System")
    logger.info("=" * 60)
    
    # Validate configuration
    try:
        config.validate_paths()
        config.create_output_dirs()
        logger.info("✅ Configuration validated")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return
    
    # Initialize detector
    try:
        detector = AdvancedPlayerDetector(
            model_path=str(config.YOLO_MODEL_PATH),
            confidence_threshold=args.confidence
        )
        logger.info("✅ Player detector initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        return
    
    # Process videos
    results = {}
    total_start_time = time.time()
    
    try:
        if args.videos in ['broadcast', 'both']:
            logger.info("🎥 Processing broadcast video...")
            results['broadcast'] = process_single_video(
                video_path=str(config.BROADCAST_VIDEO),
                output_dir=str(config.BROADCAST_RESULTS),
                detector=detector,
                max_frames=args.max_frames
            )
        
        if args.videos in ['tacticam', 'both']:
            logger.info("🎥 Processing tacticam video...")
            results['tacticam'] = process_single_video(
                video_path=str(config.TACTICAM_VIDEO),
                output_dir=str(config.TACTICAM_RESULTS),
                detector=detector,
                max_frames=args.max_frames
            )
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        return
    
    # Summary
    total_time = time.time() - total_start_time
    logger.info("🎉 Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"⏱️  Total processing time: {total_time:.2f} seconds")
    
    if 'broadcast' in results:
        broadcast_stats = results['broadcast']
        logger.info(f"📺 Broadcast video:")
        logger.info(f"   - Frames processed: {broadcast_stats['total_frames']}")
        logger.info(f"   - Total detections: {broadcast_stats['total_detections']}")
        logger.info(f"   - Avg detections/frame: {broadcast_stats['avg_detections_per_frame']:.2f}")
    
    if 'tacticam' in results:
        tacticam_stats = results['tacticam']
        logger.info(f"📹 Tacticam video:")
        logger.info(f"   - Frames processed: {tacticam_stats['total_frames']}")
        logger.info(f"   - Total detections: {tacticam_stats['total_detections']}")
        logger.info(f"   - Avg detections/frame: {tacticam_stats['avg_detections_per_frame']:.2f}")
    
    # Compare videos if both were processed
    if 'broadcast' in results and 'tacticam' in results:
        logger.info("🔍 Comparison:")
        b_avg = results['broadcast']['avg_detections_per_frame']
        t_avg = results['tacticam']['avg_detections_per_frame']
        logger.info(f"   - Detection rate difference: {abs(b_avg - t_avg):.2f} players/frame")
        
        if b_avg > t_avg:
            logger.info("   - Broadcast video has more detections on average")
        elif t_avg > b_avg:
            logger.info("   - Tacticam video has more detections on average")
        else:
            logger.info("   - Both videos have similar detection rates")
    
    logger.info("📁 Results saved in:")
    logger.info(f"   - Broadcast: {config.BROADCAST_RESULTS}")
    logger.info(f"   - Tacticam: {config.TACTICAM_RESULTS}")
    logger.info("📊 Analysis plots and detailed results available in each directory")

if __name__ == "__main__":
    main()
