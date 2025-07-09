"""
Advanced Player Detection System
===============================

This module implements a sophisticated player detection system using YOLOv11
with additional creative features for robust player re-identification.

Features:
- Multi-scale detection with confidence filtering
- Player crop extraction and enhancement
- Spatial-temporal feature extraction
- Team color analysis
- Movement pattern tracking
- Quality assessment of detections
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerDetection:
    """Data class to store comprehensive player detection information."""
    
    # Basic detection info
    frame_number: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]      # (center_x, center_y)
    
    # Visual features
    player_crop: Optional[np.ndarray] = None
    dominant_colors: Optional[List[Tuple[int, int, int]]] = None
    color_histogram: Optional[np.ndarray] = None
    texture_features: Optional[np.ndarray] = None
    
    # Spatial features
    bbox_area: Optional[float] = None
    aspect_ratio: Optional[float] = None
    position_normalized: Optional[Tuple[float, float]] = None
    
    # Quality metrics
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    occlusion_score: Optional[float] = None
    
    # Temporal info
    timestamp: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.player_crop is not None:
            data['player_crop'] = self.player_crop.tolist()
        if self.color_histogram is not None:
            data['color_histogram'] = self.color_histogram.tolist()
        if self.texture_features is not None:
            data['texture_features'] = self.texture_features.tolist()
        
        # Convert numpy scalars to Python types
        for key, value in data.items():
            if hasattr(value, 'item'):  # numpy scalar
                data[key] = value.item()
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, tuple) and len(value) > 0 and hasattr(value[0], 'item'):
                data[key] = tuple(v.item() if hasattr(v, 'item') else v for v in value)
        
        return data

class AdvancedPlayerDetector:
    """Advanced player detection system with creative features."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the advanced player detector.
        
        Args:
            model_path: Path to the YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = config.get_device()
        
        # Move model to appropriate device
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to(self.device)
        
        logger.info(f"Player detector initialized on device: {self.device}")
        
    def detect_players_in_frame(self, frame: np.ndarray, frame_number: int) -> List[PlayerDetection]:
        """
        Detect players in a single frame with comprehensive feature extraction.
        
        Args:
            frame: Input frame
            frame_number: Frame number for tracking
            
        Returns:
            List of PlayerDetection objects
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract basic detection info
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Create player detection object
                    detection = PlayerDetection(
                        frame_number=frame_number,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y)
                    )
                    
                    # Extract player crop
                    player_crop = self._extract_player_crop(frame, x1, y1, x2, y2)
                    if player_crop is not None:
                        detection.player_crop = player_crop
                        
                        # Extract comprehensive features
                        self._extract_visual_features(detection, player_crop)
                        self._extract_spatial_features(detection, frame.shape)
                        self._extract_quality_metrics(detection, player_crop)
                    
                    detections.append(detection)
        
        return detections
    
    def _extract_player_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """Extract and enhance player crop from frame."""
        try:
            # Add padding to capture more context
            h, w = frame.shape[:2]
            padding = 10
            
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Resize to standard size for consistency
            if crop.size > 0:
                crop = cv2.resize(crop, (128, 256))  # Standard person aspect ratio
                return crop
            
        except Exception as e:
            logger.warning(f"Failed to extract crop: {e}")
        
        return None
    
    def _extract_visual_features(self, detection: PlayerDetection, crop: np.ndarray):
        """Extract comprehensive visual features from player crop."""
        
        # 1. Dominant colors using K-means clustering
        detection.dominant_colors = self._extract_dominant_colors(crop)
        
        # 2. Color histogram in HSV space (more robust to lighting)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv_crop], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_crop], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_crop], [2], None, [256], [0, 256])
        
        # Normalize and concatenate
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        detection.color_histogram = np.concatenate([h_hist, s_hist, v_hist])
        
        # 3. Texture features using LBP (Local Binary Patterns)
        detection.texture_features = self._extract_texture_features(crop)
    
    def _extract_dominant_colors(self, crop: np.ndarray, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering."""
        try:
            # Reshape image to be a list of pixels
            pixels = crop.reshape(-1, 3)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by frequency (number of pixels assigned to each cluster)
            labels = kmeans.labels_
            counts = np.bincount(labels)
            sorted_indices = np.argsort(counts)[::-1]
            
            dominant_colors = [tuple(colors[i]) for i in sorted_indices]
            return dominant_colors
            
        except Exception as e:
            logger.warning(f"Failed to extract dominant colors: {e}")
            return []
    
    def _extract_texture_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract texture features using Local Binary Patterns."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Calculate LBP
            from skimage.feature import local_binary_pattern
            
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate LBP histogram
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            return hist
            
        except Exception as e:
            logger.warning(f"Failed to extract texture features: {e}")
            return np.array([])
    
    def _extract_spatial_features(self, detection: PlayerDetection, frame_shape: Tuple[int, int]):
        """Extract spatial features from detection."""
        x1, y1, x2, y2 = detection.bbox
        h, w = frame_shape[:2]
        
        # Bounding box area
        detection.bbox_area = (x2 - x1) * (y2 - y1)
        
        # Aspect ratio
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        detection.aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        
        # Normalized position (relative to frame size)
        detection.position_normalized = (detection.center[0] / w, detection.center[1] / h)
    
    def _extract_quality_metrics(self, detection: PlayerDetection, crop: np.ndarray):
        """Extract quality metrics for the detection."""
        
        # 1. Blur detection using Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        detection.blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness
        detection.brightness = np.mean(gray)
        
        # 3. Contrast (standard deviation of pixel intensities)
        detection.contrast = np.std(gray)
        
        # 4. Simple occlusion estimation (based on edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        detection.occlusion_score = 1.0 - edge_density  # Higher value = more occlusion
    
    def process_video(self, video_path: str, output_dir: str, max_frames: Optional[int] = None) -> Dict:
        """
        Process entire video and extract player detections.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary containing processing statistics and results
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Total frames to process: {total_frames}")
        logger.info(f"Video FPS: {fps}")
        
        all_detections = []
        processing_stats = {
            'video_name': video_path.name,
            'total_frames': total_frames,
            'fps': fps,
            'detections_per_frame': [],
            'processing_time': 0,
            'total_detections': 0
        }
        
        start_time = time.time()
        
        # Process frames
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_number >= max_frames):
                    break
                
                # Detect players in current frame
                frame_detections = self.detect_players_in_frame(frame, frame_number)
                all_detections.extend(frame_detections)
                
                processing_stats['detections_per_frame'].append(len(frame_detections))
                
                # Save annotated frame occasionally for visualization
                if frame_number % 30 == 0:  # Every 30 frames
                    annotated_frame = self._annotate_frame(frame, frame_detections)
                    cv2.imwrite(str(output_dir / f"frame_{frame_number:06d}_annotated.jpg"), annotated_frame)
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
        
        # Calculate processing statistics
        processing_stats['processing_time'] = time.time() - start_time
        processing_stats['total_detections'] = len(all_detections)
        processing_stats['avg_detections_per_frame'] = np.mean(processing_stats['detections_per_frame'])
        processing_stats['max_detections_per_frame'] = max(processing_stats['detections_per_frame']) if processing_stats['detections_per_frame'] else 0
        
        # Save results
        self._save_results(all_detections, processing_stats, output_dir)
        
        # Generate analysis visualizations
        self._generate_analysis_plots(all_detections, processing_stats, output_dir)
        
        logger.info(f"Processing complete! Total detections: {len(all_detections)}")
        logger.info(f"Processing time: {processing_stats['processing_time']:.2f} seconds")
        logger.info(f"Average FPS: {total_frames / processing_stats['processing_time']:.2f}")
        
        return processing_stats
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[PlayerDetection]) -> np.ndarray:
        """Annotate frame with detection results."""
        annotated = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(annotated, center, 3, (0, 0, 255), -1)
            
            # Add confidence and quality info
            text = f"ID:{i} C:{detection.confidence:.2f}"
            if detection.blur_score is not None:
                text += f" B:{detection.blur_score:.0f}"
            
            cv2.putText(annotated, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _save_results(self, detections: List[PlayerDetection], stats: Dict, output_dir: Path):
        """Save detection results and statistics."""
        
        # Save detections as pickle (preserves numpy arrays)
        with open(output_dir / 'detections.pkl', 'wb') as f:
            pickle.dump(detections, f)
        
        # Save detections as JSON (for human readability) - only basic info
        detections_json = []
        for det in detections:
            basic_info = {
                'frame_number': int(det.frame_number),
                'confidence': float(det.confidence),
                'bbox': [int(x) for x in det.bbox],
                'center': [float(x) for x in det.center],
                'bbox_area': float(det.bbox_area) if det.bbox_area is not None else None,
                'aspect_ratio': float(det.aspect_ratio) if det.aspect_ratio is not None else None,
                'position_normalized': [float(x) for x in det.position_normalized] if det.position_normalized else None,
                'blur_score': float(det.blur_score) if det.blur_score is not None else None,
                'brightness': float(det.brightness) if det.brightness is not None else None,
                'contrast': float(det.contrast) if det.contrast is not None else None,
                'occlusion_score': float(det.occlusion_score) if det.occlusion_score is not None else None
            }
            detections_json.append(basic_info)
        
        with open(output_dir / 'detections.json', 'w') as f:
            json.dump(detections_json, f, indent=2)
        
        # Save statistics
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to: {output_dir}")
    
    def _generate_analysis_plots(self, detections: List[PlayerDetection], stats: Dict, output_dir: Path):
        """Generate analysis and visualization plots."""
        
        if not detections:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Player Detection Analysis - {stats['video_name']}", fontsize=16)
        
        # 1. Detections per frame
        axes[0, 0].plot(stats['detections_per_frame'])
        axes[0, 0].set_title('Detections per Frame')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].grid(True)
        
        # 2. Confidence distribution
        confidences = [det.confidence for det in detections]
        axes[0, 1].hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 1].legend()
        
        # 3. Spatial distribution (player positions)
        positions = [det.position_normalized for det in detections if det.position_normalized]
        if positions:
            x_pos, y_pos = zip(*positions)
            axes[0, 2].scatter(x_pos, y_pos, alpha=0.5)
            axes[0, 2].set_title('Player Position Distribution')
            axes[0, 2].set_xlabel('Normalized X Position')
            axes[0, 2].set_ylabel('Normalized Y Position')
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].invert_yaxis()  # Invert Y to match image coordinates
        
        # 4. Quality metrics
        blur_scores = [det.blur_score for det in detections if det.blur_score is not None]
        if blur_scores:
            axes[1, 0].hist(blur_scores, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Blur Score Distribution')
            axes[1, 0].set_xlabel('Blur Score (higher = sharper)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. Bounding box area distribution
        areas = [det.bbox_area for det in detections if det.bbox_area is not None]
        if areas:
            axes[1, 1].hist(areas, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Bounding Box Area Distribution')
            axes[1, 1].set_xlabel('Area (pixels²)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
        Summary Statistics:
        
        Total Detections: {len(detections)}
        Total Frames: {stats['total_frames']}
        Avg Detections/Frame: {stats['avg_detections_per_frame']:.2f}
        Max Detections/Frame: {stats['max_detections_per_frame']}
        
        Processing Time: {stats['processing_time']:.2f}s
        Processing FPS: {stats['total_frames']/stats['processing_time']:.2f}
        
        Confidence Stats:
        Mean: {np.mean(confidences):.3f}
        Std: {np.std(confidences):.3f}
        Min: {np.min(confidences):.3f}
        Max: {np.max(confidences):.3f}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis plots saved to: {output_dir / 'analysis_plots.png'}")
