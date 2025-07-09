"""
Advanced Player Feature Extraction System
=========================================

This module implements sophisticated feature extraction for player re-identification
across different camera views, focusing on robust features that work despite
camera angle differences, lighting variations, and perspective changes.

Key Focus Areas:
- Jersey color analysis (team identification)
- Body proportions and size estimation
- Field position context
- Movement patterns
- Robust color features for different lighting
- Geometric features invariant to camera angles
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import logging
from tqdm import tqdm

from player_detector import PlayerDetection
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerFeatures:
    """Enhanced player features for re-identification across camera views."""
    
    # Basic identification
    detection_id: str
    frame_number: int
    video_source: str  # 'broadcast' or 'tacticam'
    
    # Jersey and appearance features
    primary_jersey_color: Optional[Tuple[int, int, int]] = None
    secondary_jersey_color: Optional[Tuple[int, int, int]] = None
    jersey_color_confidence: Optional[float] = None
    dominant_colors_robust: Optional[List[Tuple[int, int, int]]] = None
    color_distribution: Optional[np.ndarray] = None
    
    # Body and size features
    player_height_ratio: Optional[float] = None  # Height relative to image
    player_width_ratio: Optional[float] = None   # Width relative to image
    aspect_ratio: Optional[float] = None         # Height/Width ratio
    body_area_normalized: Optional[float] = None # Area relative to image
    estimated_distance: Optional[float] = None   # Distance from camera (estimated)
    
    # Field position features
    field_position_x: Optional[float] = None     # Normalized X position (0-1)
    field_position_y: Optional[float] = None     # Normalized Y position (0-1)
    field_zone: Optional[str] = None             # Defensive/Midfield/Offensive zone
    relative_position: Optional[str] = None      # Top/Center/Bottom of frame
    
    # Geometric features (camera angle invariant)
    centroid_normalized: Optional[Tuple[float, float]] = None
    bounding_box_normalized: Optional[Tuple[float, float, float, float]] = None
    
    # Advanced appearance features
    texture_entropy: Optional[float] = None      # Texture complexity
    edge_density: Optional[float] = None         # Edge information
    brightness_profile: Optional[np.ndarray] = None  # Brightness distribution
    color_coherence: Optional[float] = None      # Color consistency measure
    
    # Temporal features
    timestamp: Optional[float] = None
    temporal_id: Optional[int] = None            # For tracking across frames
    
    # Quality and reliability
    feature_quality_score: Optional[float] = None
    extraction_confidence: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Handle numpy arrays and other non-serializable types
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif hasattr(value, 'item'):  # numpy scalar
                data[key] = value.item()
            elif isinstance(value, tuple) and len(value) > 0:
                # Handle tuples with numpy values
                data[key] = [v.item() if hasattr(v, 'item') else int(v) if isinstance(v, (np.uint8, np.int64, np.int32)) else v for v in value]
            elif isinstance(value, (np.uint8, np.int64, np.int32, np.float32, np.float64)):
                data[key] = int(value) if isinstance(value, (np.uint8, np.int64, np.int32)) else float(value)
        return data

class AdvancedFeatureExtractor:
    """Advanced feature extraction system for player re-identification."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.field_zones = self._define_field_zones()
        self.color_spaces = ['RGB', 'HSV', 'LAB']
        logger.info("Advanced feature extractor initialized")
    
    def _define_field_zones(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Define field zones for position classification."""
        return {
            'defensive_left': (0.0, 0.0, 0.33, 1.0),
            'defensive_center': (0.33, 0.0, 0.66, 1.0),
            'defensive_right': (0.66, 0.0, 1.0, 1.0),
            'midfield_left': (0.0, 0.33, 0.33, 0.66),
            'midfield_center': (0.33, 0.33, 0.66, 0.66),
            'midfield_right': (0.66, 0.33, 1.0, 0.66),
            'offensive_left': (0.0, 0.66, 0.33, 1.0),
            'offensive_center': (0.33, 0.66, 0.66, 1.0),
            'offensive_right': (0.66, 0.66, 1.0, 1.0)
        }
    
    def extract_enhanced_features(self, detection: PlayerDetection, frame: np.ndarray, 
                                video_source: str) -> PlayerFeatures:
        """
        Extract comprehensive features from a player detection.
        
        Args:
            detection: PlayerDetection object
            frame: Full frame image
            video_source: 'broadcast' or 'tacticam'
            
        Returns:
            PlayerFeatures object with extracted features
        """
        # Create feature object
        features = PlayerFeatures(
            detection_id=f"{video_source}_{detection.frame_number}_{detection.center[0]:.0f}_{detection.center[1]:.0f}",
            frame_number=detection.frame_number,
            video_source=video_source,
            timestamp=detection.timestamp
        )
        
        # Extract player crop
        player_crop = self._get_enhanced_player_crop(frame, detection)
        if player_crop is None:
            return features
        
        # Extract various feature categories
        self._extract_jersey_features(player_crop, features)
        self._extract_body_size_features(detection, frame.shape, features)
        self._extract_field_position_features(detection, frame.shape, features)
        self._extract_geometric_features(detection, frame.shape, features)
        self._extract_advanced_appearance_features(player_crop, features)
        self._calculate_quality_metrics(player_crop, detection, features)
        
        return features
    
    def _get_enhanced_player_crop(self, frame: np.ndarray, detection: PlayerDetection) -> Optional[np.ndarray]:
        """Extract enhanced player crop with context."""
        try:
            x1, y1, x2, y2 = detection.bbox
            h, w = frame.shape[:2]
            
            # Add contextual padding (more for jersey analysis)
            padding_x = int((x2 - x1) * 0.15)  # 15% padding
            padding_y = int((y2 - y1) * 0.1)   # 10% padding
            
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(w, x2 + padding_x)
            y2_pad = min(h, y2 + padding_y)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Ensure minimum size for feature extraction
            if crop.shape[0] < 32 or crop.shape[1] < 16:
                return None
            
            return crop
            
        except Exception as e:
            logger.warning(f"Failed to extract enhanced crop: {e}")
            return None
    
    def _extract_jersey_features(self, player_crop: np.ndarray, features: PlayerFeatures):
        """Extract robust jersey color features."""
        try:
            # Focus on upper body for jersey analysis (top 60% of crop)
            upper_body = player_crop[:int(player_crop.shape[0] * 0.6), :]
            
            # Convert to multiple color spaces for robustness
            hsv_crop = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            lab_crop = cv2.cvtColor(upper_body, cv2.COLOR_BGR2LAB)
            
            # Extract dominant colors using improved clustering
            jersey_colors = self._extract_jersey_colors_robust(upper_body, hsv_crop)
            features.primary_jersey_color = jersey_colors.get('primary')
            features.secondary_jersey_color = jersey_colors.get('secondary')
            features.jersey_color_confidence = jersey_colors.get('confidence', 0.0)
            
            # Color distribution analysis
            features.color_distribution = self._analyze_color_distribution(upper_body)
            
            # Robust dominant colors (lighting invariant)
            features.dominant_colors_robust = self._extract_lighting_invariant_colors(upper_body)
            
        except Exception as e:
            logger.warning(f"Jersey feature extraction failed: {e}")
    
    def _extract_jersey_colors_robust(self, rgb_crop: np.ndarray, hsv_crop: np.ndarray) -> Dict:
        """Extract jersey colors with improved robustness."""
        try:
            # Mask out potential skin tones and background
            hsv_masked = self._mask_non_jersey_areas(hsv_crop)
            
            if np.sum(hsv_masked) == 0:
                return {}
            
            # Extract valid pixels
            valid_pixels = rgb_crop[hsv_masked > 0]
            
            if len(valid_pixels) < 50:  # Need sufficient pixels
                return {}
            
            # Cluster colors in LAB space (perceptually uniform)
            lab_pixels = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            
            # Use adaptive clustering
            n_clusters = min(5, max(2, len(lab_pixels) // 100))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(lab_pixels)
            
            # Get cluster centers and sizes
            centers_lab = kmeans.cluster_centers_
            cluster_sizes = np.bincount(labels)
            
            # Convert back to RGB
            centers_rgb = cv2.cvtColor(centers_lab.reshape(-1, 1, 3).astype(np.uint8), 
                                     cv2.COLOR_LAB2RGB).reshape(-1, 3)
            
            # Sort by cluster size
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            
            # Calculate confidence based on cluster separation
            confidence = self._calculate_color_confidence(centers_lab, cluster_sizes)
            
            return {
                'primary': tuple(centers_rgb[sorted_indices[0]]) if len(sorted_indices) > 0 else None,
                'secondary': tuple(centers_rgb[sorted_indices[1]]) if len(sorted_indices) > 1 else None,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Robust jersey color extraction failed: {e}")
            return {}
    
    def _mask_non_jersey_areas(self, hsv_crop: np.ndarray) -> np.ndarray:
        """Create mask to exclude skin tones and likely background areas."""
        # Define skin tone ranges in HSV
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        lower_skin2 = np.array([160, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        
        # Create skin mask
        skin_mask1 = cv2.inRange(hsv_crop, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv_crop, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Create brightness mask (exclude very bright/dark areas)
        _, _, v_channel = cv2.split(hsv_crop)
        brightness_mask = cv2.inRange(v_channel, 30, 220)
        
        # Combine masks (valid areas are NOT skin and have reasonable brightness)
        valid_mask = cv2.bitwise_and(cv2.bitwise_not(skin_mask), brightness_mask)
        
        return valid_mask
    
    def _calculate_color_confidence(self, centers_lab: np.ndarray, cluster_sizes: np.ndarray) -> float:
        """Calculate confidence in color extraction based on cluster separation."""
        if len(centers_lab) < 2:
            return 0.5
        
        # Calculate distances between cluster centers
        distances = cdist(centers_lab, centers_lab)
        distances[distances == 0] = np.inf  # Ignore self-distances
        
        min_distance = np.min(distances)
        avg_distance = np.mean(distances[distances != np.inf])
        
        # Higher confidence for well-separated clusters
        separation_score = min(1.0, min_distance / 50.0)  # Normalize to [0, 1]
        
        # Higher confidence for dominant clusters
        size_ratio = np.max(cluster_sizes) / np.sum(cluster_sizes)
        dominance_score = min(1.0, size_ratio * 2)  # Normalize to [0, 1]
        
        return (separation_score + dominance_score) / 2
    
    def _analyze_color_distribution(self, crop: np.ndarray) -> np.ndarray:
        """Analyze color distribution for jersey pattern recognition."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Calculate 3D histogram
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [30, 32, 32], 
                              [0, 180, 0, 256, 0, 256])
            
            # Normalize histogram
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-7)
            
            return hist
            
        except Exception:
            return np.array([])
    
    def _extract_lighting_invariant_colors(self, crop: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract colors that are robust to lighting changes."""
        try:
            # Convert to LAB color space (perceptually uniform)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            
            # Normalize L channel to reduce lighting effects
            lab_normalized = lab.copy()
            lab_normalized[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            
            # Convert back to RGB
            rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
            
            # Extract dominant colors from normalized image
            pixels = rgb_normalized.reshape(-1, 3)
            
            # Sample pixels for efficiency
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]
            
            # Cluster colors
            kmeans = KMeans(n_clusters=min(5, len(pixels) // 50), random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Return dominant colors sorted by frequency
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            counts = np.bincount(labels)
            
            sorted_indices = np.argsort(counts)[::-1]
            return [tuple(centers[i].astype(int)) for i in sorted_indices]
            
        except Exception:
            return []
    
    def _extract_body_size_features(self, detection: PlayerDetection, frame_shape: Tuple[int, int], 
                                  features: PlayerFeatures):
        """Extract body size and proportion features."""
        try:
            x1, y1, x2, y2 = detection.bbox
            frame_height, frame_width = frame_shape[:2]
            
            # Basic size measurements
            player_width = x2 - x1
            player_height = y2 - y1
            
            # Normalized size features
            features.player_height_ratio = player_height / frame_height
            features.player_width_ratio = player_width / frame_width
            features.aspect_ratio = player_height / max(player_width, 1)
            features.body_area_normalized = (player_width * player_height) / (frame_width * frame_height)
            
            # Estimate distance based on player size (larger = closer)
            # This is a rough approximation based on typical player sizes
            avg_player_height_ratio = 0.15  # Typical player height ratio in frame
            features.estimated_distance = avg_player_height_ratio / max(features.player_height_ratio, 0.01)
            
        except Exception as e:
            logger.warning(f"Body size feature extraction failed: {e}")
    
    def _extract_field_position_features(self, detection: PlayerDetection, frame_shape: Tuple[int, int],
                                       features: PlayerFeatures):
        """Extract field position and contextual features."""
        try:
            frame_height, frame_width = frame_shape[:2]
            center_x, center_y = detection.center
            
            # Normalized field position
            features.field_position_x = center_x / frame_width
            features.field_position_y = center_y / frame_height
            
            # Determine field zone
            features.field_zone = self._classify_field_zone(features.field_position_x, features.field_position_y)
            
            # Relative position in frame
            if features.field_position_y < 0.33:
                features.relative_position = "top"
            elif features.field_position_y < 0.66:
                features.relative_position = "center"
            else:
                features.relative_position = "bottom"
                
        except Exception as e:
            logger.warning(f"Field position feature extraction failed: {e}")
    
    def _classify_field_zone(self, x: float, y: float) -> str:
        """Classify player position into field zones."""
        for zone_name, (x_min, y_min, x_max, y_max) in self.field_zones.items():
            if x_min <= x < x_max and y_min <= y < y_max:
                return zone_name
        return "unknown"
    
    def _extract_geometric_features(self, detection: PlayerDetection, frame_shape: Tuple[int, int],
                                  features: PlayerFeatures):
        """Extract geometric features that are camera angle invariant."""
        try:
            x1, y1, x2, y2 = detection.bbox
            frame_height, frame_width = frame_shape[:2]
            
            # Normalized centroid
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height
            features.centroid_normalized = (center_x, center_y)
            
            # Normalized bounding box
            bbox_norm = (x1 / frame_width, y1 / frame_height, 
                        x2 / frame_width, y2 / frame_height)
            features.bounding_box_normalized = bbox_norm
            
        except Exception as e:
            logger.warning(f"Geometric feature extraction failed: {e}")
    
    def _extract_advanced_appearance_features(self, player_crop: np.ndarray, features: PlayerFeatures):
        """Extract advanced appearance features."""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
            
            # Texture entropy (measure of texture complexity)
            features.texture_entropy = self._calculate_texture_entropy(gray)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features.edge_density = np.sum(edges > 0) / edges.size
            
            # Brightness profile (distribution of brightness values)
            features.brightness_profile = np.histogram(gray, bins=16, range=(0, 256))[0] / gray.size
            
            # Color coherence (how consistent colors are)
            features.color_coherence = self._calculate_color_coherence(player_crop)
            
        except Exception as e:
            logger.warning(f"Advanced appearance feature extraction failed: {e}")
    
    def _calculate_texture_entropy(self, gray_image: np.ndarray) -> float:
        """Calculate texture entropy using gray level co-occurrence matrix."""
        try:
            # Create histogram
            hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
            
            # Normalize to probabilities
            hist = hist / np.sum(hist)
            
            # Calculate entropy
            entropy_val = entropy(hist + 1e-7)  # Add small value to avoid log(0)
            
            return float(entropy_val)
            
        except Exception:
            return 0.0
    
    def _calculate_color_coherence(self, crop: np.ndarray) -> float:
        """Calculate color coherence measure."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            
            # Calculate standard deviation of each channel
            std_l = np.std(lab[:, :, 0])
            std_a = np.std(lab[:, :, 1])
            std_b = np.std(lab[:, :, 2])
            
            # Coherence is inverse of color variation
            total_std = std_l + std_a + std_b
            coherence = 1.0 / (1.0 + total_std / 100.0)  # Normalize
            
            return float(coherence)
            
        except Exception:
            return 0.0
    
    def _calculate_quality_metrics(self, player_crop: np.ndarray, detection: PlayerDetection,
                                 features: PlayerFeatures):
        """Calculate feature quality and extraction confidence."""
        try:
            # Base quality on detection confidence
            base_quality = detection.confidence
            
            # Adjust based on crop size (larger crops = better quality)
            size_factor = min(1.0, (player_crop.shape[0] * player_crop.shape[1]) / (64 * 128))
            
            # Adjust based on blur score if available
            blur_factor = 1.0
            if detection.blur_score is not None:
                blur_factor = min(1.0, detection.blur_score / 100.0)
            
            # Combine factors
            features.feature_quality_score = base_quality * size_factor * blur_factor
            
            # Extraction confidence based on successful feature extraction
            confidence_factors = [
                1.0 if features.primary_jersey_color else 0.0,
                1.0 if features.field_position_x is not None else 0.0,
                1.0 if features.texture_entropy is not None else 0.0,
                1.0 if len(features.dominant_colors_robust or []) > 0 else 0.0
            ]
            
            features.extraction_confidence = np.mean(confidence_factors)
            
        except Exception as e:
            logger.warning(f"Quality metric calculation failed: {e}")
    
    def process_video_features(self, detections_file: str, video_file: str, 
                             video_source: str, output_dir: str) -> List[PlayerFeatures]:
        """Process all detections in a video and extract features."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load detections
        with open(detections_file, 'rb') as f:
            detections = pickle.load(f)
        
        logger.info(f"Processing {len(detections)} detections from {video_source}")
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        
        all_features = []
        
        # Group detections by frame for efficient processing
        frame_groups = {}
        for det in detections:
            frame_num = det.frame_number
            if frame_num not in frame_groups:
                frame_groups[frame_num] = []
            frame_groups[frame_num].append(det)
        
        # Process each frame
        for frame_num in tqdm(sorted(frame_groups.keys()), desc=f"Extracting features - {video_source}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Process all detections in this frame
            for detection in frame_groups[frame_num]:
                features = self.extract_enhanced_features(detection, frame, video_source)
                all_features.append(features)
        
        cap.release()
        
        # Save features
        self._save_features(all_features, output_dir, video_source)
        
        # Generate feature analysis
        self._analyze_extracted_features(all_features, output_dir, video_source)
        
        logger.info(f"Feature extraction complete. {len(all_features)} features extracted.")
        
        return all_features
    
    def _save_features(self, features: List[PlayerFeatures], output_dir: Path, video_source: str):
        """Save extracted features."""
        
        # Save as pickle for efficient loading
        with open(output_dir / f'{video_source}_features.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        # Save simplified JSON for human readability (only basic info)
        features_simple = []
        for feat in features:
            simple_feat = {
                'detection_id': str(feat.detection_id),
                'frame_number': int(feat.frame_number),
                'video_source': str(feat.video_source),
                'primary_jersey_color': [int(x) for x in feat.primary_jersey_color] if feat.primary_jersey_color else None,
                'field_position_x': float(feat.field_position_x) if feat.field_position_x is not None else None,
                'field_position_y': float(feat.field_position_y) if feat.field_position_y is not None else None,
                'player_height_ratio': float(feat.player_height_ratio) if feat.player_height_ratio is not None else None,
                'aspect_ratio': float(feat.aspect_ratio) if feat.aspect_ratio is not None else None,
                'field_zone': str(feat.field_zone) if feat.field_zone else None,
                'relative_position': str(feat.relative_position) if feat.relative_position else None,
                'feature_quality_score': float(feat.feature_quality_score) if feat.feature_quality_score is not None else None
            }
            features_simple.append(simple_feat)
        
        with open(output_dir / f'{video_source}_features.json', 'w') as f:
            json.dump(features_simple, f, indent=2)
        
        # Save feature summary
        summary = self._create_feature_summary(features, video_source)
        with open(output_dir / f'{video_source}_feature_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _create_feature_summary(self, features: List[PlayerFeatures], video_source: str) -> Dict:
        """Create summary of extracted features."""
        
        summary = {
            'video_source': video_source,
            'total_features': len(features),
            'frames_processed': len(set(f.frame_number for f in features)),
            'feature_statistics': {},
            'jersey_color_analysis': {},
            'field_coverage_analysis': {}
        }
        
        if not features:
            return summary
        
        # Feature statistics
        valid_jersey_colors = [f for f in features if f.primary_jersey_color]
        valid_positions = [f for f in features if f.field_position_x is not None]
        valid_sizes = [f for f in features if f.player_height_ratio is not None]
        
        summary['feature_statistics'] = {
            'jersey_color_extraction_rate': float(len(valid_jersey_colors) / len(features)),
            'position_extraction_rate': float(len(valid_positions) / len(features)),
            'size_extraction_rate': float(len(valid_sizes) / len(features)),
            'avg_feature_quality': float(np.mean([f.feature_quality_score for f in features if f.feature_quality_score])) if valid_jersey_colors else 0.0,
            'avg_extraction_confidence': float(np.mean([f.extraction_confidence for f in features if f.extraction_confidence])) if valid_jersey_colors else 0.0
        }
        
        # Jersey color analysis
        if valid_jersey_colors:
            jersey_colors = [f.primary_jersey_color for f in valid_jersey_colors]
            # Convert to tuples for counting
            jersey_color_tuples = [tuple(color) for color in jersey_colors]
            unique_colors = len(set(jersey_color_tuples))
            summary['jersey_color_analysis'] = {
                'unique_jersey_colors_detected': int(unique_colors),
                'most_common_jersey_colors': self._find_most_common_colors(jersey_colors)
            }
        
        # Field coverage analysis
        if valid_positions:
            positions = [(f.field_position_x, f.field_position_y) for f in valid_positions]
            summary['field_coverage_analysis'] = {
                'position_spread_x': float(np.std([p[0] for p in positions])),
                'position_spread_y': float(np.std([p[1] for p in positions])),
                'coverage_center': [float(np.mean([p[0] for p in positions])), float(np.mean([p[1] for p in positions]))]
            }
        
        return summary
    
    def _find_most_common_colors(self, colors: List[Tuple[int, int, int]], top_k: int = 5) -> List[Dict]:
        """Find most common jersey colors."""
        from collections import Counter
        
        # Convert numpy types to native Python types
        python_colors = [tuple(int(c) for c in color) for color in colors]
        color_counts = Counter(python_colors)
        most_common = color_counts.most_common(top_k)
        
        return [{'color': [int(c) for c in color], 'count': int(count), 'rgb': [int(c) for c in color]} for color, count in most_common]
    
    def _analyze_extracted_features(self, features: List[PlayerFeatures], output_dir: Path, video_source: str):
        """Create comprehensive analysis of extracted features."""
        
        if not features:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Feature Analysis: {video_source.title()} Video', fontsize=16, fontweight='bold')
        
        # 1. Jersey color distribution
        jersey_colors = [f.primary_jersey_color for f in features if f.primary_jersey_color]
        if jersey_colors:
            unique_colors, counts = np.unique(jersey_colors, axis=0, return_counts=True)
            
            # Show top 10 most common colors
            top_indices = np.argsort(counts)[-10:]
            axes[0, 0].bar(range(len(top_indices)), counts[top_indices])
            axes[0, 0].set_title('Top Jersey Colors Distribution')
            axes[0, 0].set_xlabel('Color Index')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. Player size distribution
        sizes = [f.player_height_ratio for f in features if f.player_height_ratio]
        if sizes:
            axes[0, 1].hist(sizes, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Player Size Distribution')
            axes[0, 1].set_xlabel('Height Ratio')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Field position heatmap
        positions = [(f.field_position_x, f.field_position_y) for f in features 
                    if f.field_position_x is not None and f.field_position_y is not None]
        if positions:
            x_pos, y_pos = zip(*positions)
            axes[0, 2].scatter(x_pos, y_pos, alpha=0.6, s=10)
            axes[0, 2].set_title('Field Position Distribution')
            axes[0, 2].set_xlabel('Field X Position')
            axes[0, 2].set_ylabel('Field Y Position')
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].invert_yaxis()
        
        # 4. Feature quality scores
        quality_scores = [f.feature_quality_score for f in features if f.feature_quality_score]
        if quality_scores:
            axes[1, 0].hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Feature Quality Scores')
            axes[1, 0].set_xlabel('Quality Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. Aspect ratio distribution
        aspect_ratios = [f.aspect_ratio for f in features if f.aspect_ratio]
        if aspect_ratios:
            axes[1, 1].hist(aspect_ratios, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Player Aspect Ratio Distribution')
            axes[1, 1].set_xlabel('Height/Width Ratio')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Distance estimation
        distances = [f.estimated_distance for f in features if f.estimated_distance]
        if distances:
            axes[1, 2].hist(distances, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('Estimated Distance Distribution')
            axes[1, 2].set_xlabel('Estimated Distance')
            axes[1, 2].set_ylabel('Frequency')
        
        # 7. Texture entropy
        entropy_vals = [f.texture_entropy for f in features if f.texture_entropy]
        if entropy_vals:
            axes[2, 0].hist(entropy_vals, bins=20, alpha=0.7, edgecolor='black')
            axes[2, 0].set_title('Texture Entropy Distribution')
            axes[2, 0].set_xlabel('Entropy')
            axes[2, 0].set_ylabel('Frequency')
        
        # 8. Color coherence
        coherence_vals = [f.color_coherence for f in features if f.color_coherence]
        if coherence_vals:
            axes[2, 1].hist(coherence_vals, bins=20, alpha=0.7, edgecolor='black')
            axes[2, 1].set_title('Color Coherence Distribution')
            axes[2, 1].set_xlabel('Coherence')
            axes[2, 1].set_ylabel('Frequency')
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        summary_text = f"""
        Feature Extraction Summary:
        
        Total Features: {len(features):,}
        
        Jersey Color Success: {len(jersey_colors)}/{len(features)} ({len(jersey_colors)/len(features)*100:.1f}%)
        Position Success: {len(positions)}/{len(features)} ({len(positions)/len(features)*100:.1f}%)
        Size Success: {len(sizes)}/{len(features)} ({len(sizes)/len(features)*100:.1f}%)
        
        Avg Quality Score: {np.mean(quality_scores):.3f}
        Avg Aspect Ratio: {np.mean(aspect_ratios):.2f}
        
        Unique Jersey Colors: {len(set(jersey_colors)) if jersey_colors else 0}
        Field Coverage: {len(set([(int(x*10), int(y*10)) for x, y in positions]))} zones
        """
        
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{video_source}_feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature analysis plots saved for {video_source}")

def main():
    """Main function to run advanced feature extraction."""
    
    print("🎯 Phase 4: Advanced Player Feature Extraction")
    print("=" * 60)
    
    # Initialize feature extractor
    extractor = AdvancedFeatureExtractor()
    
    # Process both videos
    broadcast_features = extractor.process_video_features(
        detections_file=str(config.BROADCAST_RESULTS / 'detections.pkl'),
        video_file=str(config.BROADCAST_VIDEO),
        video_source='broadcast',
        output_dir='Results/Features'
    )
    
    tacticam_features = extractor.process_video_features(
        detections_file=str(config.TACTICAM_RESULTS / 'detections.pkl'),
        video_file=str(config.TACTICAM_VIDEO),
        video_source='tacticam',
        output_dir='Results/Features'
    )
    
    print(f"\n✅ Feature extraction completed!")
    print(f"📊 Broadcast features: {len(broadcast_features)}")
    print(f"📊 Tacticam features: {len(tacticam_features)}")
    print(f"📁 Results saved in: Results/Features/")
    print("\n🚀 Ready for Phase 5: Player Matching!")

if __name__ == "__main__":
    main()
