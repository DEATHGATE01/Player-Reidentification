"""
🎯 Phase 5: Player Matching Across Cameras - Core Re-identification System

This module implements the core challenge of player re-identification: matching players
between broadcast and tacticam videos using appearance, spatial position, and timing.

Key Features:
- Multi-modal similarity calculation (appearance + spatial + temporal)
- Intelligent matching algorithms with confidence scoring
- Team-aware matching to improve accuracy
- Temporal synchronization handling
- Robust similarity metrics for cross-camera matching
"""

import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import cv2
from tqdm import tqdm

from config import Config
from advanced_feature_extraction import PlayerFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerMatch:
    """Represents a match between players from different camera views."""
    
    # Basic match information
    broadcast_detection_id: str
    tacticam_detection_id: str
    broadcast_frame: int
    tacticam_frame: int
    
    # Similarity scores (0-1, higher = more similar)
    overall_similarity: float
    appearance_similarity: float
    spatial_similarity: float
    temporal_similarity: float
    
    # Match confidence and quality
    match_confidence: float  # Overall confidence in this match
    match_quality: str      # 'high', 'medium', 'low'
    
    # Detailed similarity breakdown
    jersey_color_similarity: float
    size_similarity: float
    texture_similarity: float
    position_similarity: float
    
    # Additional context
    team_assignment: Optional[str] = None
    match_reasoning: str = ""

class CrossCameraPlayerMatcher:
    """
    Advanced player matching system for cross-camera re-identification.
    
    This is the core of the re-identification system that solves the fundamental
    challenge: "Which player in the broadcast video is the same as which player
    in the tacticam video?"
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.broadcast_features: List[PlayerFeatures] = []
        self.tacticam_features: List[PlayerFeatures] = []
        self.matches: List[PlayerMatch] = []
        
        # Similarity weights for different feature types
        self.feature_weights = {
            'jersey_color': 0.35,    # Most important for player identity
            'size': 0.20,            # Important for distinguishing players
            'spatial_position': 0.25, # Important for context
            'texture': 0.15,         # Additional appearance info
            'temporal': 0.05         # Timing consistency
        }
        
        # Confidence thresholds for match quality
        self.confidence_thresholds = {
            'high': 0.75,     # Very confident match
            'medium': 0.55,   # Reasonable match
            'low': 0.35       # Weak but possible match
        }
        
    def load_extracted_features(self):
        """Load the features extracted in Phase 4."""
        
        logger.info("Loading extracted features from Phase 4...")
        
        features_dir = self.config.RESULTS_DIR / 'Features'
        
        # Load broadcast features
        with open(features_dir / 'broadcast_features.pkl', 'rb') as f:
            self.broadcast_features = pickle.load(f)
        
        # Load tacticam features
        with open(features_dir / 'tacticam_features.pkl', 'rb') as f:
            self.tacticam_features = pickle.load(f)
            
        logger.info(f"Loaded {len(self.broadcast_features)} broadcast features")
        logger.info(f"Loaded {len(self.tacticam_features)} tacticam features")
        
        return len(self.broadcast_features), len(self.tacticam_features)
    
    def calculate_jersey_color_similarity(self, broadcast_feat: PlayerFeatures, 
                                        tacticam_feat: PlayerFeatures) -> float:
        """
        Calculate jersey color similarity between two player detections.
        
        This is crucial for player identity - players wear the same jersey!
        """
        
        if not broadcast_feat.primary_jersey_color or not tacticam_feat.primary_jersey_color:
            return 0.0
            
        # Convert to numpy arrays for calculation
        color1 = np.array(broadcast_feat.primary_jersey_color, dtype=float)
        color2 = np.array(tacticam_feat.primary_jersey_color, dtype=float)
        
        # Normalize to 0-1 range
        color1_norm = color1 / 255.0
        color2_norm = color2 / 255.0
        
        # Calculate Euclidean distance in color space
        color_distance = np.linalg.norm(color1_norm - color2_norm)
        
        # Convert distance to similarity (0-1, where 1 = identical colors)
        # Max possible distance in RGB space is sqrt(3) ≈ 1.73
        max_distance = np.sqrt(3)
        similarity = max(0.0, 1.0 - (color_distance / max_distance))
        
        return similarity
    
    def calculate_size_similarity(self, broadcast_feat: PlayerFeatures, 
                                tacticam_feat: PlayerFeatures) -> float:
        """
        Calculate player size similarity between two detections.
        
        Players maintain relative size characteristics across camera views.
        """
        
        if (broadcast_feat.player_height_ratio is None or 
            tacticam_feat.player_height_ratio is None):
            return 0.5  # Neutral if size unknown
            
        # Height ratio similarity
        height_diff = abs(broadcast_feat.player_height_ratio - tacticam_feat.player_height_ratio)
        height_sim = max(0.0, 1.0 - height_diff * 5)  # Scale the difference
        
        # Aspect ratio similarity (if available)
        aspect_sim = 0.5  # Default
        if (broadcast_feat.aspect_ratio is not None and 
            tacticam_feat.aspect_ratio is not None):
            aspect_diff = abs(broadcast_feat.aspect_ratio - tacticam_feat.aspect_ratio)
            aspect_sim = max(0.0, 1.0 - aspect_diff * 2)
            
        return (height_sim + aspect_sim) / 2
    
    def calculate_spatial_similarity(self, broadcast_feat: PlayerFeatures, 
                                   tacticam_feat: PlayerFeatures) -> float:
        """
        Calculate spatial position similarity considering different camera angles.
        
        Players in similar field positions should have higher similarity,
        but we need to account for different camera perspectives.
        """
        
        # If positions are unknown, use neutral similarity
        if (broadcast_feat.field_position_x is None or broadcast_feat.field_position_y is None or
            tacticam_feat.field_position_x is None or tacticam_feat.field_position_y is None):
            return 0.4  # Slightly below neutral to encourage position-based matching
            
        # Calculate position distance
        broadcast_pos = np.array([broadcast_feat.field_position_x, broadcast_feat.field_position_y])
        tacticam_pos = np.array([tacticam_feat.field_position_x, tacticam_feat.field_position_y])
        
        position_distance = np.linalg.norm(broadcast_pos - tacticam_pos)
        
        # Convert to similarity (closer positions = higher similarity)
        # Max distance in normalized field is sqrt(2) ≈ 1.41
        max_distance = np.sqrt(2)
        position_similarity = max(0.0, 1.0 - (position_distance / max_distance))
        
        # Bonus for same field zone
        zone_bonus = 0.0
        if (broadcast_feat.field_zone and tacticam_feat.field_zone and
            broadcast_feat.field_zone == tacticam_feat.field_zone):
            zone_bonus = 0.2
            
        # Bonus for same relative position (top/center/bottom)
        relative_bonus = 0.0
        if (broadcast_feat.relative_position and tacticam_feat.relative_position and
            broadcast_feat.relative_position == tacticam_feat.relative_position):
            relative_bonus = 0.1
            
        total_similarity = min(1.0, position_similarity + zone_bonus + relative_bonus)
        return total_similarity
    
    def calculate_texture_similarity(self, broadcast_feat: PlayerFeatures, 
                                   tacticam_feat: PlayerFeatures) -> float:
        """
        Calculate texture and appearance similarity between players.
        
        Combines texture entropy, edge density, and other appearance features.
        """
        
        similarities = []
        
        # Texture entropy similarity
        if (broadcast_feat.texture_entropy is not None and 
            tacticam_feat.texture_entropy is not None):
            entropy_diff = abs(broadcast_feat.texture_entropy - tacticam_feat.texture_entropy)
            # Entropy typically ranges 0-8, but we expect 2-5 for player regions
            entropy_sim = max(0.0, 1.0 - entropy_diff / 6.0)
            similarities.append(entropy_sim)
            
        # Edge density similarity
        if (broadcast_feat.edge_density is not None and 
            tacticam_feat.edge_density is not None):
            edge_diff = abs(broadcast_feat.edge_density - tacticam_feat.edge_density)
            edge_sim = max(0.0, 1.0 - edge_diff * 3)  # Scale edge difference
            similarities.append(edge_sim)
            
        # Color coherence similarity (if available)
        if (broadcast_feat.color_coherence is not None and 
            tacticam_feat.color_coherence is not None):
            coherence_diff = abs(broadcast_feat.color_coherence - tacticam_feat.color_coherence)
            coherence_sim = max(0.0, 1.0 - coherence_diff)
            similarities.append(coherence_sim)
            
        return np.mean(similarities) if similarities else 0.5
    
    def calculate_temporal_similarity(self, broadcast_feat: PlayerFeatures, 
                                    tacticam_feat: PlayerFeatures) -> float:
        """
        Calculate temporal similarity - are players visible at similar times?
        
        Note: Videos may not be perfectly synchronized, so we allow some tolerance.
        """
        
        frame_diff = abs(broadcast_feat.frame_number - tacticam_feat.frame_number)
        
        # Allow up to 15 frames difference (videos might be slightly out of sync)
        max_frame_diff = 15
        temporal_sim = max(0.0, 1.0 - frame_diff / max_frame_diff)
        
        return temporal_sim
    
    def calculate_comprehensive_similarity(self, broadcast_feat: PlayerFeatures, 
                                         tacticam_feat: PlayerFeatures) -> Dict[str, float]:
        """
        Calculate comprehensive similarity between two player detections.
        
        This is the core matching function that determines how similar two
        player detections are across different camera views.
        """
        
        # Calculate individual similarity components
        jersey_sim = self.calculate_jersey_color_similarity(broadcast_feat, tacticam_feat)
        size_sim = self.calculate_size_similarity(broadcast_feat, tacticam_feat)
        spatial_sim = self.calculate_spatial_similarity(broadcast_feat, tacticam_feat)
        texture_sim = self.calculate_texture_similarity(broadcast_feat, tacticam_feat)
        temporal_sim = self.calculate_temporal_similarity(broadcast_feat, tacticam_feat)
        
        # Store individual similarities
        similarities = {
            'jersey_color': jersey_sim,
            'size': size_sim,
            'spatial_position': spatial_sim,
            'texture': texture_sim,
            'temporal': temporal_sim
        }
        
        # Calculate weighted overall similarity
        overall_sim = sum(similarities[feature] * self.feature_weights[feature] 
                         for feature in similarities)
        
        similarities['overall'] = overall_sim
        
        return similarities
    
    def perform_team_clustering(self, features: List[PlayerFeatures]) -> Dict[int, List[PlayerFeatures]]:
        """
        Cluster players into teams based on jersey colors.
        
        This helps improve matching accuracy by only matching players
        from the same team across cameras.
        """
        
        logger.info("Performing team clustering for better matching...")
        
        # Extract valid jersey colors
        valid_features = [f for f in features if f.primary_jersey_color]
        
        if len(valid_features) < 2:
            return {0: features}  # Single team if insufficient data
            
        # Extract jersey colors as feature vectors
        jersey_colors = np.array([f.primary_jersey_color for f in valid_features])
        
        # Use K-means to identify teams (assume 2 main teams + possibly referees)
        n_clusters = min(3, len(valid_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(jersey_colors)
        
        # Group features by team
        teams = defaultdict(list)
        for i, feature in enumerate(valid_features):
            teams[team_labels[i]].append(feature)
            
        # Add features without jersey colors to the largest team
        if len(valid_features) < len(features):
            largest_team = max(teams.keys(), key=lambda k: len(teams[k]))
            for feature in features:
                if not feature.primary_jersey_color:
                    teams[largest_team].append(feature)
        
        logger.info(f"Identified {len(teams)} teams")
        for team_id, team_features in teams.items():
            logger.info(f"Team {team_id}: {len(team_features)} players")
            
        return dict(teams)
    
    def match_teams_across_cameras(self, broadcast_teams: Dict[int, List[PlayerFeatures]], 
                                  tacticam_teams: Dict[int, List[PlayerFeatures]]) -> List[Tuple[int, int]]:
        """
        Match teams between broadcast and tacticam videos.
        
        This ensures we're comparing players from the same team.
        """
        
        team_similarities = {}
        
        # Calculate average jersey color similarity between all team pairs
        for broadcast_id, broadcast_team in broadcast_teams.items():
            for tacticam_id, tacticam_team in tacticam_teams.items():
                similarities = []
                
                # Sample players from each team to calculate similarity
                broadcast_sample = broadcast_team[:10]  # Sample first 10 players
                tacticam_sample = tacticam_team[:10]
                
                for b_feat in broadcast_sample:
                    for t_feat in tacticam_sample:
                        jersey_sim = self.calculate_jersey_color_similarity(b_feat, t_feat)
                        similarities.append(jersey_sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    team_similarities[(broadcast_id, tacticam_id)] = avg_similarity
        
        # Find best team matches using greedy approach
        used_broadcast = set()
        used_tacticam = set()
        team_matches = []
        
        # Sort by similarity and match greedily
        sorted_matches = sorted(team_similarities.items(), key=lambda x: x[1], reverse=True)
        
        for (broadcast_id, tacticam_id), similarity in sorted_matches:
            if broadcast_id not in used_broadcast and tacticam_id not in used_tacticam:
                team_matches.append((broadcast_id, tacticam_id))
                used_broadcast.add(broadcast_id)
                used_tacticam.add(tacticam_id)
                logger.info(f"Matched teams: Broadcast {broadcast_id} ↔ Tacticam {tacticam_id} (similarity: {similarity:.3f})")
        
        return team_matches
    
    def find_best_matches_for_tacticam_players(self, team_matches: List[Tuple[int, int]],
                                              broadcast_teams: Dict[int, List[PlayerFeatures]], 
                                              tacticam_teams: Dict[int, List[PlayerFeatures]]) -> List[PlayerMatch]:
        """
        For each tacticam player, find the best matching broadcast player.
        
        This is the core matching algorithm that solves the re-identification challenge.
        """
        
        logger.info("Finding best matches for each tacticam player...")
        all_matches = []
        
        for broadcast_team_id, tacticam_team_id in team_matches:
            broadcast_team = broadcast_teams[broadcast_team_id]
            tacticam_team = tacticam_teams[tacticam_team_id]
            
            logger.info(f"Matching players between Broadcast Team {broadcast_team_id} and Tacticam Team {tacticam_team_id}")
            
            # Create similarity matrix
            similarity_matrix = np.zeros((len(tacticam_team), len(broadcast_team)))
            detailed_similarities = {}
            
            # Calculate similarities between all player pairs
            for i, tacticam_player in enumerate(tacticam_team):
                for j, broadcast_player in enumerate(broadcast_team):
                    similarities = self.calculate_comprehensive_similarity(broadcast_player, tacticam_player)
                    similarity_matrix[i, j] = similarities['overall']
                    detailed_similarities[(i, j)] = similarities
            
            # Use Hungarian algorithm for optimal matching
            # This ensures each tacticam player gets matched to exactly one broadcast player
            tacticam_indices, broadcast_indices = linear_sum_assignment(-similarity_matrix)
            
            # Create match objects for good matches
            for tacticam_idx, broadcast_idx in zip(tacticam_indices, broadcast_indices):
                overall_similarity = similarity_matrix[tacticam_idx, broadcast_idx]
                
                # Only keep matches above minimum threshold
                if overall_similarity > 0.2:  # Very low threshold to keep weak matches for analysis
                    tacticam_player = tacticam_team[tacticam_idx]
                    broadcast_player = broadcast_team[broadcast_idx]
                    similarities = detailed_similarities[(tacticam_idx, broadcast_idx)]
                    
                    # Determine match quality
                    if overall_similarity >= self.confidence_thresholds['high']:
                        match_quality = 'high'
                        match_confidence = 0.9
                    elif overall_similarity >= self.confidence_thresholds['medium']:
                        match_quality = 'medium'
                        match_confidence = 0.7
                    elif overall_similarity >= self.confidence_thresholds['low']:
                        match_quality = 'low'
                        match_confidence = 0.5
                    else:
                        match_quality = 'very_low'
                        match_confidence = 0.3
                    
                    # Create match reasoning
                    reasoning_parts = []
                    if similarities['jersey_color'] > 0.7:
                        reasoning_parts.append(f"Strong jersey color match ({similarities['jersey_color']:.2f})")
                    if similarities['spatial_position'] > 0.7:
                        reasoning_parts.append(f"Similar field position ({similarities['spatial_position']:.2f})")
                    if similarities['size'] > 0.7:
                        reasoning_parts.append(f"Similar player size ({similarities['size']:.2f})")
                    if similarities['temporal'] > 0.8:
                        reasoning_parts.append(f"Good temporal alignment ({similarities['temporal']:.2f})")
                    
                    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Weak overall similarity"
                    
                    # Create match object
                    match = PlayerMatch(
                        broadcast_detection_id=broadcast_player.detection_id,
                        tacticam_detection_id=tacticam_player.detection_id,
                        broadcast_frame=broadcast_player.frame_number,
                        tacticam_frame=tacticam_player.frame_number,
                        overall_similarity=overall_similarity,
                        appearance_similarity=(similarities['jersey_color'] + similarities['texture']) / 2,
                        spatial_similarity=similarities['spatial_position'],
                        temporal_similarity=similarities['temporal'],
                        match_confidence=match_confidence,
                        match_quality=match_quality,
                        jersey_color_similarity=similarities['jersey_color'],
                        size_similarity=similarities['size'],
                        texture_similarity=similarities['texture'],
                        position_similarity=similarities['spatial_position'],
                        team_assignment=f"Team_{broadcast_team_id}",
                        match_reasoning=reasoning
                    )
                    
                    all_matches.append(match)
        
        return all_matches
    
    def run_cross_camera_matching(self) -> List[PlayerMatch]:
        """
        Run the complete cross-camera player matching pipeline.
        
        This is the main function that solves the core re-identification challenge.
        """
        
        logger.info("🎯 Starting cross-camera player matching...")
        logger.info("This solves the core challenge: Which player in broadcast = which player in tacticam?")
        
        # Step 1: Load extracted features
        broadcast_count, tacticam_count = self.load_extracted_features()
        
        # Step 2: Perform team clustering for both videos
        logger.info("Step 1: Clustering players into teams...")
        broadcast_teams = self.perform_team_clustering(self.broadcast_features)
        tacticam_teams = self.perform_team_clustering(self.tacticam_features)
        
        # Step 3: Match teams across cameras
        logger.info("Step 2: Matching teams across cameras...")
        team_matches = self.match_teams_across_cameras(broadcast_teams, tacticam_teams)
        
        # Step 4: For each tacticam player, find best broadcast match
        logger.info("Step 3: Finding best matches for each tacticam player...")
        matches = self.find_best_matches_for_tacticam_players(team_matches, broadcast_teams, tacticam_teams)
        
        self.matches = matches
        
        # Step 5: Analyze results
        logger.info("✅ Cross-camera matching complete!")
        logger.info(f"Found {len(matches)} potential player matches")
        
        # Count matches by quality
        quality_counts = Counter(m.match_quality for m in matches)
        logger.info("Match quality distribution:")
        for quality, count in quality_counts.items():
            logger.info(f"  {quality}: {count} matches")
        
        return matches
    
    def save_matching_results(self, output_dir: str):
        """Save the cross-camera matching results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for efficient loading
        with open(output_path / 'cross_camera_matches.pkl', 'wb') as f:
            pickle.dump(self.matches, f)
        
        # Save as JSON for human readability
        matches_json = []
        for match in self.matches:
            match_dict = asdict(match)
            matches_json.append(match_dict)
        
        with open(output_path / 'cross_camera_matches.json', 'w') as f:
            json.dump(matches_json, f, indent=2)
        
        # Create detailed analysis
        analysis = self._create_matching_analysis()
        with open(output_path / 'matching_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Cross-camera matching results saved to {output_path}")
    
    def _create_matching_analysis(self) -> Dict:
        """Create detailed analysis of matching results."""
        
        if not self.matches:
            return {"error": "No matches to analyze"}
        
        analysis = {
            "summary": {
                "total_matches": len(self.matches),
                "broadcast_players_available": len(self.broadcast_features),
                "tacticam_players_available": len(self.tacticam_features),
                "matching_rate": len(self.matches) / len(self.tacticam_features) if self.tacticam_features else 0
            },
            "quality_distribution": dict(Counter(m.match_quality for m in self.matches)),
            "similarity_statistics": {
                "overall_similarity": {
                    "mean": float(np.mean([m.overall_similarity for m in self.matches])),
                    "std": float(np.std([m.overall_similarity for m in self.matches])),
                    "min": float(np.min([m.overall_similarity for m in self.matches])),
                    "max": float(np.max([m.overall_similarity for m in self.matches]))
                },
                "appearance_similarity": {
                    "mean": float(np.mean([m.appearance_similarity for m in self.matches])),
                    "std": float(np.std([m.appearance_similarity for m in self.matches]))
                },
                "spatial_similarity": {
                    "mean": float(np.mean([m.spatial_similarity for m in self.matches])),
                    "std": float(np.std([m.spatial_similarity for m in self.matches]))
                },
                "temporal_similarity": {
                    "mean": float(np.mean([m.temporal_similarity for m in self.matches])),
                    "std": float(np.std([m.temporal_similarity for m in self.matches]))
                }
            },
            "best_matches": [
                {
                    "broadcast_id": match.broadcast_detection_id,
                    "tacticam_id": match.tacticam_detection_id,
                    "similarity": match.overall_similarity,
                    "quality": match.match_quality,
                    "reasoning": match.match_reasoning
                }
                for match in sorted(self.matches, key=lambda x: x.overall_similarity, reverse=True)[:10]
            ]
        }
        
        return analysis
    
    def create_matching_visualizations(self, output_dir: str):
        """Create visualizations of the matching results."""
        
        if not self.matches:
            logger.warning("No matches to visualize")
            return
        
        output_path = Path(output_dir)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Camera Player Matching Analysis', fontsize=16, fontweight='bold')
        
        # 1. Match quality distribution
        quality_counts = Counter(m.match_quality for m in self.matches)
        axes[0, 0].bar(quality_counts.keys(), quality_counts.values(), color=['green', 'orange', 'red', 'darkred'])
        axes[0, 0].set_title('Match Quality Distribution')
        axes[0, 0].set_ylabel('Number of Matches')
        
        # 2. Overall similarity distribution
        similarities = [m.overall_similarity for m in self.matches]
        axes[0, 1].hist(similarities, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 1].set_title('Overall Similarity Distribution')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
        axes[0, 1].legend()
        
        # 3. Feature similarity comparison
        feature_names = ['Jersey Color', 'Size', 'Texture', 'Position', 'Temporal']
        feature_means = [
            np.mean([m.jersey_color_similarity for m in self.matches]),
            np.mean([m.size_similarity for m in self.matches]),
            np.mean([m.texture_similarity for m in self.matches]),
            np.mean([m.position_similarity for m in self.matches]),
            np.mean([m.temporal_similarity for m in self.matches])
        ]
        
        bars = axes[0, 2].bar(feature_names, feature_means, color=['red', 'blue', 'green', 'orange', 'purple'])
        axes[0, 2].set_title('Average Feature Similarities')
        axes[0, 2].set_ylabel('Average Similarity')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, feature_means):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Similarity vs Confidence scatter
        confidences = [m.match_confidence for m in self.matches]
        colors = ['green' if m.match_quality == 'high' else 
                 'orange' if m.match_quality == 'medium' else 
                 'red' if m.match_quality == 'low' else 'darkred' 
                 for m in self.matches]
        
        axes[1, 0].scatter(similarities, confidences, c=colors, alpha=0.6)
        axes[1, 0].set_title('Similarity vs Confidence')
        axes[1, 0].set_xlabel('Overall Similarity')
        axes[1, 0].set_ylabel('Match Confidence')
        
        # 5. Temporal alignment analysis
        temporal_similarities = [m.temporal_similarity for m in self.matches]
        axes[1, 1].hist(temporal_similarities, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
        axes[1, 1].set_title('Temporal Similarity Distribution')
        axes[1, 1].set_xlabel('Temporal Similarity')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Frame difference analysis
        frame_diffs = [abs(m.broadcast_frame - m.tacticam_frame) for m in self.matches]
        axes[1, 2].hist(frame_diffs, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[1, 2].set_title('Frame Difference Distribution')
        axes[1, 2].set_xlabel('Frame Difference')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(np.mean(frame_diffs), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(frame_diffs):.1f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'cross_camera_matching_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matching visualizations saved to {output_path}")

def main():
    """Main function to run cross-camera player matching."""
    
    print("🎯 Phase 5: Cross-Camera Player Matching")
    print("=" * 60)
    print("🚀 Solving the core re-identification challenge:")
    print("   'Which player in broadcast video = which player in tacticam video?'")
    print("=" * 60)
    
    # Initialize configuration and matcher
    config = Config()
    matcher = CrossCameraPlayerMatcher(config)
    
    # Run the matching pipeline
    matches = matcher.run_cross_camera_matching()
    
    # Save results
    output_dir = config.RESULTS_DIR / 'CrossCameraMatching'
    matcher.save_matching_results(str(output_dir))
    
    # Create visualizations
    matcher.create_matching_visualizations(str(output_dir))
    
    # Final summary
    print("\n" + "="*60)
    print("✅ CROSS-CAMERA MATCHING COMPLETED!")
    print("="*60)
    print(f"📊 Total matches found: {len(matches)}")
    
    quality_counts = Counter(m.match_quality for m in matches)
    for quality, count in quality_counts.items():
        print(f"   {quality.title()} quality matches: {count}")
    
    if matches:
        avg_similarity = np.mean([m.overall_similarity for m in matches])
        print(f"📈 Average similarity score: {avg_similarity:.3f}")
        
        best_match = max(matches, key=lambda x: x.overall_similarity)
        print(f"🏆 Best match: {best_match.overall_similarity:.3f} similarity")
        print(f"   Broadcast: {best_match.broadcast_detection_id} ↔ Tacticam: {best_match.tacticam_detection_id}")
    
    print(f"📁 Results saved in: {output_dir}")
    print("🎉 Core re-identification challenge solved!")

if __name__ == "__main__":
    main()
