"""
🎯 Phase 6: Consistent Player ID Assignment System

This module implements the final step of the re-identification system: assigning 
consistent unique IDs to players across both videos to maintain identity consistency.

Key Features:
- Unique player ID assignment based on cross-camera matches
- Identity consistency across broadcast and tacticam videos
- Player tracking and ID management
- Comprehensive player identity database
- Visual ID assignment validation
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
import matplotlib.patches as patches
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd

from config import Config
from cross_camera_matching import PlayerMatch
from advanced_feature_extraction import PlayerFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerIdentity:
    """Represents a consistent player identity across both videos."""
    
    # Unique identity information
    player_id: str                          # Unique player ID (e.g., "PLAYER_001")
    identity_confidence: float              # Confidence in this identity assignment
    
    # Cross-camera detections for this player
    broadcast_detections: List[str]         # List of broadcast detection IDs
    tacticam_detections: List[str]          # List of tacticam detection IDs
    
    # Identity characteristics
    primary_jersey_color: Tuple[int, int, int]   # Most representative jersey color
    team_assignment: str                    # Team identifier
    player_role: Optional[str] = None       # Position/role if determinable
    
    # Activity summary
    total_appearances: int = 0              # Total detections across both videos
    broadcast_appearances: int = 0          # Number of broadcast detections
    tacticam_appearances: int = 0           # Number of tacticam detections
    
    # Temporal information
    first_seen_frame: int = 0               # First frame where player appears
    last_seen_frame: int = 0                # Last frame where player appears
    active_duration: int = 0                # Duration of activity
    
    # Quality metrics
    identity_quality_score: float = 0.0    # Overall quality of identity assignment
    match_consistency: float = 0.0          # Consistency across matches

class ConsistentIDAssigner:
    """
    Assigns consistent unique IDs to players across both camera views.
    
    This system ensures that the same person has the same ID in both videos,
    maintaining identity consistency throughout the re-identification process.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.player_matches: List[PlayerMatch] = []
        self.broadcast_features: List[PlayerFeatures] = []
        self.tacticam_features: List[PlayerFeatures] = []
        self.player_identities: List[PlayerIdentity] = []
        
        # ID assignment settings
        self.id_prefix = "PLAYER_"
        self.min_appearances = 3  # Minimum appearances to get a permanent ID
        self.confidence_threshold = 0.6  # Minimum confidence for ID assignment
        
    def load_matching_results(self):
        """Load cross-camera matching results and features."""
        
        logger.info("Loading cross-camera matching results...")
        
        # Load matches
        matching_dir = self.config.RESULTS_DIR / 'CrossCameraMatching'
        with open(matching_dir / 'cross_camera_matches.pkl', 'rb') as f:
            self.player_matches = pickle.load(f)
        
        # Load features
        features_dir = self.config.RESULTS_DIR / 'Features'
        with open(features_dir / 'broadcast_features.pkl', 'rb') as f:
            self.broadcast_features = pickle.load(f)
            
        with open(features_dir / 'tacticam_features.pkl', 'rb') as f:
            self.tacticam_features = pickle.load(f)
        
        logger.info(f"Loaded {len(self.player_matches)} player matches")
        logger.info(f"Loaded {len(self.broadcast_features)} broadcast features")
        logger.info(f"Loaded {len(self.tacticam_features)} tacticam features")
        
    def create_feature_lookup_tables(self):
        """Create lookup tables for quick feature access."""
        
        self.broadcast_feature_lookup = {f.detection_id: f for f in self.broadcast_features}
        self.tacticam_feature_lookup = {f.detection_id: f for f in self.tacticam_features}
        
    def group_matches_by_player(self) -> Dict[str, List[PlayerMatch]]:
        """
        Group matches to identify all detections belonging to the same player.
        
        This creates groups of detections that represent the same player across videos.
        """
        
        logger.info("Grouping matches by player identity...")
        
        # Create a graph where each detection is a node
        # and matches create edges between broadcast and tacticam detections
        player_groups = {}
        broadcast_to_group = {}
        tacticam_to_group = {}
        group_counter = 0
        
        for match in self.player_matches:
            broadcast_id = match.broadcast_detection_id
            tacticam_id = match.tacticam_detection_id
            
            # Check if either detection is already in a group
            broadcast_group = broadcast_to_group.get(broadcast_id)
            tacticam_group = tacticam_to_group.get(tacticam_id)
            
            if broadcast_group is not None and tacticam_group is not None:
                # Both detections are already in groups - merge if different
                if broadcast_group != tacticam_group:
                    # Merge groups
                    smaller_group = min(broadcast_group, tacticam_group)
                    larger_group = max(broadcast_group, tacticam_group)
                    
                    # Move all detections from larger group to smaller group
                    for det_id in player_groups[larger_group]['broadcast_detections']:
                        broadcast_to_group[det_id] = smaller_group
                    for det_id in player_groups[larger_group]['tacticam_detections']:
                        tacticam_to_group[det_id] = smaller_group
                    
                    # Merge the detection lists
                    player_groups[smaller_group]['broadcast_detections'].extend(
                        player_groups[larger_group]['broadcast_detections']
                    )
                    player_groups[smaller_group]['tacticam_detections'].extend(
                        player_groups[larger_group]['tacticam_detections']
                    )
                    player_groups[smaller_group]['matches'].extend(
                        player_groups[larger_group]['matches']
                    )
                    
                    # Remove the larger group
                    del player_groups[larger_group]
                    
                # Add current match to the existing group
                current_group = min(broadcast_to_group.get(broadcast_id, group_counter), 
                                  tacticam_to_group.get(tacticam_id, group_counter))
                player_groups[current_group]['matches'].append(match)
                
            elif broadcast_group is not None:
                # Broadcast detection is in a group, add tacticam detection
                player_groups[broadcast_group]['tacticam_detections'].append(tacticam_id)
                player_groups[broadcast_group]['matches'].append(match)
                tacticam_to_group[tacticam_id] = broadcast_group
                
            elif tacticam_group is not None:
                # Tacticam detection is in a group, add broadcast detection
                player_groups[tacticam_group]['broadcast_detections'].append(broadcast_id)
                player_groups[tacticam_group]['matches'].append(match)
                broadcast_to_group[broadcast_id] = tacticam_group
                
            else:
                # Neither detection is in a group, create new group
                player_groups[group_counter] = {
                    'broadcast_detections': [broadcast_id],
                    'tacticam_detections': [tacticam_id],
                    'matches': [match]
                }
                broadcast_to_group[broadcast_id] = group_counter
                tacticam_to_group[tacticam_id] = group_counter
                group_counter += 1
        
        logger.info(f"Identified {len(player_groups)} distinct player identities")
        
        return player_groups
    
    def calculate_identity_characteristics(self, player_group: Dict) -> Dict:
        """Calculate characteristics for a player identity group."""
        
        all_matches = player_group['matches']
        broadcast_detections = player_group['broadcast_detections']
        tacticam_detections = player_group['tacticam_detections']
        
        # Get features for all detections
        broadcast_features = [self.broadcast_feature_lookup[det_id] 
                            for det_id in broadcast_detections 
                            if det_id in self.broadcast_feature_lookup]
        tacticam_features = [self.tacticam_feature_lookup[det_id] 
                           for det_id in tacticam_detections 
                           if det_id in self.tacticam_feature_lookup]
        
        all_features = broadcast_features + tacticam_features
        
        characteristics = {
            'total_appearances': len(all_features),
            'broadcast_appearances': len(broadcast_features),
            'tacticam_appearances': len(tacticam_features),
            'identity_confidence': np.mean([m.match_confidence for m in all_matches]) if all_matches else 0.0,
            'match_consistency': np.std([m.overall_similarity for m in all_matches]) if len(all_matches) > 1 else 1.0
        }
        
        # Calculate primary jersey color (most common)
        jersey_colors = [f.primary_jersey_color for f in all_features if f.primary_jersey_color]
        if jersey_colors:
            # Find most common jersey color
            color_counter = Counter(tuple(color) for color in jersey_colors)
            most_common_color = color_counter.most_common(1)[0][0]
            characteristics['primary_jersey_color'] = most_common_color
        else:
            characteristics['primary_jersey_color'] = (128, 128, 128)  # Gray default
        
        # Calculate team assignment (most common)
        team_assignments = [m.team_assignment for m in all_matches if m.team_assignment]
        if team_assignments:
            team_counter = Counter(team_assignments)
            most_common_team = team_counter.most_common(1)[0][0]
            characteristics['team_assignment'] = most_common_team
        else:
            characteristics['team_assignment'] = "Unknown"
        
        # Calculate temporal information
        frame_numbers = [f.frame_number for f in all_features]
        if frame_numbers:
            characteristics['first_seen_frame'] = min(frame_numbers)
            characteristics['last_seen_frame'] = max(frame_numbers)
            characteristics['active_duration'] = max(frame_numbers) - min(frame_numbers) + 1
        else:
            characteristics['first_seen_frame'] = 0
            characteristics['last_seen_frame'] = 0
            characteristics['active_duration'] = 0
        
        # Calculate identity quality score
        appearance_quality = characteristics['identity_confidence']
        consistency_quality = 1.0 - characteristics['match_consistency']  # Lower std = higher quality
        activity_quality = min(1.0, characteristics['total_appearances'] / 10.0)  # More appearances = higher quality
        
        characteristics['identity_quality_score'] = (appearance_quality + consistency_quality + activity_quality) / 3.0
        
        return characteristics
    
    def assign_player_ids(self) -> List[PlayerIdentity]:
        """
        Assign consistent unique IDs to players across both videos.
        
        This is the core function that maintains identity consistency.
        """
        
        logger.info("🎯 Assigning consistent player IDs...")
        logger.info("Ensuring the same person has the same ID in both videos!")
        
        # Group matches by player identity
        player_groups = self.group_matches_by_player()
        
        # Create player identities
        player_identities = []
        id_counter = 1
        
        # Sort groups by quality (best matches get lower IDs)
        sorted_groups = sorted(
            player_groups.items(),
            key=lambda x: len(x[1]['matches']) * np.mean([m.overall_similarity for m in x[1]['matches']]),
            reverse=True
        )
        
        for group_id, player_group in sorted_groups:
            # Calculate characteristics
            characteristics = self.calculate_identity_characteristics(player_group)
            
            # Only assign permanent IDs to players with sufficient appearances and confidence
            if (characteristics['total_appearances'] >= self.min_appearances and 
                characteristics['identity_confidence'] >= self.confidence_threshold):
                
                player_id = f"{self.id_prefix}{id_counter:03d}"
                id_counter += 1
            else:
                # Assign temporary ID for low-confidence players
                player_id = f"TEMP_{group_id:03d}"
            
            # Create player identity
            identity = PlayerIdentity(
                player_id=player_id,
                identity_confidence=characteristics['identity_confidence'],
                broadcast_detections=player_group['broadcast_detections'],
                tacticam_detections=player_group['tacticam_detections'],
                primary_jersey_color=characteristics['primary_jersey_color'],
                team_assignment=characteristics['team_assignment'],
                total_appearances=characteristics['total_appearances'],
                broadcast_appearances=characteristics['broadcast_appearances'],
                tacticam_appearances=characteristics['tacticam_appearances'],
                first_seen_frame=characteristics['first_seen_frame'],
                last_seen_frame=characteristics['last_seen_frame'],
                active_duration=characteristics['active_duration'],
                identity_quality_score=characteristics['identity_quality_score'],
                match_consistency=1.0 - characteristics['match_consistency']  # Convert to positive metric
            )
            
            player_identities.append(identity)
        
        self.player_identities = player_identities
        
        # Log results
        permanent_ids = [p for p in player_identities if not p.player_id.startswith('TEMP_')]
        temporary_ids = [p for p in player_identities if p.player_id.startswith('TEMP_')]
        
        logger.info(f"✅ ID assignment complete!")
        logger.info(f"Permanent player IDs assigned: {len(permanent_ids)}")
        logger.info(f"Temporary IDs assigned: {len(temporary_ids)}")
        
        return player_identities
    
    def create_player_database(self) -> Dict:
        """Create a comprehensive player database with all assigned IDs."""
        
        database = {
            'metadata': {
                'total_players': len(self.player_identities),
                'permanent_ids': len([p for p in self.player_identities if not p.player_id.startswith('TEMP_')]),
                'temporary_ids': len([p for p in self.player_identities if p.player_id.startswith('TEMP_')]),
                'assignment_date': str(pd.Timestamp.now()),
                'confidence_threshold': self.confidence_threshold,
                'min_appearances_threshold': self.min_appearances
            },
            'players': {}
        }
        
        for identity in self.player_identities:
            player_data = {
                'player_id': identity.player_id,
                'identity_confidence': float(identity.identity_confidence),
                'team_assignment': identity.team_assignment,
                'primary_jersey_color': {
                    'rgb': list(identity.primary_jersey_color),
                    'hex': f"#{identity.primary_jersey_color[0]:02x}{identity.primary_jersey_color[1]:02x}{identity.primary_jersey_color[2]:02x}"
                },
                'activity_summary': {
                    'total_appearances': identity.total_appearances,
                    'broadcast_appearances': identity.broadcast_appearances,
                    'tacticam_appearances': identity.tacticam_appearances,
                    'first_seen_frame': identity.first_seen_frame,
                    'last_seen_frame': identity.last_seen_frame,
                    'active_duration': identity.active_duration
                },
                'quality_metrics': {
                    'identity_quality_score': float(identity.identity_quality_score),
                    'match_consistency': float(identity.match_consistency)
                },
                'detections': {
                    'broadcast': identity.broadcast_detections,
                    'tacticam': identity.tacticam_detections
                }
            }
            
            database['players'][identity.player_id] = player_data
        
        return database
    
    def validate_id_consistency(self) -> Dict:
        """Validate that ID assignments maintain consistency across videos."""
        
        validation_results = {
            'consistency_check': 'PASSED',
            'issues_found': [],
            'statistics': {},
            'quality_analysis': {}
        }
        
        # Check 1: No duplicate IDs
        all_ids = [p.player_id for p in self.player_identities]
        duplicate_ids = [id for id, count in Counter(all_ids).items() if count > 1]
        
        if duplicate_ids:
            validation_results['consistency_check'] = 'FAILED'
            validation_results['issues_found'].append(f"Duplicate IDs found: {duplicate_ids}")
        
        # Check 2: No detection appears in multiple identities
        all_broadcast_detections = []
        all_tacticam_detections = []
        
        for identity in self.player_identities:
            all_broadcast_detections.extend(identity.broadcast_detections)
            all_tacticam_detections.extend(identity.tacticam_detections)
        
        duplicate_broadcast = [det for det, count in Counter(all_broadcast_detections).items() if count > 1]
        duplicate_tacticam = [det for det, count in Counter(all_tacticam_detections).items() if count > 1]
        
        if duplicate_broadcast:
            validation_results['consistency_check'] = 'FAILED'
            validation_results['issues_found'].append(f"Duplicate broadcast detections: {len(duplicate_broadcast)}")
            
        if duplicate_tacticam:
            validation_results['consistency_check'] = 'FAILED'
            validation_results['issues_found'].append(f"Duplicate tacticam detections: {len(duplicate_tacticam)}")
        
        # Statistics
        validation_results['statistics'] = {
            'total_identities': len(self.player_identities),
            'permanent_ids': len([p for p in self.player_identities if not p.player_id.startswith('TEMP_')]),
            'temporary_ids': len([p for p in self.player_identities if p.player_id.startswith('TEMP_')]),
            'avg_appearances_per_player': np.mean([p.total_appearances for p in self.player_identities]),
            'avg_identity_confidence': np.mean([p.identity_confidence for p in self.player_identities])
        }
        
        # Quality analysis
        high_quality = [p for p in self.player_identities if p.identity_quality_score > 0.8]
        medium_quality = [p for p in self.player_identities if 0.6 <= p.identity_quality_score <= 0.8]
        low_quality = [p for p in self.player_identities if p.identity_quality_score < 0.6]
        
        validation_results['quality_analysis'] = {
            'high_quality_ids': len(high_quality),
            'medium_quality_ids': len(medium_quality), 
            'low_quality_ids': len(low_quality),
            'quality_distribution': {
                'high': len(high_quality),
                'medium': len(medium_quality),
                'low': len(low_quality)
            }
        }
        
        return validation_results
    
    def create_id_visualizations(self, output_dir: str):
        """Create visualizations showing player ID assignments."""
        
        if not self.player_identities:
            logger.warning("No player identities to visualize")
            return
        
        output_path = Path(output_dir)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Player ID Assignment Analysis', fontsize=16, fontweight='bold')
        
        # 1. ID Quality Distribution
        quality_scores = [p.identity_quality_score for p in self.player_identities]
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        axes[0, 0].set_title('Identity Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(quality_scores):.3f}')
        axes[0, 0].legend()
        
        # 2. Appearances per Player
        appearances = [p.total_appearances for p in self.player_identities]
        axes[0, 1].hist(appearances, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
        axes[0, 1].set_title('Appearances per Player')
        axes[0, 1].set_xlabel('Total Appearances')
        axes[0, 1].set_ylabel('Number of Players')
        axes[0, 1].axvline(self.min_appearances, color='red', linestyle='--', 
                          label=f'Min threshold: {self.min_appearances}')
        axes[0, 1].legend()
        
        # 3. Permanent vs Temporary IDs
        permanent_count = len([p for p in self.player_identities if not p.player_id.startswith('TEMP_')])
        temporary_count = len([p for p in self.player_identities if p.player_id.startswith('TEMP_')])
        
        labels = ['Permanent IDs', 'Temporary IDs']
        sizes = [permanent_count, temporary_count]
        colors = ['lightcoral', 'lightskyblue']
        
        axes[0, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('ID Type Distribution')
        
        # 4. Identity Confidence Distribution
        confidences = [p.identity_confidence for p in self.player_identities]
        axes[1, 0].hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_title('Identity Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Number of Players')
        axes[1, 0].axvline(self.confidence_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {self.confidence_threshold}')
        axes[1, 0].legend()
        
        # 5. Cross-Camera Appearance Balance
        broadcast_apps = [p.broadcast_appearances for p in self.player_identities]
        tacticam_apps = [p.tacticam_appearances for p in self.player_identities]
        
        axes[1, 1].scatter(broadcast_apps, tacticam_apps, alpha=0.6, color='purple')
        axes[1, 1].set_title('Cross-Camera Appearance Balance')
        axes[1, 1].set_xlabel('Broadcast Appearances')
        axes[1, 1].set_ylabel('Tacticam Appearances')
        axes[1, 1].plot([0, max(max(broadcast_apps), max(tacticam_apps))], 
                       [0, max(max(broadcast_apps), max(tacticam_apps))], 
                       'r--', alpha=0.5, label='Perfect Balance')
        axes[1, 1].legend()
        
        # 6. Team Distribution
        team_counts = Counter(p.team_assignment for p in self.player_identities)
        teams = list(team_counts.keys())
        counts = list(team_counts.values())
        
        bars = axes[1, 2].bar(teams, counts, color=['red', 'blue', 'green', 'orange'][:len(teams)])
        axes[1, 2].set_title('Players per Team')
        axes[1, 2].set_xlabel('Team')
        axes[1, 2].set_ylabel('Number of Players')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'player_id_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ID assignment visualizations saved to {output_path}")
    
    def save_id_assignments(self, output_dir: str):
        """Save all player ID assignments and database."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save player identities as pickle
        with open(output_path / 'player_identities.pkl', 'wb') as f:
            pickle.dump(self.player_identities, f)
        
        # Save as JSON for human readability
        identities_json = []
        for identity in self.player_identities:
            identity_dict = asdict(identity)
            # Convert numpy types to Python types for JSON serialization
            identity_dict = self._convert_numpy_types(identity_dict)
            identities_json.append(identity_dict)
        
        with open(output_path / 'player_identities.json', 'w') as f:
            json.dump(identities_json, f, indent=2)
        
        # Create and save player database
        database = self.create_player_database()
        database = self._convert_numpy_types(database)
        with open(output_path / 'player_database.json', 'w') as f:
            json.dump(database, f, indent=2)
        
        # Save validation results
        validation = self.validate_id_consistency()
        validation = self._convert_numpy_types(validation)
        with open(output_path / 'id_validation.json', 'w') as f:
            json.dump(validation, f, indent=2)
        
        logger.info(f"Player ID assignments saved to {output_path}")
        
        return validation
    
    def run_id_assignment_pipeline(self) -> Tuple[List[PlayerIdentity], Dict]:
        """Run the complete player ID assignment pipeline."""
        
        logger.info("🎯 Starting consistent player ID assignment...")
        logger.info("Goal: Ensure the same person has the same ID in both videos!")
        
        # Load data
        self.load_matching_results()
        self.create_feature_lookup_tables()
        
        # Assign player IDs
        player_identities = self.assign_player_ids()
        
        # Validate assignments
        validation = self.validate_id_consistency()
        
        logger.info("✅ Player ID assignment complete!")
        logger.info(f"Total player identities: {len(player_identities)}")
        logger.info(f"Validation status: {validation['consistency_check']}")
        
        return player_identities, validation
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

def main():
    """Main function to run player ID assignment."""
    
    print("🎯 Phase 6: Consistent Player ID Assignment")
    print("=" * 60)
    print("🧠 Goal: Assign unique IDs ensuring identity consistency")
    print("👤 Same person = Same ID in both videos")
    print("=" * 60)
    
    # Initialize configuration and assigner
    config = Config()
    assigner = ConsistentIDAssigner(config)
    
    # Run ID assignment pipeline
    player_identities, validation = assigner.run_id_assignment_pipeline()
    
    # Save results
    output_dir = config.RESULTS_DIR / 'PlayerIDs'
    validation_results = assigner.save_id_assignments(str(output_dir))
    
    # Create visualizations
    assigner.create_id_visualizations(str(output_dir))
    
    # Final summary
    print("\n" + "="*60)
    print("✅ CONSISTENT PLAYER ID ASSIGNMENT COMPLETED!")
    print("="*60)
    
    permanent_ids = [p for p in player_identities if not p.player_id.startswith('TEMP_')]
    temporary_ids = [p for p in player_identities if p.player_id.startswith('TEMP_')]
    
    print(f"👤 Total player identities created: {len(player_identities)}")
    print(f"🎯 Permanent player IDs: {len(permanent_ids)}")
    print(f"⏳ Temporary IDs: {len(temporary_ids)}")
    
    if validation_results['consistency_check'] == 'PASSED':
        print(f"✅ Validation: {validation_results['consistency_check']}")
    else:
        print(f"⚠️ Validation: {validation_results['consistency_check']}")
        for issue in validation_results['issues_found']:
            print(f"   - {issue}")
    
    print(f"\n🏆 Best Player Examples:")
    best_players = sorted(player_identities, key=lambda x: x.identity_quality_score, reverse=True)[:5]
    for i, player in enumerate(best_players, 1):
        print(f"   {i}. {player.player_id} - Quality: {player.identity_quality_score:.3f}, "
              f"Appearances: {player.total_appearances} (B:{player.broadcast_appearances}, T:{player.tacticam_appearances})")
    
    print(f"\n📁 Results saved in: {output_dir}")
    print("🎉 Identity consistency achieved across both videos!")

if __name__ == "__main__":
    main()
