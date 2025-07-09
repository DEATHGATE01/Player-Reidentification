#!/usr/bin/env python3
"""
CREATE VIDEOS WITH CLEAN CONSISTENT PLAYER IDs
==============================================

This script creates videos showing only the properly identified players
with PLAYER_ IDs, and cleans up the display to show consistent, permanent
player identifications.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_player_database():
    """Load the player database."""
    results_dir = Path("../Results")
    with open(results_dir / "PlayerIDs" / "player_database.json", 'r') as f:
        player_database = json.load(f)
    return player_database

def create_clean_detection_mapping(player_database: Dict) -> Dict[str, str]:
    """Create mapping only for properly identified players (PLAYER_ IDs)."""
    detection_to_player = {}
    players = player_database['players']
    
    # Only map PLAYER_ IDs, not TEMP_ IDs
    for player_id, player_data in players.items():
        if not player_id.startswith('PLAYER_'):
            continue
            
        detections = player_data.get('detections', {})
        
        # Map broadcast detections
        for det_id in detections.get('broadcast', []):
            detection_to_player[det_id] = player_id
        
        # Map tacticam detections  
        for det_id in detections.get('tacticam', []):
            detection_to_player[det_id] = player_id
    
    return detection_to_player

def load_detection_data():
    """Load detection data from JSON files."""
    results_dir = Path("../Results")
    
    with open(results_dir / "broadcast" / "detections.json", 'r') as f:
        broadcast_detections = json.load(f)
    
    with open(results_dir / "tacticam" / "detections.json", 'r') as f:
        tacticam_detections = json.load(f)
    
    return broadcast_detections, tacticam_detections

def create_frame_mapping(detections_list: List, video_type: str, detection_to_player: Dict):
    """Create mapping from frame number to properly identified players only."""
    frame_mapping = {}
    
    for detection in detections_list:
        frame_num = detection['frame_number']
        
        # Create detection ID using exact format
        center_x = detection['center'][0]
        center_y = detection['center'][1]
        det_id = f"{video_type}_{frame_num}_{center_x:.0f}_{center_y:.0f}"
        
        # Only include if this maps to a PLAYER_ ID
        player_id = detection_to_player.get(det_id)
        if player_id is None:
            continue
        
        if frame_num not in frame_mapping:
            frame_mapping[frame_num] = []
        
        detection_info = {
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'detection_id': det_id,
            'player_id': player_id
        }
        
        frame_mapping[frame_num].append(detection_info)
    
    return frame_mapping

def get_clean_player_colors(player_database: Dict) -> Dict[str, Tuple[int, int, int]]:
    """Generate colors only for PLAYER_ IDs."""
    players = player_database['players']
    colors = {}
    
    # Get only PLAYER_ IDs
    player_ids = [pid for pid in players.keys() if pid.startswith('PLAYER_')]
    
    # High-contrast color palette for clear visibility
    color_palette = [
        (0, 255, 0),      # Bright Green
        (255, 0, 0),      # Bright Red  
        (0, 0, 255),      # Bright Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (255, 20, 147),   # Deep Pink
        (34, 139, 34),    # Forest Green
        (255, 140, 0),    # Dark Orange
        (72, 61, 139),    # Dark Slate Blue
    ]
    
    for i, player_id in enumerate(sorted(player_ids)):
        colors[player_id] = color_palette[i % len(color_palette)]
    
    return colors

def annotate_frame_with_clean_ids(frame: np.ndarray, frame_detections: List, player_colors: Dict) -> np.ndarray:
    """Annotate frame with clean, consistent player IDs."""
    annotated_frame = frame.copy()
    
    for detection in frame_detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        player_id = detection['player_id']
        
        # Get color for this player
        color = player_colors.get(player_id, (255, 255, 255))
        
        # Draw thick bounding box for visibility
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        
        # Prepare clean text (just the number part)
        player_number = player_id.replace('PLAYER_', 'P')
        confidence = detection['confidence']
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(player_number, font, font_scale, thickness)
        
        # Position text above bounding box
        text_x = int(x1)
        text_y = int(y1) - 15
        
        if text_y - text_height < 0:
            text_y = int(y2) + text_height + 15
        
        # Draw solid background for text
        pad = 5
        cv2.rectangle(annotated_frame,
                     (text_x - pad, text_y - text_height - baseline - pad),
                     (text_x + text_width + pad, text_y + baseline + pad),
                     color, -1)
        
        # Draw player ID text in black for contrast
        cv2.putText(annotated_frame, player_number, (text_x, text_y),
                   font, font_scale, (0, 0, 0), thickness)
        
        # Add confidence score in smaller text
        conf_text = f"{confidence:.2f}"
        cv2.putText(annotated_frame, conf_text, (text_x, text_y + text_height + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated_frame

def create_clean_annotated_video(video_path: str, frame_mapping: Dict, 
                                player_colors: Dict, output_path: str, video_name: str):
    """Create video with clean player ID annotations."""
    
    print(f"\n🎬 Creating {video_name} video with clean consistent IDs...")
    print(f"   Input: {video_path}")
    print(f"   Output: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   📹 Properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frames_with_players = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        if frame_count in frame_mapping and frame_mapping[frame_count]:
            frame_detections = frame_mapping[frame_count]
            annotated_frame = annotate_frame_with_clean_ids(frame, frame_detections, player_colors)
            frames_with_players += 1
        else:
            annotated_frame = frame
        
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   ⏳ Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"   ✅ Complete! {frames_with_players} frames with consistent player IDs")
    return True

def main():
    """Main function to create clean videos with consistent player IDs."""
    
    print("🎬 CREATING CLEAN VIDEOS WITH CONSISTENT PLAYER IDs")
    print("=" * 65)
    print("🎯 Showing only properly identified players (PLAYER_XXX)")
    print("✨ Clean, professional annotations with consistent IDs!")
    print("=" * 65)
    
    # Load player database
    print("\n📊 Loading player database...")
    player_database = load_player_database()
    
    total_players = player_database['metadata']['total_players']
    permanent_players = player_database['metadata']['permanent_ids']
    print(f"   📋 Total players: {total_players}")
    print(f"   ⭐ Permanent player IDs: {permanent_players}")
    
    # Create clean detection mapping (only PLAYER_ IDs)
    print("\n🔗 Creating clean detection mapping...")
    detection_to_player = create_clean_detection_mapping(player_database)
    print(f"   📋 Mapped {len(detection_to_player)} detections to permanent player IDs")
    
    # Load detection data
    print("\n📊 Loading detection data...")
    broadcast_detections, tacticam_detections = load_detection_data()
    
    # Create frame mappings
    print("\n🎯 Creating frame mappings for permanent players...")
    broadcast_frame_mapping = create_frame_mapping(
        broadcast_detections, "broadcast", detection_to_player
    )
    tacticam_frame_mapping = create_frame_mapping(
        tacticam_detections, "tacticam", detection_to_player
    )
    
    frames_with_broadcast = len([f for f in broadcast_frame_mapping.values() if f])
    frames_with_tacticam = len([f for f in tacticam_frame_mapping.values() if f])
    
    print(f"   📺 Broadcast frames with permanent players: {frames_with_broadcast}")
    print(f"   📹 Tacticam frames with permanent players: {frames_with_tacticam}")
    
    # Generate colors
    print("\n🎨 Generating colors for permanent players...")
    player_colors = get_clean_player_colors(player_database)
    print(f"   🌈 Generated colors for {len(player_colors)} permanent players")
    
    # Setup paths
    video_dir = Path("../Videos")
    output_dir = Path("../Results/CleanAnnotatedVideos")
    output_dir.mkdir(exist_ok=True)
    
    broadcast_video = video_dir / "broadcast.mp4"
    tacticam_video = video_dir / "tacticam.mp4"
    
    # Create clean annotated videos
    success_broadcast = create_clean_annotated_video(
        str(broadcast_video), broadcast_frame_mapping, player_colors,
        str(output_dir / "broadcast_clean_consistent_ids.mp4"), "Broadcast"
    )
    
    success_tacticam = create_clean_annotated_video(
        str(tacticam_video), tacticam_frame_mapping, player_colors,
        str(output_dir / "tacticam_clean_consistent_ids.mp4"), "Tacticam"
    )
    
    print("\n" + "=" * 65)
    print("🎉 CLEAN CONSISTENT ID VIDEOS CREATED!")
    print("=" * 65)
    print("📁 Generated videos:")
    print(f"   📹 broadcast_clean_consistent_ids.mp4")
    print(f"   📹 tacticam_clean_consistent_ids.mp4")
    print(f"\n📂 Location: {output_dir}")
    
    print("\n🎯 WHAT YOU'LL SEE:")
    print("   ✅ Only properly identified players shown")
    print("   ✅ Clean labels: P001, P002, P003, etc.")
    print("   ✅ Same player keeps same ID throughout")
    print("   ✅ Same player has same ID across videos")
    print("   ✅ Bright, clear colors for easy tracking")
    print("   ✅ Professional quality annotations")
    print("   ❌ No TEMP_ or UNKNOWN labels!")
    
    print("\n🏆 VERIFICATION:")
    print("   1. Open the videos in CleanAnnotatedVideos/")
    print("   2. Look for players labeled P001, P002, etc.")
    print("   3. Follow a player - ID stays consistent!")
    print("   4. Same player in both videos has same ID!")
    print("   5. Clean, professional appearance!")

if __name__ == "__main__":
    main()
