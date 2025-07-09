# Note about Large Files

This repository has been optimized for GitHub by excluding large binary files:

## Excluded Files (Due to Size Limitations)
- `*.pkl` files (Feature and detection data - can be regenerated)
- `*.mp4` files (Video inputs and outputs - too large for GitHub)
- `*.pt` files (YOLOv8 model weights - download separately)
- Binary dependencies (DLLs, libs - installed via pip)

## To Run the System

1. **Install Dependencies**:
   ```bash
   pip install ultralytics opencv-python numpy matplotlib seaborn pillow scikit-learn
   ```

2. **Download YOLOv8 Model**:
   ```bash
   # The system will automatically download yolov8n.pt on first run
   ```

3. **Add Input Videos**:
   - Place your input videos in `Videos/` directory
   - Name them `broadcast.mp4` and `tacticam.mp4`

4. **Run the System**:
   ```bash
   python Code/final_system_demonstration.py
   ```

## Repository Contents

This repository contains the essential code and documentation:
- ✅ Complete Python implementation
- ✅ Technical documentation and reports  
- ✅ Configuration files
- ✅ Setup instructions
- ✅ Generated analysis plots and proofs (PNG files only)

The system is fully functional and will regenerate all data files when run with appropriate input videos.

## Generated Outputs

When you run the system, it will create:
- Annotated video files with consistent player IDs
- Feature extraction data
- Cross-camera matching results
- Player identity database
- Visual proof and validation plots

## Original Project Status

✅ **COMPLETED**: Player re-identification with consistent IDs across all frames and videos  
✅ **VERIFIED**: System produces professional-quality annotated videos  
✅ **DOCUMENTED**: Comprehensive technical report and code documentation  
✅ **REPRODUCIBLE**: Complete pipeline with clear setup instructions  

The system successfully assigns consistent, unique player IDs across multiple camera views in sports footage.
