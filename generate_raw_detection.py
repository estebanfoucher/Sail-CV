from pathlib import Path
import sys
import json

project_root = Path(__file__).parent
# Ensure local 'src' modules are importable when running this script directly
sys.path.insert(0, str(project_root / "src"))

from models import ModelSpecs, Image
from detector import Detector
from video import VideoReader, FFmpegVideoWriter

from loguru import logger

assets_dir = project_root / "assets"
output_folder = assets_dir / "raw_detection"
output_folder.mkdir(parents=True, exist_ok=True)

# Configure model (default to RT-DETR)
model_path = project_root / "checkpoints" / "best_1_class.pt"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Initialize detector and tracker once
specs = ModelSpecs(model_path=model_path, architecture="rt-detr")
detector = Detector(specs)

# Create output directory
output_folder.mkdir(parents=True, exist_ok=True)

# Convert Path to string for OpenCV
video_path_str = str(assets_dir / "2Ce-CKKCtV4.mp4")

# Open video reader
reader = VideoReader.open_video_file(video_path_str)
fps = reader.specs.fps
total_frames = reader.specs.frame_count

# Process full video
logger.info(f"Processing full video: {total_frames} frames at {fps} fps")

# Dictionary to store detections per frame
detections_by_frame = {}

# Setup video writer
writer = FFmpegVideoWriter(
    str(output_folder / "2Ce-CKKCtV4_raw_detection.mp4"), reader.specs.fps, reader.specs.resolution
)

# Color for rendering bounding boxes
bbox_color = (255, 0, 0)  # Red in BGR

# Process each frame
for frame_num in range(total_frames):
    ret, frame = reader.read()
    if not ret:
        break
    
    # Create Image object
    image = Image(image=frame, rgb_bgr="BGR")
    
    # Run detection
    detections = detector.detect(image)
    
    # Convert detections to serializable format using Pydantic's model_dump()
    detections_list = [detection.model_dump() for detection in detections]
    
    # Store in dictionary with frame number as key
    detections_by_frame[frame_num] = detections_list
    
    # Render detections on frame
    rendered_image = detector.render_result(image, detections, color=bbox_color, thickness=2)
    
    # Write rendered frame to video
    writer.write(rendered_image.image)
    
    # Log progress every 100 frames
    if (frame_num + 1) % 100 == 0:
        logger.info(f"Processed {frame_num + 1}/{total_frames} frames")

reader.release()
writer.release()

# Save to JSON file
output_json_path = output_folder / "2Ce-CKKCtV4_raw_detection.json"
with open(output_json_path, 'w') as f:
    json.dump(detections_by_frame, f, indent=2)

output_video_path = output_folder / "2Ce-CKKCtV4_raw_detection.mp4"
logger.info(f"Detections saved to {output_json_path}")
logger.info(f"Video with rendered boxes saved to {output_video_path}")
logger.info(f"Total frames processed: {len(detections_by_frame)}")