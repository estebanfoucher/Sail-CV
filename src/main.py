import cv2
import os
import subprocess
import json

DATA_FOLDER = "/workspace/data/scene_3/camera_1"

def list_videos(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4") or f.endswith(".MP4")]

def get_video_info(video_path):
    """Get video information using ffmpeg"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}")
        return None

def process_video_with_ffmpeg(video_path):
    """Process video using ffmpeg"""
    try:
        # Check file size
        file_size = os.path.getsize(video_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Try to extract a frame
        output_frame = os.path.join("/workspace/output", os.path.basename(video_path).replace('.MP4', '_frame.jpg').replace('.mp4', '_frame.jpg'))
        frame_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vframes', '1', 
            '-q:v', '2', output_frame
        ]
        
        subprocess.run(frame_cmd, check=True, capture_output=True)
        print(f"Extracted frame to: {output_frame}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def main():
    video_files = list_videos(DATA_FOLDER)
    print(f"Found {len(video_files)} videos.")

    for video_path in video_files:
        print(f"Processing {video_path}...")
        success = process_video_with_ffmpeg(video_path)
        if success:
            print(f"Successfully processed {video_path}")
        else:
            print(f"Failed to process {video_path}")
        print("-" * 50)

if __name__ == "__main__":
    main()
