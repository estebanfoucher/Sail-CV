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

def process_video(video_path):
    pass

def main():
    video_files = list_videos(DATA_FOLDER)
    print(f"Found {len(video_files)} videos.")
    
    for video_path in video_files:
        print(f"Processing {video_path}...")
        process_video(video_path)
        print("-" * 50)

if __name__ == "__main__":
    main()
