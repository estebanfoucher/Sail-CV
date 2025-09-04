import os
from video import VideoReader, FFmpegVideoWriter

DATA_FOLDER = "/workspace/data/calibration_intrinsics_1/camera_1"

OUTPUT_FOLDER = "/workspace/output"

def list_videos(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4") or f.endswith(".MP4")]

def process_frame(frame):
    # place holder for processing
    return frame

def process_video(video_path):
    print(f"Processing {video_path}")
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path).split(".")[0] + "_output.mp4")   
    writer = FFmpegVideoWriter(output_path, reader.fps, reader.frame_size)
    
    frame_count = reader.frame_count
    
    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        # place holder for processing
        frame = process_frame(frame)
        writer.write(frame)
    
    writer.release()
    
    # check output video has same size and fps as input video fps upt to 0.1
    reader_output = VideoReader(output_path)
    assert abs(reader_output.fps - reader.fps) < 0.1, f"Output video fps {reader_output.fps} is not the same as input video fps {reader.fps}"
    assert reader_output.frame_size == reader.frame_size, f"Output video frame size {reader_output.frame_size} is not the same as input video frame size {reader.frame_size}"
    assert reader_output.frame_count == frame_count, f"Output video frame count {reader_output.frame_count} is not the same as input video frame count {frame_count}"
    reader_output.release()
    
    reader.release()
    
    print(f"Processed {video_path}")


    
def main():
    video_files = list_videos(DATA_FOLDER)
    print(f"Found {len(video_files)} videos.")
    
    for video_path in video_files:
        process_video(video_path)
        

if __name__ == "__main__":
    main()
