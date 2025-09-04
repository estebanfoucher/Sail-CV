import os


def test_video_reader_and_writer():
    from video import VideoReader, FFmpegVideoWriter

    video_path = "/workspace/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/workspace/output"
    output_name = "output_test_video_reader_and_writer.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)   
    writer = FFmpegVideoWriter(output_path, reader.fps, reader.frame_size)
    
    frame_count = 10
        
    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        writer.write(frame)
    
    writer.release()
    
    # check output video has same size and fps as input video fps upt to 0.1
    reader_output = VideoReader(output_path)
    assert abs(reader_output.fps - reader.fps) < 0.1, f"Output video fps {reader_output.fps} is not the same as input video fps {reader.fps}"
    assert reader_output.frame_size == reader.frame_size, f"Output video frame size {reader_output.frame_size} is not the same as input video frame size {reader.frame_size}"
    assert reader_output.frame_count == frame_count, f"Output video frame count {reader_output.frame_count} is not the same as input video frame count {frame_count}"
    reader_output.release()
    
    reader.release()
    
    print("Test video reader and writer passed")

def test_sam():
    from video import VideoReader, FFmpegVideoWriter
    from unitaries.sam import SAM

    video_path = "/workspace/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/workspace/output"
    output_name = "output_test_sam.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)   
    writer = FFmpegVideoWriter(output_path, reader.fps, reader.frame_size)
    
    frame_count = 10
    sam = SAM()
    
    def process_frame(frame):
        point = (1400, 540)
        result = sam.predict(frame, point, 1)
        frame = sam.render_result(frame, result['mask'], [point])
        return frame
    
    
    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        frame = process_frame(frame)
        writer.write(frame)
    
    writer.release()
    reader.release()
    print("Test sam passed")

def test_tell_tale_detector(model_path, architecture):
    from video import VideoReader, FFmpegVideoWriter
    from unitaries.tell_tale_detector import TellTaleDetector

    video_path = "/workspace/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/workspace/output"
    output_name = f"output_test_tell_tale_detector_{architecture}.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)   
    writer = FFmpegVideoWriter(output_path, reader.fps, reader.frame_size)
    
    frame_count = 10
    detector = TellTaleDetector(model_path=model_path, architecture=architecture)
    
    def process_frame(frame):
        result = detector.predict(frame, conf=0.1)
        frame = detector.render_result(frame, result)
        return frame
    
    
    for _ in range(frame_count):
        ret, frame = reader.read()
        if not ret:
            break
        frame = process_frame(frame)
        writer.write(frame)
    
    writer.release()
    reader.release()
    print("Test tell_tale_detector passed")

    

def test_all():
    test_video_reader_and_writer()
    test_sam()
    test_tell_tale_detector(model_path="models/rt-detr.pt", architecture="rt-detr")
    test_tell_tale_detector(model_path="models/yolos.pt", architecture="yolo")
    print("All tests passed")

if __name__ == "__main__":
    test_all()