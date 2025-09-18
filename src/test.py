import os
import cv2




def test_video_reader_and_writer():
    from video import VideoReader, FFmpegVideoWriter

    video_path = "/app/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/app/output/tests/video_reader_and_writer"
    output_name = "output_test_video_reader_and_writer.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
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

    video_path = "/app/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/app/output/tests/sam"
    output_name = "output_test_sam.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    writer = FFmpegVideoWriter(output_path, reader.fps, reader.frame_size)
    
    frame_count = 10
    sam = SAM("FastSAM-x.pt")
    
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

    video_path = "/app/data/calibration_intrinsics_1/camera_1/GH010815.MP4"
    OUTPUT_FOLDER = "/app/output/tests/tell_tale_detector"
    output_name = f"output_test_tell_tale_detector_{architecture}.mp4"
    reader = VideoReader(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
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

def test_mv_utils():
    from mv_utils import Scene, load_stereo_data_folder_structure
    
    stereo_data_folder_structure = load_stereo_data_folder_structure()
    scene_names = stereo_data_folder_structure.get_scene_folders()
    
    for scene_name in scene_names:
        scene = Scene(scene_name)
        extrinsic_calibration = scene.create_extrinsic_calibration()
        extrinsic_calibration.calibrate_extrinsics(force_recompute=True)
        extrinsic_calibration.save_extrinsics_summary()
        
        # Add explicit cleanup
        extrinsic_calibration.cleanup()
        
        print(f"Scene {scene_name} passed")
        
    
    

def test_stereo_video_reader():
    from video import StereoVideoReader, FFmpegVideoWriter
    from mv_utils import Scene, load_stereo_data_folder_structure
    stereo_data_folder_structure = load_stereo_data_folder_structure()
    scene_names = stereo_data_folder_structure.get_scene_folders()
    scene_name = "scene_3"
    assert scene_name in scene_names, f"Scene {scene_name} is not in the list of scene names"
    
    scene = Scene(scene_name)
    video_paths = scene.get_video_paths()
    sync_frame_offset = scene.sync_frame_offset
    start_frame = 900
    stereo_video_reader = StereoVideoReader(video_paths["camera_1"], video_paths["camera_2"], start_video_1_frame=start_frame, start_video_2_frame=start_frame - sync_frame_offset)
    frame_count = 100
    OUTPUT_FOLDER = "/app/output/tests/stereo_video_reader"
    output_name = f"output_test_stereo_video_reader.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Read first frame to get the combined frame size
    ret, ret_2, frame_1, frame_2 = stereo_video_reader.read()
    if not ret or not ret_2:
        print("Failed to read first frame")
        return
    
    # Create a test combined frame to get the correct dimensions
    test_combined_frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
    combined_height, combined_width = test_combined_frame.shape[:2]
    combined_frame_size = (combined_width, combined_height)
    
    writer = FFmpegVideoWriter(output_path, stereo_video_reader.video_1.fps, combined_frame_size)
    writer.write(test_combined_frame)  # Write the first frame
    
    for i in range(start_frame + 1, start_frame + frame_count): 
        ret, ret_2, frame_1, frame_2 = stereo_video_reader.read()
        if not ret or not ret_2:
            break
        frame = stereo_video_reader.render_side_by_side(frame_1, frame_2)
        writer.write(frame)
        print(f"Stereo rendered frame {i}/{start_frame + frame_count}")
    writer.release()
    stereo_video_reader.release()
    print("Test stereo video reader passed")


def test_stereo_image_saver():
    from video import StereoVideoReader, FFmpegVideoWriter
    from mv_utils import Scene, load_stereo_data_folder_structure
    stereo_data_folder_structure = load_stereo_data_folder_structure()
    scene_names = stereo_data_folder_structure.get_scene_folders()
    scene_name = "scene_8"
    assert scene_name in scene_names, f"Scene {scene_name} is not in the list of scene names"
    
    scene = Scene(scene_name)
    video_paths = scene.get_video_paths()
    sync_frame_offset = scene.sync_frame_offset
    frame_to_save_number_list = [4900,4910,4920,4930,4940,4950,4960]
    # save frame 1 and frame 2 to the output folder
    output_folder = "/app/output/tests/stereo_image_saver"
    output_name_1 = f"frame_1.png"
    output_name_2 = f"frame_2.png"
    
    os.makedirs(output_folder, exist_ok=True)
    for frame_to_save_number in frame_to_save_number_list:
        stereo_video_reader = StereoVideoReader(video_paths["camera_1"], video_paths["camera_2"], start_video_1_frame=frame_to_save_number, start_video_2_frame=frame_to_save_number - sync_frame_offset)
        ret_1, ret_2, frame_1, frame_2 = stereo_video_reader.read()
        output_name_1 = f"frame_1_{frame_to_save_number}.png"
        output_name_2 = f"frame_2_{frame_to_save_number}.png"
        output_path_1 = os.path.join(output_folder, f"{frame_to_save_number}", output_name_1)
        output_path_2 = os.path.join(output_folder, f"{frame_to_save_number}", output_name_2)
        os.makedirs(os.path.join(output_folder, f"{frame_to_save_number}"), exist_ok=True)
        cv2.imwrite(output_path_1, frame_1)
        cv2.imwrite(output_path_2, frame_2)
        stereo_video_reader.release()
        

def test_all():
    #test_stereo_video_reader()
    #test_video_reader_and_writer()
    #test_sam()
    test_mv_utils()
    #test_tell_tale_detector(model_path="/app/models/rt-detr.pt", architecture="rt-detr")
    #test_tell_tale_detector(model_path="/app/models/yolos.pt", architecture="yolo")
    #test_stereo_image_saver()
    print("All tests passed")

if __name__ == "__main__":
    test_all()