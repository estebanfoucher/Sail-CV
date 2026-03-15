"""Tests for crop module tracking pipeline."""

import json
from pathlib import Path

import pytest
from detector import FakeDetector
from dumper import Dumper
from pipeline import Pipeline
from streamer import Streamer
from video import VideoReader

from crop_module.background_detector import BackgroundDetectorOCV
from crop_module.mask_detector import MaskDetectorGrabCut
from models import Layout, PipelineConfig

project_root = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "parameters_file",
    ["parameters/test_classif.yml", "parameters/test_vector.yml"],
)
def test_pipeline_with_fixture(parameters_file: str):
    """Test pipeline using the C1 fixture (20 frames) with classif or vector config."""
    parameters_path = project_root / parameters_file
    config_stem = Path(parameters_file).stem  # test_classif or test_vector

    # Paths
    fixture_video = project_root / "fixtures" / "C1_fixture.mp4"
    fixture_layout = project_root / "fixtures" / "C1_layout.json"
    output_folder = project_root / "output_tests" / "pipeline"

    # Validate fixture exists
    assert fixture_video.exists(), f"Fixture video not found: {fixture_video}"
    assert fixture_layout.exists(), f"Fixture layout not found: {fixture_layout}"
    assert parameters_path.exists(), f"Parameters file not found: {parameters_path}"

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = PipelineConfig.from_yaml(parameters_path)
    has_classifier = config.classifier is not None

    # Load layout
    with fixture_layout.open() as f:
        layout_data = json.load(f)
    layout = Layout.from_json_dict(layout_data)

    # Initialize pipeline
    pipeline = Pipeline(config, layout, project_root=project_root)

    # Verify that FakeDetector is being used
    assert isinstance(pipeline.detector, FakeDetector), (
        f"Expected FakeDetector, got {type(pipeline.detector)}"
    )

    # Output paths per config so runs do not overwrite each other
    output_json_path = output_folder / f"C1_fixture_tracked_{config_stem}.json"
    output_video_path = output_folder / f"C1_fixture_tracked_{config_stem}.mp4"
    output_fgmask_path = (
        output_folder / f"C1_fixture_fgmask_{config_stem}.mp4"
        if config.output.generate_fgmask_video
        else None
    )

    # Initialize dumper (main tracking video when output_tracking_video is true)
    dumper = Dumper(
        output_json_path=output_json_path,
        output_video_path=output_video_path
        if config.output.output_tracking_video
        else None,
        output_fgmask_path=output_fgmask_path,
    )

    # Stream video, process frames, and dump results
    with Streamer(fixture_video) as streamer:
        # Initialize pipeline for video dimensions
        pipeline.initialize_for_video(streamer.width, streamer.height, streamer.fps)

        # Verify that BackgroundDetectorOCV is being used
        assert isinstance(pipeline.background_detector, BackgroundDetectorOCV), (
            f"Expected BackgroundDetectorOCV, got {type(pipeline.background_detector)}"
        )

        # Verify that MaskDetectorGrabCut is being used
        assert isinstance(pipeline.mask_detector, MaskDetectorGrabCut), (
            f"Expected MaskDetectorGrabCut, got {type(pipeline.mask_detector)}"
        )

        # Initialize dumper video writers
        dumper.initialize_video_writers(streamer.fps, (streamer.width, streamer.height))

        # Main processing loop
        for frame_number, frame in streamer:
            # Process frame through pipeline
            result = pipeline.process_frame(frame, frame_number)

            # Dump result
            dumper.dump_frame(result)

    # Close dumper
    dumper.close()

    # Verify outputs exist
    assert output_json_path.exists(), f"Output JSON not created: {output_json_path}"

    # Verify output video has correct frame count (20 frames) if it was created
    if (
        config.output.output_tracking_video
        and output_video_path
        and output_video_path.exists()
    ):
        reader = VideoReader.open_video_file(str(output_video_path))
        assert reader.specs.frame_count == 20, (
            f"Output video has {reader.specs.frame_count} frames, expected 20"
        )
        reader.release()

    # Verify output JSON structure
    with output_json_path.open() as f:
        results = json.load(f)

    assert isinstance(results, list), "Results should be a list of frames"
    assert len(results) == 20, f"Expected 20 frames in results, got {len(results)}"

    # Check that at least some frames have tracks
    frames_with_tracks = sum(1 for frame in results if frame.get("tracks"))
    assert frames_with_tracks > 0, "No frames have tracks"

    # Check that at least some frames have PCA vectors
    frames_with_pca = sum(1 for frame in results if frame.get("pca_vectors"))
    assert frames_with_pca > 0, "No frames have PCA vectors"

    # Verify frame structure
    for frame_result in results:
        assert "frame_number" in frame_result, "Frame missing frame_number"
        assert "tracks" in frame_result, "Frame missing tracks"
        assert "pca_vectors" in frame_result, "Frame missing pca_vectors"
        assert isinstance(frame_result["tracks"], list), "Tracks should be a list"
        assert isinstance(frame_result["pca_vectors"], dict), (
            "PCA vectors should be a dict"
        )

    # Config-specific assertions: classifier vs vector-only
    if has_classifier:
        # test_classif: require classifications in every frame that has tracks
        for frame_result in results:
            if frame_result.get("tracks"):
                assert "classifications" in frame_result, (
                    "Classifier config must produce classifications for frames with tracks"
                )
                classifications = frame_result["classifications"]
                assert isinstance(classifications, dict), (
                    "classifications should be a dict"
                )
                tracks = frame_result["tracks"]
                assert len(classifications) == len(tracks), (
                    f"Expected one classification per track, "
                    f"got {len(classifications)} for {len(tracks)} tracks"
                )
                for track in tracks:
                    track_id = track.get("track_id")
                    assert track_id is not None
                    assert track_id in classifications, (
                        f"Track {track_id} missing from classifications"
                    )
                    assert isinstance(classifications[track_id], int), (
                        f"Classification for track {track_id} should be int"
                    )
    else:
        # test_vector: no classifier, so classifications should be absent
        for frame_result in results:
            assert "classifications" not in frame_result or not frame_result.get(
                "classifications"
            ), "Vector config should not produce classifications"

    # Check PCA vectors format (should be 2D unit vectors)
    for frame_result in results:
        pca_vectors = frame_result["pca_vectors"]
        for track_id, vector in pca_vectors.items():
            assert isinstance(vector, list), (
                f"PCA vector for track {track_id} should be a list"
            )
            assert len(vector) == 2, (
                f"PCA vector for track {track_id} should be 2D, got {len(vector)}"
            )
            # Check that vector is approximately unit length (within tolerance)
            magnitude = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
            assert abs(magnitude - 1.0) < 0.1, (
                f"PCA vector for track {track_id} should be unit length, got magnitude {magnitude}"
            )

    print("✓ Pipeline test passed")
    print(f"  - Config: {parameters_file}")
    print(f"  - Output JSON: {output_json_path}")
    if output_video_path and output_video_path.exists():
        print(f"  - Output video: {output_video_path}")
    print(f"  - Frames with tracks: {frames_with_tracks}/20")
    print(f"  - Frames with PCA vectors: {frames_with_pca}/20")
    if has_classifier:
        frames_with_class = sum(1 for frame in results if frame.get("classifications"))
        print(f"  - Frames with classifications: {frames_with_class}/20")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "test_pipeline_with_fixture"])
