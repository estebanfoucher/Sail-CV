# Tracking tests: analysis of usefulness and redundancy

This document analyzes each tracking test file and test: what it covers, what it depends on, and whether it is redundant, deprecated, or optional.

---

## 1. test_pipeline.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_pipeline_with_fixture[test_classif.yml]` | Full pipeline on C1 fixture: FakeDetector + layout tracker + PCA + classifier. Asserts tracks, pca_vectors, classifications. | fixtures/C1_fixture.mp4, C1_layout.json, C1_raw_detections.json, parameters/test_classif.yml, HF or local sailcv-yolo11n-cls224.pt | **Keep** – main integration test for detector + classifier path. |
| `test_pipeline_with_fixture[test_vector.yml]` | Same pipeline without classifier. Asserts tracks, pca_vectors; no classifications. | Same minus classifier weight | **Keep** – main integration test for detector + PCA (vector) path. |

**Summary:** Core tests. No redundancy. Both configs are needed.

---

## 2. test_classifier_minimal.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_classifier_minimal` | Loads C1_fixture_tracked_test_classif.json, extracts crops from first frame, runs Classifier on each crop, saves images and JSON. | Pipeline classif output must exist (run pipeline test first); sailcv-yolo11n-cls224.pt (HF or local) | **Redundant for pipeline coverage.** The pipeline test with test_classif.yml already runs the classifier inside the pipeline and asserts classifications. This test re-implements crop extraction and classification in isolation. **Options:** (a) Remove it and rely on pipeline test for classifier. (b) Keep as a standalone “classifier sanity check” that can run after pipeline (order-dependent). (c) Convert to a unit test that mocks pipeline output or uses a tiny fixture. |

**Summary:** Partially redundant. Safe to remove if you are satisfied with pipeline classif test; otherwise keep as optional sanity check and document the run order.

---

## 3. test_detector.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_tell_tale_detector_rt_detr` | Runs real Detector (rt-detr) on 10 frames of assets/tracking/2Ce-CKKCtV4.mp4, writes video. | sailcv-rtdetrl640.pt (HF or local) | **Keep** – only test that runs the real RT-DETR detector on real video. Pipeline uses FakeDetector. |
| `test_tell_tale_detector_yolo` | Same with YOLO detector. | yolo-s.pt (local only; skips if missing) | **Keep** – only test for YOLO detector. Skip when no local weight is acceptable. |

**Summary:** Not redundant. They are the only tests of the real Detector; pipeline tests use fake detections.

---

## 4. test_track_video.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_track_video_yolo` | Runs track_video() (detector + ByteTracker, no pipeline) on 2Ce-CKKCtV4.mp4, 60 frames. | yolo-s.pt (local); skips if missing | **Keep** – integration test for the track_video script with YOLO. |
| `test_track_video_rt_detr` | Same with RT-DETR on IMG_9496_0.0_3.0.MOV, 10 frames. | sailcv-rtdetrl640.pt (HF or local); video file | **Keep** – integration test for track_video with RT-DETR. |

**Summary:** Not redundant. They test the “detector + tracker only” path (no pipeline, no PCA, no classifier). Different code path from test_pipeline.

---

## 5. test_tracker.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_track_model`, `test_tracker_config_model`, `test_tracker_config_validation`, `test_tracker_initialization`, `test_tracker_update`, `test_track_serialization` | Unit tests for Track and TrackerConfig models, Tracker init/update, serialization. | None | **Keep** – essential unit tests. |
| `test_tracker_integration_with_detector` | One frame: Detector (rt-detr) + Tracker on a blank image. | sailcv-rtdetrl640.pt (HF or local) | **Keep** – light integration test for Detector + Tracker. Overlaps in spirit with test_track_video but no video I/O; useful as a quick integration check. |

**Summary:** No redundancy. Unit tests are necessary; integration test is a lighter complement to test_track_video.

---

## 6. tests/tracking/dataset/test_crop_module.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_loader` | YOLODatasetLoader + Detector rendering on home_telltale. | home_telltale/images (and labels); yolo-s.pt (local); skip if missing | **Optional.** Not redundant with pipeline (different data path). Useful only if you have the optional dataset. |
| `test_crop_extraction` | Crop extraction from dataset pairs. | home_telltale | **Optional.** Same as above. |
| `test_mask_generation` | SAM mask generation on dataset crops. | home_telltale; SAM optional | **Optional.** |
| `test_pca_results` | PCA on dataset crops, visualizations. | home_telltale | **Optional.** |
| `test_full_crop_module_pipeline` | Full flow: load → mask → PCA on dataset. | home_telltale; yolo-s.pt for loader | **Optional.** |

**Summary:** All five tests are **optional** and **skip when home_telltale is missing**. They are not redundant with pipeline tests (pipeline uses C1 fixture + fake detections). They are useful for development with the YOLO dataset; otherwise they just skip. Consider grouping under a “dataset” or “optional” marker so CI can run core tests without them.

---

## 7. test_crop_module_simple.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_pca_1_axis`, `test_pca_1_axis_vertical` | CropModulePCA with n_components=1 on synthetic images (no detector, no pipeline). | None | **Keep** – fast unit tests for PCA math. Pipeline does not assert PCA behavior in isolation. |

**Summary:** Not redundant. Valuable unit-level coverage for the crop module.

---

## 8. test_video_tracking.py

| Test | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| `test_is_ffmpeg_available` | Checks ffmpeg is on PATH. | ffmpeg installed | **Keep** – cheap environment check. |
| `test_video_reader_and_writer` | VideoReader + FFmpegVideoWriter round-trip on IMG_9496_0.0_3.0.MOV. | assets/tracking video | **Keep** – tests video I/O used by pipeline and track_video. |

**Summary:** No redundancy. Useful infrastructure tests.

---

## 9. test_sail_3d_tracker.py and test_sail_3d_tracker_v2.py

| Area | What it does | Depends on | Verdict |
|------|----------------|------------|---------|
| Projection (sail_to_world, world_to_camera, camera_to_pixel, project_telltales, get_sail_corners_world) | Math and projection APIs. | pytest fixtures (synthetic sail/camera) | **Keep** – unit/projection tests. |
| Sail3DTracker / Sail3DTrackerV2 | Assignment, angle recovery, filtering, mesh. | Same fixtures | **Keep** – no overlap with pipeline/detector/classifier. |
| TestSail3DTrackerV2Benchmark::test_latency_target | Latency benchmark (e.g. &lt; 40 ms). | Same fixtures | **Keep but optional.** Can be flaky on slow CI; consider marking as benchmark or running in a separate job. |

**Summary:** No redundancy with pipeline or detector tests. All useful; benchmark is the only one to treat as optional/flaky.

---

## Redundancy and deprecation summary

| Category | Tests | Recommendation |
|----------|--------|-----------------|
| **Redundant** | `test_classifier_minimal` | Redundant with `test_pipeline_with_fixture[test_classif.yml]` for “classifier in pipeline” coverage. Remove, or keep as an optional standalone classifier check and document run order. |
| **Deprecated / legacy** | None clearly deprecated. | `test_tell_tale_detector_yolo` and `test_track_video_yolo` depend on legacy `yolo-s.pt` (not on HF); they are still useful when the file exists and can stay as “skip if missing”. |
| **Optional (skip when resource missing)** | All of `dataset/test_crop_module.py` (home_telltale); `test_tell_tale_detector_yolo`; `test_track_video_yolo` | Keep; they already skip. Optionally mark with `pytest.mark.optional` or run in a separate “full” suite. |
| **Flaky / environment-sensitive** | `TestSail3DTrackerV2Benchmark::test_latency_target` | Keep with relaxed threshold (e.g. 40 ms); consider `pytest.mark.benchmark` or separate CI job. |

---

## Suggested test groups (for CI or local use)

- **Core (always run):**
  test_pipeline (both configs), test_tracker (all), test_crop_module_simple, test_video_tracking, test_detector (rt_detr), test_track_video_rt_detr, test_tracker_integration_with_detector, test_sail_3d_tracker, test_sail_3d_tracker_v2 (excluding benchmark).

- **Optional / when resources exist:**
  test_classifier_minimal (after pipeline classif), test_detector_yolo, test_track_video_yolo, dataset/test_crop_module (when home_telltale and optionally yolo-s.pt exist).

- **Benchmark (optional or separate):**
  TestSail3DTrackerV2Benchmark::test_latency_target.

---

## One-line summary

- **Redundant:** `test_classifier_minimal` with the current pipeline classif test; consider removing or keeping as an optional sanity check.
- **Not redundant:** Pipeline (both configs), detector, track_video, tracker (unit + integration), crop_module_simple, video_tracking, sail_3d (and v2).
- **Optional by design:** dataset/test_crop_module (home_telltale), YOLO detector/track_video tests when `yolo-s.pt` is missing; benchmark latency test.
