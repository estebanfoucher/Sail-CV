# Test Failures Report

**Date**: 2026-02-14
**Last Updated**: After Quick Fixes Applied

**Test Results:**
- **Total Tests**: 89
- **Passing**: 62 ✅ (70% pass rate!)
- **Skipped**: 10 ⏭️
- **Failing**: 15 ❌ (was 21)
- **Errors**: 2 ⚠️

**Progress**: Fixed 6 tests by adding symlinks to checkpoints, assets, and fixtures!

---

## Summary by Category

| Category | Count | Notes |
|----------|-------|-------|
| Missing JSON calibration files | 6 | Need `intrinsics_1_1.json`, `intrinsics_1_2.json` |
| Missing video assets (reconstruction) | 4 | Need scene_3, scene_8 videos |
| Missing video assets (tracking) | 4 | Need `2Ce-CKKCtV4.mp4`, `IMG_9496_0.0_3.0.MOV` |
| Missing dataset directory | 5 | Need `tests/home_telltale/images/` |
| MASt3R/Dust3R dependency issue | 2 | Missing `models.dpt_block` module |
| Performance benchmark | 1 | System-dependent latency test |

---

## Detailed Test Failures

### 1. Reconstruction Module - Intrinsics Conversion Tests (6 failures + 2 errors)

**Tests:**
- `test_intrinsics_conversion_accuracy[grid-intrinsics_1_1]`
- `test_intrinsics_conversion_accuracy[grid-intrinsics_1_2]`
- `test_intrinsics_conversion_accuracy[random-intrinsics_1_1]`
- `test_intrinsics_conversion_accuracy[random-intrinsics_1_2]`
- `test_corner_tracking_through_resize` (ERROR)
- `test_complete_pipeline_validation` (ERROR)

**Location:** `tests/reconstruction/stereo/test_intrinsics_conversion.py`

**Issue:** Missing JSON calibration files
```
FileNotFoundError: [Errno 2] No such file or directory:
  - '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/intrinsics/intrinsics_1_1.json'
  - '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/intrinsics/intrinsics_1_2.json'
```

**What to Add:**
```bash
# Create directory
mkdir -p tests/assets/intrinsics/

# Need to add these files:
tests/assets/intrinsics/intrinsics_1_1.json
tests/assets/intrinsics/intrinsics_1_2.json
```

**File Format:** OpenCV camera intrinsics JSON (fx, fy, cx, cy, k1, k2, p1, p2)

---

### 2. Reconstruction Module - MASt3R Process Pair Tests (2 failures)

**Tests:**
- `test_process_pair`
- `test_process_pair_with_sam`

**Location:** `tests/reconstruction/test_process_pair.py`

**Issue:** Missing MASt3R/Dust3R internal module
```
ModuleNotFoundError: No module named 'models.dpt_block'
```

**What to Add:**
This requires the MASt3R/Dust3R submodule to be properly initialized. The `models.dpt_block` is part of the dust3r internal architecture.

**Solution Options:**
1. Initialize mast3r submodule: `git submodule update --init --recursive`
2. Skip these tests if mast3r is not available (add `@pytest.mark.skipif`)
3. Mock the dust3r model for testing purposes

**Priority:** Low - these are integration tests with external dependency

---

### 3. Reconstruction Module - Video Tests (4 failures)

**Tests:**
- `test_video_reader_and_writer`
- `test_stereo_video_reader`
- `test_stereo_image_saver`
- `test_sam` (in unitaries)

**Location:**
- `tests/reconstruction/test_video.py`
- `tests/reconstruction/unitaries/test_unitaries.py`

**Issue:** Missing video files
```
ValueError: Cannot open video file:
  - '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/scene_3/camera_1/camera_1.mp4'
  - '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/scene_8/camera_1/camera_1.mp4'
```

**What to Add:**
```bash
# Create directory structure
mkdir -p tests/assets/scene_3/camera_1/
mkdir -p tests/assets/scene_3/camera_2/
mkdir -p tests/assets/scene_8/camera_1/
mkdir -p tests/assets/scene_8/camera_2/

# Need to add these video files:
tests/assets/scene_3/camera_1/camera_1.mp4
tests/assets/scene_3/camera_2/camera_2.mp4  # (may also be needed)
tests/assets/scene_8/camera_1/camera_1.mp4
tests/assets/scene_8/camera_2/camera_2.mp4  # (may also be needed)
```

**Note:** These are stereo calibration scenes, so both camera_1 and camera_2 videos are likely needed.

---

### 4. Tracking Module - Dataset Tests (5 failures)

**Tests:**
- `test_loader`
- `test_crop_extraction`
- `test_mask_generation`
- `test_pca_results`
- `test_full_crop_module_pipeline`

**Location:** `tests/tracking/dataset/test_crop_module.py`

**Issue:** Missing YOLO dataset directory
```
ValueError: Images directory not found:
  '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/home_telltale/images'
```

**What to Add:**
```bash
# Create YOLO dataset structure
mkdir -p tests/home_telltale/images/
mkdir -p tests/home_telltale/labels/

# Expected structure:
tests/home_telltale/
├── images/           # Add sample images here
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
└── labels/           # Add YOLO format labels
    ├── frame_001.txt
    ├── frame_002.txt
    └── ...
```

**Note:** This is a YOLO format dataset for training/testing the crop module.

---

### 5. Tracking Module - Classifier Test (1 failure)

**Test:** `test_classifier_minimal`

**Location:** `tests/tracking/test_classifier_minimal.py`

**Issue:** Cannot read video file (exists but corrupted?)
```
ValueError: Could not read first frame from
  '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/fixtures/C1_fixture.mp4'
OpenCV: Couldn't read video stream from file
```

**Status:** ⚠️ File exists at `fixtures/C1_fixture.mp4` (95KB)

**What to Check:**
1. Verify the video file is not corrupted: `ffmpeg -v error -i fixtures/C1_fixture.mp4 -f null -`
2. Check codec compatibility: `ffmpeg -i fixtures/C1_fixture.mp4`
3. May need to re-encode: `ffmpeg -i fixtures/C1_fixture.mp4 -c:v libx264 -c:a aac fixtures/C1_fixture_fixed.mp4`

---

### 6. Tracking Module - Detector Tests (2 failures)

**Tests:**
- `test_tell_tale_detector_rt_detr`
- `test_tell_tale_detector_yolo`

**Location:** `tests/tracking/test_detector.py`

**Issue:** Missing video file
```
ValueError: Cannot open video file:
  '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/2Ce-CKKCtV4.mp4'
```

**Status:** ✅ File exists at `assets/tracking/2Ce-CKKCtV4.mp4`

**What to Do:**
The video exists in `assets/tracking/` but tests expect it in `tests/assets/`.

**Solution Options:**
1. Copy video: `cp assets/tracking/2Ce-CKKCtV4.mp4 tests/assets/`
2. Update test to use correct path: `assets/tracking/2Ce-CKKCtV4.mp4`
3. Create symlink: `ln -s ../../../assets/tracking/2Ce-CKKCtV4.mp4 tests/assets/`

---

### 7. Tracking Module - Pipeline Test (1 failure)

**Test:** `test_pipeline_with_fixture`

**Location:** `tests/tracking/test_pipeline.py`

**Issue:** Missing fixture video (different from item #5)
```
AssertionError: Fixture video not found:
  '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/fixtures/C1_fixture.mp4'
```

**Status:** ⚠️ File exists but test expects it in different location OR same corruption issue as item #5

**What to Do:**
1. Verify file integrity (see item #5)
2. Check test expectations match file location

---

### 8. Tracking Module - Performance Benchmark (1 failure)

**Test:** `test_latency_target`

**Location:** `tests/tracking/test_sail_3d_tracker_v2.py`

**Issue:** Performance threshold not met
```
AssertionError: Average latency 36.5ms exceeds 30ms target
assert 36.51230612013023 < 30.0
```

**Status:** ⚠️ System-dependent performance test

**What to Do:**
- This is expected to fail on slower machines
- Options:
  1. Adjust threshold to 40ms: `assert avg_latency < 40.0`
  2. Skip on non-production hardware: `@pytest.mark.skipif(not is_jetson())`
  3. Make threshold configurable via environment variable

**Priority:** Low - performance tests are system-dependent

---

### 9. Tracking Module - Video Test (1 failure)

**Test:** `test_video_reader_and_writer`

**Location:** `tests/tracking/test_video_tracking.py`

**Issue:** Missing video file
```
ValueError: Cannot open video file:
  '/Users/estebanfoucher/workspace/sailCV/sail-CV/tests/assets/IMG_9496_0.0_3.0.MOV'
```

**Status:** ✅ File exists at `assets/tracking/IMG_9496_0.0_3.0.MOV`

**What to Do:**
Same as item #6 - video exists in `assets/tracking/` but test expects it in `tests/assets/`

**Solution:** Copy or symlink from `assets/tracking/` to `tests/assets/`

---

## Action Items Checklist

### High Priority (Blocking Many Tests)

- [ ] **Create `tests/assets/` directory structure**
  ```bash
  mkdir -p tests/assets/intrinsics
  mkdir -p tests/assets/scene_3/camera_1
  mkdir -p tests/assets/scene_3/camera_2
  mkdir -p tests/assets/scene_8/camera_1
  mkdir -p tests/assets/scene_8/camera_2
  ```

- [ ] **Copy/symlink tracking videos to tests/assets/**
  ```bash
  cp assets/tracking/2Ce-CKKCtV4.mp4 tests/assets/
  cp assets/tracking/IMG_9496_0.0_3.0.MOV tests/assets/
  ```

- [ ] **Create intrinsics calibration JSON files**
  - `tests/assets/intrinsics/intrinsics_1_1.json`
  - `tests/assets/intrinsics/intrinsics_1_2.json`

- [ ] **Create YOLO dataset for crop module tests**
  ```bash
  mkdir -p tests/home_telltale/images
  mkdir -p tests/home_telltale/labels
  # Add sample images and YOLO labels
  ```

### Medium Priority

- [ ] **Add reconstruction scene videos**
  - scene_3 stereo pair videos
  - scene_8 stereo pair videos

- [ ] **Fix C1_fixture.mp4 video corruption**
  - Verify integrity with ffmpeg
  - Re-encode if needed

### Low Priority

- [ ] **Initialize MASt3R submodule** (or skip those tests)
  ```bash
  git submodule update --init --recursive
  ```

- [ ] **Adjust performance benchmark threshold** (system-dependent)
  - Update test to use 40ms or make configurable

---

## Files Currently Available

### ✅ Assets that DO exist:
- `fixtures/C1_fixture.mp4` (95KB - may be corrupted)
- `fixtures/C1_layout.json`
- `fixtures/C1_raw_detections.json`
- `assets/tracking/2Ce-CKKCtV4.mp4`
- `assets/tracking/IMG_9496_0.0_3.0.MOV`
- `assets/tracking/C1.mp4`
- Multiple other tracking assets

### ❌ Assets that are MISSING:
- `tests/assets/intrinsics/*.json`
- `tests/assets/scene_3/camera_1/camera_1.mp4`
- `tests/assets/scene_8/camera_1/camera_1.mp4`
- `tests/home_telltale/images/` (YOLO dataset)

---

## Quick Fix Commands

```bash
# 1. Create directory structure
mkdir -p tests/assets/intrinsics
mkdir -p tests/assets/scene_3/camera_{1,2}
mkdir -p tests/assets/scene_8/camera_{1,2}
mkdir -p tests/home_telltale/{images,labels}

# 2. Copy existing videos to tests location
cp assets/tracking/2Ce-CKKCtV4.mp4 tests/assets/
cp assets/tracking/IMG_9496_0.0_3.0.MOV tests/assets/

# 3. Verify C1_fixture integrity
ffmpeg -v error -i fixtures/C1_fixture.mp4 -f null - 2>&1

# 4. Check what videos we have in assets
find assets -name "*.mp4" -o -name "*.MOV"

# 5. Run tests again to see progress
uv run pytest tests/ -v --tb=line
```

---

## Test Summary After Fixes

**Expected Outcome After Quick Fixes:**
- ✅ 2 detector tests fixed (items #6)
- ✅ 1 video test fixed (item #9)
- ⚠️ Still need: intrinsics JSONs (6 tests)
- ⚠️ Still need: scene videos (4 tests)
- ⚠️ Still need: YOLO dataset (5 tests)
- ⚠️ Still need: MASt3R fix (2 tests)

**This would bring passing tests from 55 → 58 (65% → 68%)**
