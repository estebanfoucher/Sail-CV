# ByteTrack: Multi-Object Tracking Algorithm

## Overview

ByteTrack is the core tracking algorithm used in this project to maintain consistent identities of tell-tales across video frames. It was originally introduced in the paper ["ByteTrack: Multi-Object Tracking by Associating Every Detection Box"](https://arxiv.org/abs/2110.06864) (ECCV 2022).

This implementation is a **pure Python version** that uses `scipy.optimize.linear_sum_assignment` for optimal assignment, avoiding the need for the external LAP (Linear Assignment Problem) library.

---

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frame N Detections                          │
│                    [Detection₁, Detection₂, ...]                    │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 1: Filter by Confidence Threshold                 │
│                                                                     │
│   Keep only detections where confidence >= track_thresh (0.5)       │
│   → Creates list of candidate STrack objects                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 2: Clean Up Lost Tracks                           │
│                                                                     │
│   Remove tracks from lost_stracks if:                               │
│   (current_frame - last_seen_frame) > track_buffer (30 frames)      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 3: Associate Active Tracks                        │
│                                                                     │
│   Match tracked_stracks ↔ detections using IoU                      │
│   Threshold: match_thresh (0.8)                                     │
│                                                                     │
│   Uses Hungarian Algorithm (linear_sum_assignment)                  │
│   Cost matrix: C[i,j] = 1 - IoU(track_i, detection_j)              │
└───────────┬─────────────────────────────────────────┬───────────────┘
            │                                         │
            ▼                                         ▼
    ┌───────────────┐                        ┌───────────────┐
    │   Matched     │                        │  Unmatched    │
    │    Pairs      │                        │    Tracks     │
    │               │                        │               │
    │ Update track  │                        │ Move to       │
    │ with new      │                        │ lost_stracks  │
    │ detection     │                        │               │
    └───────────────┘                        └───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 4: Recover Lost Tracks                            │
│                                                                     │
│   Match lost_stracks ↔ unmatched_detections                         │
│   Threshold: match_thresh_lost (0.5) - more lenient!                │
│                                                                     │
│   This allows tracks that were temporarily occluded or missed       │
│   to be recovered with their original ID                            │
└───────────┬─────────────────────────────────────────┬───────────────┘
            │                                         │
            ▼                                         ▼
    ┌───────────────┐                        ┌───────────────┐
    │  Reactivated  │                        │   Remaining   │
    │    Tracks     │                        │  Unmatched    │
    │               │                        │  Detections   │
    │ Remove from   │                        │               │
    │ lost_stracks  │                        │               │
    └───────────────┘                        └───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 5: Initialize New Tracks                          │
│                                                                     │
│   For each remaining unmatched detection:                           │
│   - Create new STrack                                               │
│   - Assign unique track_id (next_id++)                              │
│   - Add to tracked_stracks                                          │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 6: Output Active Tracks                           │
│                                                                     │
│   Return List[Track] for all activated tracks in tracked_stracks    │
│   Each Track contains: detection, track_id, frame_id                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### STrack (Single Track)

An internal representation of a tracked object:

```python
class STrack:
    detection: Detection    # Current bounding box + confidence + class
    track_id: int | None    # Unique identifier (assigned on activation)
    is_activated: bool      # Whether track has been confirmed
    frame_id: int           # Last frame this track was updated
    start_frame: int        # First frame this track appeared
    end_frame: int          # Last frame before becoming lost
```

### Track Lists

The tracker maintains three separate lists:

| List | Description |
|------|-------------|
| `tracked_stracks` | Currently active and visible tracks |
| `lost_stracks` | Tracks that weren't matched in recent frames (within buffer) |
| `removed_stracks` | Tracks permanently removed (exceeded buffer time) |

---

## IoU (Intersection over Union)

The core similarity metric used for matching:

```
        ┌─────────────┐
        │   Box A     │
        │    ┌────────┼──────┐
        │    │ Inter- │      │
        └────┼─section│      │
             │        │Box B │
             └────────┴──────┘

IoU = Area(Intersection) / Area(Union)
    = Area(Intersection) / (Area(A) + Area(B) - Area(Intersection))
```

**Implementation:**

```python
def iou(track1, track2):
    # Extract coordinates
    x1, y1, x2, y2 = track1.bbox.xyxy
    x1g, y1g, x2g, y2g = track2.bbox.xyxy
    
    # Calculate intersection
    xa, ya = max(x1, x1g), max(y1, y1g)
    xb, yb = min(x2, x2g), min(y2, y2g)
    inter = max(0, xb - xa) * max(0, yb - ya)
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0
```

---

## Hungarian Algorithm (Optimal Assignment)

Given N tracks and M detections, we need to find the optimal one-to-one matching that minimizes total cost.

### Cost Matrix

```
                    Detection₀  Detection₁  Detection₂
                   ┌───────────┬───────────┬───────────┐
        Track₀     │  1 - IoU  │  1 - IoU  │  1 - IoU  │
                   ├───────────┼───────────┼───────────┤
        Track₁     │  1 - IoU  │  1 - IoU  │  1 - IoU  │
                   ├───────────┼───────────┼───────────┤
        Track₂     │  1 - IoU  │  1 - IoU  │  1 - IoU  │
                   └───────────┴───────────┴───────────┘
```

The cost is `1 - IoU` so that:
- **High IoU (good match)** → **Low cost**
- **Low IoU (poor match)** → **High cost**

### Assignment

```python
from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(cost_matrix)
# row_ind[i] is matched with col_ind[i]
```

After assignment, matches are **filtered** by the IoU threshold:
- If `IoU >= match_thresh`: Accept the match
- If `IoU < match_thresh`: Reject (both remain unmatched)

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_thresh` | 0.5 | Minimum detection confidence to consider for tracking |
| `track_buffer` | 30 | Number of frames to keep a lost track before removing |
| `match_thresh` | 0.8 | IoU threshold for matching active tracks (strict) |
| `match_thresh_lost` | 0.5 | IoU threshold for recovering lost tracks (lenient) |

### Why Two Thresholds?

- **Active matching (0.8)**: High threshold ensures we only match when very confident. Prevents ID switches.
- **Lost recovery (0.5)**: Lower threshold allows recovering tracks after occlusion or missed detections. The object may have moved more, so we're more lenient.

---

## Example Scenario

```
Frame 1:  [Det_A, Det_B]
          → Create Track_1 (Det_A), Track_2 (Det_B)

Frame 2:  [Det_C, Det_D]
          → Match: Track_1 ↔ Det_C (IoU=0.85 ✓)
          → Match: Track_2 ↔ Det_D (IoU=0.82 ✓)
          → Update positions

Frame 3:  [Det_E]  (Det_B disappeared - occlusion)
          → Match: Track_1 ↔ Det_E (IoU=0.88 ✓)
          → Track_2 unmatched → Move to lost_stracks

Frame 4:  [Det_F, Det_G]
          → Match: Track_1 ↔ Det_F (IoU=0.90 ✓)
          → Try lost recovery: Track_2 ↔ Det_G (IoU=0.55 ✓)
          → Track_2 reactivated with same ID!

Frame 5+: If Track_2 stays lost for 30 frames → permanently removed
```

---

## Comparison with Other Trackers

| Tracker | Motion Model | Appearance | Complexity |
|---------|--------------|------------|------------|
| **ByteTrack** | None (IoU only) | None | Low |
| SORT | Kalman Filter | None | Medium |
| DeepSORT | Kalman Filter | CNN embeddings | High |
| BoT-SORT | Kalman + Camera Motion | CNN embeddings | High |

ByteTrack's simplicity makes it:
- ✅ Fast and lightweight
- ✅ Easy to understand and debug
- ✅ Good for stable camera scenarios
- ⚠️ May struggle with fast motion or long occlusions

---

## File Location

The implementation can be found at:
```
src/tracker_utils/byte_tracker.py
```

The wrapper service class is at:
```
src/tracker.py
```
