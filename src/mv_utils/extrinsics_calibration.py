import cv2
import numpy as np
from typing import Tuple, Dict, Any
import yaml
import pupil_apriltags as apriltag
from typing import List
from loguru import logger



def calibrate_stereo_many(
    object_points_list: List[np.ndarray],
    image_points1_list: List[np.ndarray],
    image_points2_list: List[np.ndarray],
    camera_matrix1, dist_coeffs1,
    camera_matrix2, dist_coeffs2,
    image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Perform stereo calibration using multiple pairs of correspondences.
    
    Args:
        object_points_list: List of 3D points for each pair
        image_points1_list: List of 2D points from first camera for each pair
        image_points2_list: List of 2D points from second camera for each pair
        intrinsics1_path: Path to first camera intrinsics JSON
        intrinsics2_path: Path to second camera intrinsics JSON
        image_size: Image size as (width, height)
        
    Returns:
        Dictionary with stereo calibration results
    """
    total_points = int(sum(len(op) for op in object_points_list))
    print(f"Using {total_points} 3D-2D correspondences from {len(object_points_list)} pair(s)")

    flags = cv2.CALIB_FIX_INTRINSIC

    ret, camera_matrix1_cal, dist_coeffs1_cal, camera_matrix2_cal, dist_coeffs2_cal, \
    R, T, E, F = cv2.stereoCalibrate(
        object_points_list,
        image_points1_list,
        image_points2_list,
        camera_matrix1, dist_coeffs1,
        camera_matrix2, dist_coeffs2,
        image_size,
        flags=flags
    )

    results = {
        'success': True,
        'reprojection_error': float(ret),
        'num_correspondences': total_points,
        'num_pairs': len(object_points_list),
        'camera_matrix1': camera_matrix1_cal.tolist(),
        'camera_matrix2': camera_matrix2_cal.tolist(),
        'dist_coeffs1': dist_coeffs1_cal.tolist(),
        'dist_coeffs2': dist_coeffs2_cal.tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.tolist(),
        'essential_matrix': E.tolist(),
        'fundamental_matrix': F.tolist(),
        'image_size': image_size
    }
    return results


class StereoTagDetector:
    """Detect April tags in stereo images and extract correspondences."""
    
    def __init__(self, config_path: str = None):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.detector = self._create_detector()
        
        # Define 3D points of the April tag in tag coordinate system
        tag_size = self.config['apriltag']['tag_size_meters']
        half_size = tag_size / 2.0
        self.tag_3d_points = np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right  
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ], dtype=np.float32)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Config file {config_path} not found.")
            raise
    
    def _create_detector(self):
        """Create AprilTag detector with configured parameters."""
        tag_config = self.config.get('apriltag', {})
        
        return apriltag.Detector(
            families=tag_config.get('tag_family', 'tag36h11'),
            nthreads=4,
            quad_decimate=tag_config.get('decimation', 1.0),
            quad_sigma=tag_config.get('blur', 0.0),
            refine_edges=int(tag_config.get('refine_edges', True)),
            decode_sharpening=0.25,
            debug=0
        )
    
    def detect_tags(self, frame: np.ndarray) -> List:
        """Detect AprilTags in the given frame."""
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Detect tags
        detections = self.detector.detect(gray)
        
        # Filter detections by decision margin
        min_margin = self.config.get('apriltag', {}).get('min_decision_margin', 10.0)
        filtered_detections = [
            detection for detection in detections 
            if detection.decision_margin >= min_margin
        ]
        
        # Filter by specific tag ID if specified
        target_tag_id = self.config.get('apriltag', {}).get('target_tag_id')
        if target_tag_id is not None:
            filtered_detections = [
                detection for detection in filtered_detections
                if detection.tag_id == target_tag_id
            ]
        
        return filtered_detections
    
    def find_matching_tags(self, detections1: List, detections2: List) -> List[Tuple]:
        """Find matching tags between two camera views."""
        matches = []
        
        for det1 in detections1:
            for det2 in detections2:
                if det1.tag_id == det2.tag_id:
                    matches.append((det1, det2))
                    break
        
        return matches
    
    def extract_correspondences(self, matches: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D-2D correspondences and 3D points from matching tags."""
        points_3d = []
        points_2d_1 = []
        points_2d_2 = []
        
        for det1, det2 in matches:
            # Add 3D points (same for both cameras since they're in world coordinates)
            points_3d.append(self.tag_3d_points)
            
            # Add 2D points from both cameras
            points_2d_1.append(det1.corners.astype(np.float32))
            points_2d_2.append(det2.corners.astype(np.float32))
        
        if not points_3d:
            return None, None, None
        
        # Stack all points
        points_3d = np.vstack(points_3d)
        points_2d_1 = np.vstack(points_2d_1)
        points_2d_2 = np.vstack(points_2d_2)
        
        return points_3d, points_2d_1, points_2d_2
    
    def get_correspondences(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take two stereo images and return the extracted correspondences.
        
        Args:
            img1: First camera image as numpy array
            img2: Second camera image as numpy array
            
        Returns:
            Tuple of (points_3d, points_2d_1, points_2d_2) correspondences
        """
        if img1 is None or img2 is None:
            raise ValueError("Images cannot be None")
        
        # Detect tags in both images
        detections1 = self.detect_tags(img1)
        detections2 = self.detect_tags(img2)
        
        # Find matching tags
        matches = self.find_matching_tags(detections1, detections2)
        
        if len(matches) < 1:
            logger.debug("No matching tags found")
            return None, None, None
        
        # Extract correspondences
        points_3d, points_2d_1, points_2d_2 = self.extract_correspondences(matches)
        
        if points_3d is None:
            logger.debug("No correspondences found")
            return None, None, None
        
        return points_3d, points_2d_1, points_2d_2


def get_summary(results: Dict[str, Any]) -> str:
    """Get a summary of the calibration results."""
    summary = ""
    
    summary += "\n" + "="*50
    summary += "STEREO CALIBRATION RESULTS"
    summary += "="*50
    summary += f"\nSuccess: {results['success']}"
    summary += f"\nReprojection Error: {results['reprojection_error']:.6f}"
    summary += f"\nNumber of Correspondences: {results['num_correspondences']}"
    
    # Extract translation and rotation info
    T = np.array(results['translation_vector'])
    R = np.array(results['rotation_matrix'])
    
    # Convert rotation matrix to Euler angles
    euler_angles = _rotation_matrix_to_euler_angles(R)
    summary += f"\nRotation (roll, pitch, yaw): ({euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°)"
    
    summary += f"\nTranslation: ({T[0,0]:.6f}, {T[1,0]:.6f}, {T[2,0]:.6f}) m"
    # Calculate baseline distance
    baseline = np.linalg.norm(T)
    summary += f"\nComputed baseline distance: {baseline:.6f} m"

    return summary


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    # Extract Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees([roll, pitch, yaw])

