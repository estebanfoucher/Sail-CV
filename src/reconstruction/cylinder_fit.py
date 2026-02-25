import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------
# 0. Load parameters from JSON
# -------------------------------------------------------
params_json_path = "output/point_clouds/post_process_parameters.json"
with open(params_json_path) as f:
    params_dict = json.load(f)

print(f"\n=== LOADED PARAMETERS FROM {params_json_path} ===")
print(f"Found {len(params_dict)} point cloud configurations\n")


# -------------------------------------------------------
# 2. Statistical Outlier Removal for 3D points
# -------------------------------------------------------
def remove_statistical_outliers_3d(points, n_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from 3D points using nearest neighbor distance analysis.

    Points whose average distance to their k nearest neighbors is beyond mean + std_ratio*std
    are considered outliers.

    Args:
        points: 3D point array (N, 3)
        n_neighbors: Number of neighbors to consider (default: 20)
        std_ratio: Number of standard deviations for outlier threshold (default: 2.0)

    Returns:
        filtered_points: Points with outliers removed
        inlier_mask: Boolean mask indicating which points were kept
    """
    if len(points) < n_neighbors:
        # Not enough points for statistical analysis
        print(
            f"  Not enough points for 3D outlier removal ({len(points)} < {n_neighbors})"
        )
        return points, np.ones(len(points), dtype=bool)

    # Use actual n_neighbors or fewer if not enough points
    k = min(n_neighbors, len(points) - 1)

    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(points)
    distances, _indices = nbrs.kneighbors(points)

    # Average distance to k nearest neighbors (excluding self at index 0)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Compute statistics
    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)

    # Outlier threshold
    threshold = mean_dist + std_ratio * std_dist

    # Inliers are points with average distance below threshold
    inlier_mask = avg_distances < threshold
    filtered_points = points[inlier_mask]

    n_removed = len(points) - len(filtered_points)
    removal_pct = 100 * n_removed / len(points)
    print(
        f"  3D Statistical outlier removal: removed {n_removed}/{len(points)} points ({removal_pct:.1f}%)"
    )
    print(
        f"    Mean neighbor distance: {mean_dist:.4f}, Std: {std_dist:.4f}, Threshold: {threshold:.4f}"
    )

    return filtered_points, inlier_mask


# -------------------------------------------------------
# 3. Find best long axis using PCA
# -------------------------------------------------------
def find_principal_axis(points):
    """
    Find the principal axis (best long axis) of the point cloud using PCA.

    Returns:
        axis_direction: Normalized direction vector of the principal axis
        axis_point: Point on the axis (centroid of the point cloud)
    """
    # Center the points
    centroid = points.mean(axis=0)
    centered_points = points - centroid

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # The first principal component (largest eigenvalue) is the cylinder axis
    axis_direction = pca.components_[0]
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    return axis_direction, centroid


# -------------------------------------------------------
# 4. Slice point cloud along axis
# -------------------------------------------------------
def slice_along_axis(points, axis, axis_point, n_slices=4):
    """
    Divide the point cloud into N slices along the axis.

    Args:
        points: Point cloud array (N, 3)
        axis: Normalized axis direction vector
        axis_point: Point on the axis
        n_slices: Number of slices to create

    Returns:
        slices: List of point arrays, one for each slice
        slice_positions: List of axial positions (center of each slice)
    """
    # Compute axial coordinates (projection onto axis)
    diffs = points - axis_point
    axial_coords = diffs @ axis

    # Determine slice boundaries
    t_min = axial_coords.min()
    t_max = axial_coords.max()
    slice_width = (t_max - t_min) / n_slices

    slices = []
    slice_positions = []

    for i in range(n_slices):
        t_low = t_min + i * slice_width
        t_high = t_min + (i + 1) * slice_width

        # Extract points in this slice
        mask = (axial_coords >= t_low) & (axial_coords < t_high)
        if i == n_slices - 1:  # Include upper boundary for last slice
            mask = (axial_coords >= t_low) & (axial_coords <= t_high)

        slice_points = points[mask]

        if len(slice_points) > 0:
            slices.append(slice_points)
            slice_positions.append((t_low + t_high) / 2)

    return slices, slice_positions


# -------------------------------------------------------
# 5. Project points to orthogonal plane
# -------------------------------------------------------
def project_to_orthogonal_plane(points, axis, axis_point):
    """
    Project 3D points onto the plane perpendicular to the axis.

    Args:
        points: Point cloud array (N, 3)
        axis: Normalized axis direction vector
        axis_point: Point on the axis (origin for projection)

    Returns:
        points_2d: 2D coordinates in the orthogonal plane (N, 2)
    """
    # Compute vectors from axis point to each point
    diffs = points - axis_point

    # Project onto axis (axial component)
    axial_proj = np.outer(diffs @ axis, axis)

    # Remove axial component to get perpendicular component
    perp = diffs - axial_proj

    # Create an orthonormal basis for the plane
    # Use the first perpendicular vector as u
    if np.linalg.norm(perp[0]) > 1e-6:
        u = perp[0] / np.linalg.norm(perp[0])
    else:
        # If first point is on axis, use a default direction
        # Find a vector perpendicular to axis
        if abs(axis[0]) < 0.9:
            u = np.cross(axis, [1, 0, 0])
        else:
            u = np.cross(axis, [0, 1, 0])
        u = u / np.linalg.norm(u)

    # v is perpendicular to both axis and u
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)

    # Project points onto the plane basis
    points_2d = np.column_stack([perp @ u, perp @ v])

    return points_2d


# -------------------------------------------------------
# 6. Fit circle to 2D points - Helper Functions
# -------------------------------------------------------
def fit_circle_3points(points_2d):
    """
    Fit a circle through exactly 3 points using circumcircle formula.

    Args:
        points_2d: 2D point array (3, 2)

    Returns:
        center: Circle center (2D) or None if collinear
        radius: Circle radius or None if collinear
    """
    if len(points_2d) != 3:
        return None, None

    p1, p2, p3 = points_2d[0], points_2d[1], points_2d[2]

    # Check for collinearity using cross product
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if abs(cross) < 1e-10:
        return None, None  # Points are collinear

    # Compute circumcenter using perpendicular bisectors
    # Mid points
    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2

    # Slopes of perpendicular bisectors
    dx1, dy1 = p2 - p1
    dx2, dy2 = p3 - p2

    # Avoid division by zero
    if abs(dx1) < 1e-10:
        # First line is vertical, perpendicular is horizontal
        cx = mid1[0]
        if abs(dx2) < 1e-10:
            return None, None  # Both vertical
        slope2 = -dx2 / dy2
        cy = mid2[1] + slope2 * (cx - mid2[0])
    elif abs(dx2) < 1e-10:
        # Second line is vertical, perpendicular is horizontal
        cx = mid2[0]
        slope1 = -dx1 / dy1
        cy = mid1[1] + slope1 * (cx - mid1[0])
    else:
        slope1 = -dx1 / dy1
        slope2 = -dx2 / dy2

        if abs(slope1 - slope2) < 1e-10:
            return None, None  # Parallel perpendicular bisectors

        # Solve for intersection
        cx = (mid2[1] - mid1[1] + slope1 * mid1[0] - slope2 * mid2[0]) / (
            slope1 - slope2
        )
        cy = mid1[1] + slope1 * (cx - mid1[0])

    center = np.array([cx, cy])
    radius = np.linalg.norm(p1 - center)

    return center, radius


def fit_circle_least_squares(
    points_2d, initial_center=None, initial_radius=None, observability=1.0
):
    """
    Fit a circle to 2D points using least-squares.
    If initial guess is provided, uses geometric least-squares (iterative).
    Otherwise, uses algebraic least-squares (Pratt method, bias-minimizing).

    Args:
        points_2d: 2D point array (N, 2)
        initial_center: Initial guess for center (2D array, default: None)
        initial_radius: Initial guess for radius (float, default: None)
        observability: Arc observability score [0,1] - controls prior strength (default: 1.0)

    Returns:
        center: Circle center (2D)
        radius: Circle radius
        sigma: Uncertainty (standard deviation) of the radius estimate
    """
    if len(points_2d) < 3:
        return None, None, None

    # If initial guess is provided, use geometric least-squares (iterative optimization)
    if initial_center is not None and initial_radius is not None:
        try:
            # CURVATURE-DOMAIN REGULARIZATION for poorly observable arcs
            # Key insight: Working in R-space is asymmetric (noise pushes toward larger R).
            # Working in κ = 1/R space is symmetric (κ ± δ equally likely).
            # For short arcs (low observability), add strong curvature prior to prevent drift.

            use_curvature_prior = observability < 0.7

            if use_curvature_prior:
                # Prior strength inversely proportional to observability
                # Low observability → strong prior needed
                lambda_weight = (1.0 - observability) * len(points_2d) * 0.5
                kappa_prior = 1.0 / initial_radius  # Target curvature

                # Define residual function with curvature regularization
                def residuals_with_prior(params):
                    cx, cy, r = params
                    center = np.array([cx, cy])
                    # Geometric residuals (distance from points to circle)
                    geo_residuals = np.linalg.norm(points_2d - center, axis=1) - r
                    # Curvature regularization term (in 1/R space for symmetry)
                    # This prevents the optimizer from drifting toward large R (flat circle)
                    kappa_residual = np.sqrt(lambda_weight) * (1.0 / r - kappa_prior)
                    return np.concatenate([geo_residuals, [kappa_residual]])

                residuals_func = residuals_with_prior
            else:
                # High observability: standard geometric fit (no regularization)
                def residuals_geo_only(params):
                    cx, cy, r = params
                    center = np.array([cx, cy])
                    distances = np.linalg.norm(points_2d - center, axis=1)
                    return distances - r

                residuals_func = residuals_geo_only

            # Initial parameters
            x0 = [initial_center[0], initial_center[1], initial_radius]

            # Optimize using geometric least-squares with curvature regularization
            result = least_squares(residuals_func, x0, loss="soft_l1", f_scale=0.1)

            if result.success:
                cx, cy, radius = result.x
                center = np.array([cx, cy])

                # Calculate residuals for uncertainty
                distances_to_circle = np.abs(
                    np.linalg.norm(points_2d - center, axis=1) - radius
                )
                sigma = (
                    np.std(distances_to_circle, ddof=1)
                    if len(distances_to_circle) > 1
                    else 0.0
                )

                return center, radius, sigma
        except:
            # If geometric optimization fails, fall back to algebraic method
            pass

    # Pratt algebraic circle fitting - gradient-weighted, approximately unbiased
    # This method minimizes the GRADIENT-WEIGHTED algebraic distance, which removes
    # the systematic bias toward larger radii that plagues the Kasa method.
    #
    # Key insight: Kasa minimizes Σ(F(x,y))² where F is algebraic distance.
    # But ∂F/∂params depends on radius → bias. Pratt weights by |∇F|² to compensate.
    # This makes the estimator approximately unbiased for short arcs with radial noise.

    x = points_2d[:, 0]
    y = points_2d[:, 1]
    n = len(x)

    # Center the data (improves numerical stability)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u = x - x_mean
    v = y - y_mean

    # Compute moments (centered coordinates)
    Suu = np.sum(u**2) / n
    Svv = np.sum(v**2) / n
    Suv = np.sum(u * v) / n
    Suuu = np.sum(u**3) / n
    Svvv = np.sum(v**3) / n
    Suvv = np.sum(u * v**2) / n
    Svuu = np.sum(v * u**2) / n

    # Pratt method: solve the constrained eigensystem
    # Build the scatter matrix M and constraint matrix N
    # Pratt constraint: 4*A² + 4*B² - C = 1 (gradient normalization)

    try:
        # Algebraic distance vector for each point
        # For circle: F = x² + y² + A*x + B*y + C = 0
        # We want to find [A, B, C] that minimize Σ F² weighted by gradient

        # Pratt solution via Newton-Raphson or direct eigensystem
        # Simplified version: solve for center and radius directly

        # Build linear system for Pratt fit
        # Right-hand side
        np.sum(u) / n  # Should be 0 after centering, but keep for robustness
        np.sum(v) / n
        Suuv = Suuu + Suvv
        Svvu = Svvv + Svuu

        # Coefficient matrix (2x2 system)
        A_mat = np.array([[Suu, Suv], [Suv, Svv]])
        b_vec = np.array([Suuv, Svvu]) / 2.0

        # Solve for center offset (in centered coordinates)
        uc, vc = np.linalg.solve(A_mat, b_vec)

        # Transform back to original coordinates
        cx = uc + x_mean
        cy = vc + y_mean
        center = np.array([cx, cy])

        # Compute radius from center
        distances_from_center = np.linalg.norm(points_2d - center, axis=1)
        radius = np.mean(distances_from_center)  # Pratt uses mean distance for radius

        # Calculate residuals: distance from each point to the fitted circle
        distances_to_circle = np.abs(distances_from_center - radius)

        # Standard deviation of residuals gives uncertainty in radius
        # Use sample standard deviation (ddof=1 for unbiased estimate)
        if len(distances_to_circle) > 1:
            sigma = np.std(distances_to_circle, ddof=1)
        else:
            sigma = 0.0

        return center, radius, sigma

    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use centroid and mean distance
        center = points_2d.mean(axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        radius = np.mean(distances)
        # Estimate sigma from spread of distances
        sigma = np.std(distances, ddof=1) if len(distances) > 1 else 0.0
        return center, radius, sigma


# -------------------------------------------------------
# 6a-bis. Fit circle with fixed (imposed) radius
# -------------------------------------------------------
def fit_circle_fixed_radius(points_2d, fixed_radius, initial_center):
    """
    Fit a circle to 2D points with radius locked to fixed_radius.
    Only the center (cx, cy) is optimized (2 DOF).

    Args:
        points_2d: 2D point array (N, 2)
        fixed_radius: Imposed radius (not optimized)
        initial_center: Starting center from a previous free fit (2D array)

    Returns:
        center: Optimized circle center (2D)
        residual_std: Standard deviation of geometric residuals
    """
    if len(points_2d) < 3 or initial_center is None:
        return None, None

    def residuals(params):
        cx, cy = params
        center = np.array([cx, cy])
        return np.linalg.norm(points_2d - center, axis=1) - fixed_radius

    result = least_squares(residuals, initial_center, loss="soft_l1", f_scale=0.1)
    if result.success:
        center = np.array(result.x)
        res_std = (
            np.std(np.abs(residuals(result.x)), ddof=1) if len(points_2d) > 1 else 0.0
        )
        return center, res_std
    return None, None


# -------------------------------------------------------
# 6a-ter. Mean distance between two circles within bbox (for error in mm)
# -------------------------------------------------------
def mean_circle_error_in_bbox_mm(
    points_2d,
    center_free,
    radius_free,
    center_constr,
    radius_constr,
    n_sample=720,
    units_to_mm=1000.0,
):
    """
    Mean distance (in mm) between the free-fit and constrained-radius circles
    over the arc of the free-fit circle that lies inside the 2D bbox of the data.

    Samples points on the free-fit circle; keeps those inside the bbox of points_2d;
    for each, computes distance to the constrained circle; returns mean in mm.

    Assumes 2D/3D coordinates are in meters (units_to_mm=1000).
    """
    if points_2d is None or len(points_2d) < 2:
        return None
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    angles = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
    # Points on free-fit circle
    px = center_free[0] + radius_free * np.cos(angles)
    py = center_free[1] + radius_free * np.sin(angles)
    inside = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max)
    if not np.any(inside):
        return None
    pts = np.column_stack([px[inside], py[inside]])
    dist_to_constr_center = np.linalg.norm(pts - center_constr, axis=1)
    # Signed distance to constrained circle (positive outside)
    dist_to_constr_circle = np.abs(dist_to_constr_center - radius_constr)
    mean_dist = float(np.mean(dist_to_constr_circle))
    return mean_dist * units_to_mm


def sigma_circle_error_in_bbox_mm(
    points_2d,
    center_free,
    radius_free,
    center_constr,
    radius_constr,
    n_sample=720,
    units_to_mm=1000.0,
):
    """
    Std (sigma) of the distance between the free-fit and constrained circles
    over the arc of the free-fit circle that lies inside the 2D bbox of the data.
    Same sampling as mean_circle_error_in_bbox_mm; returns std in mm.
    """
    if points_2d is None or len(points_2d) < 2:
        return None
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    angles = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
    px = center_free[0] + radius_free * np.cos(angles)
    py = center_free[1] + radius_free * np.sin(angles)
    inside = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max)
    if not np.any(inside):
        return None
    pts = np.column_stack([px[inside], py[inside]])
    dist_to_constr_center = np.linalg.norm(pts - center_constr, axis=1)
    dist_to_constr_circle = np.abs(dist_to_constr_center - radius_constr)
    sigma_dist = float(np.std(dist_to_constr_circle))
    return sigma_dist * units_to_mm


# -------------------------------------------------------
# 6b. Statistical Outlier Removal for 2D points
# -------------------------------------------------------
def remove_statistical_outliers_2d(points_2d, n_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from 2D points using nearest neighbor distance analysis.

    Points whose average distance to their k nearest neighbors is beyond mean + std_ratio*std
    are considered outliers.

    Args:
        points_2d: 2D point array (N, 2)
        n_neighbors: Number of neighbors to consider (default: 20)
        std_ratio: Number of standard deviations for outlier threshold (default: 2.0)

    Returns:
        filtered_points: Points with outliers removed
        inlier_mask: Boolean mask indicating which points were kept
    """
    if len(points_2d) < n_neighbors:
        # Not enough points for statistical analysis
        return points_2d, np.ones(len(points_2d), dtype=bool)

    # Use actual n_neighbors or fewer if not enough points
    k = min(n_neighbors, len(points_2d) - 1)

    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(points_2d)
    distances, _indices = nbrs.kneighbors(points_2d)

    # Average distance to k nearest neighbors (excluding self at index 0)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Compute statistics
    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)

    # Outlier threshold
    threshold = mean_dist + std_ratio * std_dist

    # Inliers are points with average distance below threshold
    inlier_mask = avg_distances < threshold
    filtered_points = points_2d[inlier_mask]

    return filtered_points, inlier_mask


# -------------------------------------------------------
# 6c. Compute arc observability for adaptive priors
# -------------------------------------------------------
def compute_arc_observability(points_2d, expected_radius=None):
    """
    Compute observability metric for circle fitting on short arcs.

    Observability measures how well curvature can be determined from the data:
    - High observability (→1): Long arc, many points, low noise → curvature well-constrained
    - Low observability (→0): Short arc, few points, high noise → need strong priors

    Args:
        points_2d: 2D point array (N, 2)
        expected_radius: Expected radius for noise estimation (optional)

    Returns:
        observability: Score in [0, 1] indicating how observable curvature is
        angular_span_deg: Angular span of the arc in degrees
        noise_ratio: Ratio of radial noise to expected radius
    """
    if len(points_2d) < 3:
        return 0.0, 0.0, 1.0

    # Compute centroid and center the points
    centroid = np.mean(points_2d, axis=0)
    centered = points_2d - centroid

    # Compute angles of all points relative to centroid
    angles = np.arctan2(centered[:, 1], centered[:, 0])

    # Angular span: difference between max and min angles (handling wraparound)
    # Sort angles and find the largest gap
    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    gaps = np.append(gaps, 2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
    max_gap = np.max(gaps)
    angular_span_rad = 2 * np.pi - max_gap
    angular_span_deg = np.degrees(angular_span_rad)

    # Estimate noise from radial spread
    distances = np.linalg.norm(centered, axis=1)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    radial_noise = 1.4826 * mad  # Convert MAD to std

    # Noise ratio (high noise → low observability)
    if expected_radius is not None and expected_radius > 0:
        noise_ratio = radial_noise / expected_radius
    else:
        noise_ratio = radial_noise / max(median_dist, 0.1)

    # Observability components:
    # 1. Angular coverage: longer arcs → better curvature observability
    #    60° is a reasonable threshold for good curvature determination
    angular_factor = min(1.0, angular_span_deg / 60.0)

    # 2. Point density: more points → better statistics
    #    30 points is a reasonable threshold for good fit
    point_factor = min(1.0, len(points_2d) / 30.0)

    # 3. Signal-to-noise: low noise → better curvature resolution
    #    5% noise is reasonable threshold
    snr_factor = min(1.0, 0.05 / max(noise_ratio, 0.01))

    # Combined observability (geometric mean to avoid compensating low factors)
    observability = (angular_factor * point_factor * snr_factor) ** (1 / 3)

    return observability, angular_span_deg, noise_ratio


# -------------------------------------------------------
# 7. Fit circle to 2D points using RANSAC with geometric constraints
# -------------------------------------------------------
def fit_circle_2d(
    points_2d,
    max_iterations=1000,
    inlier_threshold=None,
    min_inlier_ratio=0.5,
    expected_radius=None,
    max_center_offset=None,
    prior_center=None,
    prior_radius=None,
):
    """
    Fit a circle to 2D points using RANSAC for robustness against outliers.
    Assumes points lie on a cylinder surface, so they should be at a consistent
    distance from the origin (0,0) in the 2D projection plane.

    Args:
        points_2d: 2D point array (N, 2)
        max_iterations: Maximum RANSAC iterations (default: 1000)
        inlier_threshold: Distance threshold for inliers. If None, computed adaptively (default: None)
        min_inlier_ratio: Minimum ratio of inliers required (default: 0.5)
        expected_radius: Expected radius for validation (default: None)
        max_center_offset: Maximum allowed distance of circle center from origin (default: None)
        prior_center: Center from previous slice as prior (2D array, default: None)
        prior_radius: Radius from previous slice as prior (float, default: None)

    Returns:
        center: Circle center (2D)
        radius: Circle radius
        sigma: Uncertainty (standard deviation) of the radius estimate
    """
    if len(points_2d) < 3:
        return None, None, None

    # SYMMETRIC RADIAL FILTER: Remove outliers symmetrically around median
    # This ELIMINATES BIAS: unlike outer-envelope filtering which systematically selects
    # points farther from center (causing radius inflation), symmetric filtering around
    # the median radial distance has no directional bias.
    distances_from_origin = np.linalg.norm(points_2d, axis=1)

    # Use median (robust to outliers) and MAD for scale estimation
    median_dist = np.median(distances_from_origin)
    mad = np.median(np.abs(distances_from_origin - median_dist))

    # Convert MAD to equivalent standard deviation (factor 1.4826 for normal distribution)
    radial_std = 1.4826 * mad

    # Keep points within ±2.5*sigma of median (symmetric filter, no bias)
    # 2.5*sigma captures ~98.8% of normally distributed noise while removing extreme outliers
    tolerance = 2.5 * radial_std

    # SYMMETRIC filtering: |distance - median| < tolerance
    valid_mask = np.abs(distances_from_origin - median_dist) < tolerance

    if np.sum(valid_mask) < 3:
        # If too aggressive (very low noise), relax to ±3*sigma
        tolerance = 3.0 * radial_std
        valid_mask = np.abs(distances_from_origin - median_dist) < tolerance

        if np.sum(valid_mask) < 3:
            print(
                f"  WARNING: Symmetric filtering too aggressive, only {np.sum(valid_mask)} points remain. Using all points."
            )
            filtered_points_2d = points_2d
        else:
            filtered_points_2d = points_2d[valid_mask]
            print(
                f"  Symmetric radial filter (relaxed): kept {np.sum(valid_mask)}/{len(points_2d)} points "
                f"within ±{tolerance:.3f}m of median {median_dist:.3f}m"
            )
    else:
        filtered_points_2d = points_2d[valid_mask]
        print(
            f"  Symmetric radial filter: kept {np.sum(valid_mask)}/{len(points_2d)} points "
            f"within ±{tolerance:.3f}m of median {median_dist:.3f}m (MAD={mad:.4f}m)"
        )

    # STATISTICAL OUTLIER REMOVAL: Remove isolated points using k-NN distance analysis
    if len(filtered_points_2d) >= 20:
        # Only apply if we have enough points for statistical analysis
        points_after_sor, _sor_mask = remove_statistical_outliers_2d(
            filtered_points_2d,
            n_neighbors=min(20, len(filtered_points_2d) - 1),
            std_ratio=2.0,  # Remove points >2 std deviations from mean neighbor distance
        )
        if len(points_after_sor) >= 3:
            n_removed_sor = len(filtered_points_2d) - len(points_after_sor)
            if n_removed_sor > 0:
                print(
                    f"  Statistical outlier removal: removed {n_removed_sor}/{len(filtered_points_2d)} isolated points"
                )
            points_for_fitting = points_after_sor
        else:
            print(
                "  WARNING: Statistical outlier removal too aggressive. Skipping SOR."
            )
            points_for_fitting = filtered_points_2d
    else:
        points_for_fitting = filtered_points_2d

    n_points = len(points_for_fitting)

    if n_points < 3:
        return None, None, None

    # COMPUTE ARC OBSERVABILITY for adaptive priors and scoring
    # This determines how well curvature can be determined from the data
    observability, angular_span_deg, noise_ratio = compute_arc_observability(
        points_for_fitting, expected_radius=expected_radius
    )

    # DIAGNOSTIC: Log observability metrics
    print(
        f"  Arc observability: {observability:.3f} (span={angular_span_deg:.1f}°, "
        f"n_pts={n_points}, noise_ratio={noise_ratio:.3f})"
    )
    if observability < 0.3:
        print(
            "  WARNING: Very low observability - curvature poorly constrained, using strong priors"
        )
    elif observability < 0.7:
        print(
            "  INFO: Moderate observability - applying adaptive curvature regularization"
        )

    # Adaptive threshold calculation using Median Absolute Deviation (MAD)
    if inlier_threshold is None:
        # Compute distances from origin (for cylinder, all points should be at similar distance)
        distances_from_origin_filtered = np.linalg.norm(points_for_fitting, axis=1)
        median_dist_filtered = np.median(distances_from_origin_filtered)
        mad = np.median(np.abs(distances_from_origin_filtered - median_dist_filtered))
        # Use MAD as scale estimator for radial variation
        # Scale by 1.4826 to make MAD consistent with standard deviation
        # Use tighter threshold (2*MAD instead of 3*MAD) for higher precision
        inlier_threshold = max(2.0 * 1.4826 * mad, 0.05)  # At least 0.05

    # Set maximum center offset if not provided
    if max_center_offset is None:
        # For a cylinder, the circle center should be near the origin
        # But for partial arcs, allow more flexibility
        if expected_radius is not None:
            max_center_offset = (
                1.5 * expected_radius
            )  # More permissive: 150% of expected radius
        else:
            max_center_offset = (
                2.0 * median_dist_filtered
            )  # Fallback: 200% of median distance

    # RANSAC loop
    best_inliers = 0
    best_inlier_mask = None
    best_score = -np.inf

    for _iteration in range(max_iterations):
        # 1. Randomly sample 3 points
        if n_points < 3:
            break
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_points = points_for_fitting[sample_indices]

        # 2. Fit circle to 3 points
        center, radius = fit_circle_3points(sample_points)
        if center is None or radius is None:
            continue

        # Geometric constraint 1: Circle center should be near origin (0,0)
        center_offset = np.linalg.norm(center)
        if center_offset > max_center_offset:
            continue  # Reject circles with center too far from origin

        # Geometric constraint 2: Radius should be reasonable
        if radius < 0.3 or radius > 15.0:  # Assuming meters, broad range
            continue

        # Geometric constraint 3: If expected radius is provided, prefer circles near it (but don't hard reject)
        # For partial arcs, the fitted radius can vary, so use soft scoring instead of hard constraint

        # 3. Compute distances from all points to fitted circle
        distances = np.abs(np.linalg.norm(points_for_fitting - center, axis=1) - radius)

        # 4. Count inliers
        inlier_mask = distances < inlier_threshold
        num_inliers = np.sum(inlier_mask)

        # 5. Compute score (prefer more inliers, smaller center offset, closer to expected radius)
        inlier_score = num_inliers
        center_penalty = min(center_offset / max_center_offset, 1.0)  # 0 to 1, capped

        # Radius penalty based on expected radius (R-space penalty)
        if expected_radius is not None:
            radius_penalty = min(abs(radius - expected_radius) / expected_radius, 1.0)
        else:
            radius_penalty = 0

        # CURVATURE PENALTY for short arcs (prevents bias toward flat solutions)
        # Key: For poorly observable arcs, penalize deviations in CURVATURE space (1/R)
        # not just radius space, because κ-space is symmetric under noise.
        curvature_penalty = 0
        if observability < 0.7 and expected_radius is not None:
            # Compute curvatures
            kappa_expected = 1.0 / expected_radius
            kappa_fitted = 1.0 / radius if radius > 0 else 0

            # Curvature error (symmetric in κ-space)
            curvature_error = abs(kappa_fitted - kappa_expected) / kappa_expected
            curvature_penalty = min(curvature_error, 1.0)

            # Weight by (1 - observability): strong penalty for poorly observable arcs
            # This prevents RANSAC from accepting flat circles on short arcs
            curvature_penalty *= 1.0 - observability

        # Combined score: maximize inliers (primary), minimize radius error (strong secondary),
        # curvature error (strong for short arcs), center offset (weak tertiary)
        score = (
            inlier_score
            - 0.5 * num_inliers * radius_penalty
            - 0.7
            * num_inliers
            * curvature_penalty  # Strong penalty for curvature deviation on short arcs
            - 0.1 * num_inliers * center_penalty
        )

        # 6. Update best model if this is better
        if score > best_score:
            best_score = score
            best_inliers = num_inliers
            best_inlier_mask = inlier_mask

        # 7. Early termination if we have very good consensus
        if num_inliers / n_points > 0.95 and center_offset < max_center_offset * 0.5:
            break

    # Check if we found enough inliers
    if best_inliers < max(3, int(min_inlier_ratio * n_points)):
        print(
            f"  WARNING: RANSAC found only {best_inliers}/{n_points} inliers "
            f"({100 * best_inliers / n_points:.1f}%). Falling back to constrained least-squares."
        )
        # Fallback to least-squares on filtered points, using prior as starting point if available
        center, radius, sigma = fit_circle_least_squares(
            points_for_fitting,
            initial_center=prior_center,
            initial_radius=prior_radius,
            observability=observability,
        )
        # Validate fallback result
        if (
            (center is not None)
            and (expected_radius is not None)
            and abs(radius - expected_radius) / expected_radius > 0.5
        ):
            print(
                f"  WARNING: Fitted radius {radius:.3f} is far from expected {expected_radius:.3f}"
            )
        return center, radius, sigma

    # Refine circle using least-squares on inliers, using prior as starting point if available
    inlier_points = points_for_fitting[best_inlier_mask]
    center, radius, sigma = fit_circle_least_squares(
        inlier_points,
        initial_center=prior_center,
        initial_radius=prior_radius,
        observability=observability,
    )

    # POST-RANSAC ITERATIVE REFINEMENT: Remove outliers from inliers for higher accuracy
    # Iteratively remove points with high residuals (>2 sigma) and refit
    max_refinement_iterations = 3
    refinement_sigma_threshold = 2.0  # Remove points >2 sigma from fitted circle

    for _refine_iter in range(max_refinement_iterations):
        if center is None or radius is None or len(inlier_points) < 10:
            break

        # Compute residuals (distance from each point to fitted circle)
        distances_to_circle = np.abs(
            np.linalg.norm(inlier_points - center, axis=1) - radius
        )

        # Identify inliers within refinement_sigma_threshold * sigma
        refine_threshold = refinement_sigma_threshold * sigma
        refined_mask = distances_to_circle < refine_threshold
        n_refined = np.sum(refined_mask)

        # Only refine if we're removing outliers and keeping enough points
        if n_refined < len(inlier_points) * 0.8 or n_refined < 10:
            # Don't remove too many points
            break

        if n_refined == len(inlier_points):
            # No more outliers to remove
            break

        # Refit with refined inliers, using current fit as starting point
        inlier_points = inlier_points[refined_mask]
        center_new, radius_new, sigma_new = fit_circle_least_squares(
            inlier_points,
            initial_center=center,
            initial_radius=radius,
            observability=observability,
        )

        # Check if refinement improved the fit
        if center_new is None or radius_new is None:
            break

        # Update if improvement
        improvement = abs(sigma_new - sigma)
        center, radius, sigma = center_new, radius_new, sigma_new

        if improvement < sigma * 0.05:  # Less than 5% improvement
            break

    # Print diagnostic information
    center_offset_final = np.linalg.norm(center) if center is not None else 0
    print(
        f"  RANSAC: {best_inliers}/{n_points} inliers ({100 * best_inliers / n_points:.1f}%)"
    )
    print(f"  Post-refinement: {len(inlier_points)} final inliers, sigma={sigma:.4f}")
    print(
        f"  Inlier threshold: {inlier_threshold:.4f}, Center offset: {center_offset_final:.4f}"
    )

    # Diagnostic: Prior usage and strength
    if prior_center is not None and prior_radius is not None:
        # Compute prior strength based on observability
        if observability < 0.7:
            lambda_weight = (1.0 - observability) * len(inlier_points) * 0.5
            kappa_prior = 1.0 / prior_radius
            print(
                f"  Prior: radius={prior_radius:.3f}m, κ={kappa_prior:.4f}/m, "
                f"strength λ={lambda_weight:.1f} (obs={observability:.2f})"
            )
        else:
            print(
                f"  Prior: radius={prior_radius:.3f}m (not applied, high observability)"
            )

    # Diagnostic: Fitted radius and curvature
    if expected_radius is not None and radius is not None:
        kappa_fitted = 1.0 / radius
        kappa_expected = 1.0 / expected_radius
        radius_error_pct = 100 * abs(radius - expected_radius) / expected_radius
        kappa_error_pct = 100 * abs(kappa_fitted - kappa_expected) / kappa_expected
        print(
            f"  Fitted: radius={radius:.3f}m (error={radius_error_pct:.2f}%), "
            f"κ={kappa_fitted:.4f}/m (error={kappa_error_pct:.2f}%)"
        )

    return center, radius, sigma


# -------------------------------------------------------
# 8. Compute camber from fitted circle and filter radius
# -------------------------------------------------------
def compute_camber(center_2d, radius, filter_radius):
    """
    Compute camber from a fitted circle and filter radius.

    The camber is defined as: sagitta / chord_length = h/c
    where h (sagitta) and c (chord) are defined by:
    - filter_radius**2 = h**2 + (c/2)**2
    - fitted_radius**2 = (fitted_radius - h)**2 + (c/2)**2

    Args:
        center_2d: Center of fitted circle (2D) - not used in calculation but kept for compatibility
        radius: Radius of fitted circle (fitted_radius)
        filter_radius: Radius of filter circle (centered at origin)

    Returns:
        camber: Camber value (h/c)
        chord_length: Length of the chord (c)
        sagitta: Sagitta (h)
        intersection_points: None (not computed, kept for compatibility)
    """
    # From the equations:
    # filter_radius**2 = h**2 + (c/2)**2  ... (1)
    # fitted_radius**2 = (fitted_radius - h)**2 + (c/2)**2  ... (2)

    # Expanding equation (2):
    # fitted_radius**2 = fitted_radius**2 - 2*fitted_radius*h + h**2 + (c/2)**2
    # 0 = -2*fitted_radius*h + h**2 + (c/2)**2
    # 2*fitted_radius*h = h**2 + (c/2)**2

    # Substituting from equation (1):
    # 2*fitted_radius*h = h**2 + filter_radius**2 - h**2
    # 2*fitted_radius*h = filter_radius**2
    # h = filter_radius**2 / (2*fitted_radius)

    if radius <= 0 or filter_radius <= 0:
        return None, None, None, None

    # Calculate sagitta (h)
    sagitta = filter_radius**2 / (2 * radius)

    # Check if sagitta is valid (must be <= filter_radius)
    if sagitta > filter_radius or sagitta < 0:
        return None, None, None, None

    # From equation (1): filter_radius**2 = h**2 + (c/2)**2
    # (c/2)**2 = filter_radius**2 - h**2
    # c = 2 * sqrt(filter_radius**2 - h**2)

    chord_half_squared = filter_radius**2 - sagitta**2
    if chord_half_squared < 0:
        return None, None, None, None

    chord_length = 2 * np.sqrt(chord_half_squared)

    if chord_length < 1e-10:
        return None, None, None, None

    # Camber = h/c
    camber = sagitta / chord_length

    return camber, chord_length, sagitta, None


# -------------------------------------------------------
# 9. Filter points by distance from axis
# -------------------------------------------------------
def filter_points_by_axis_distance(points, axis, axis_point, filter_radius=1.0):
    """
    Filter out points that are more than filter_radius away from the axis.

    Args:
        points: Point cloud array (N, 3)
        axis: Normalized axis direction vector
        axis_point: Point on the axis
        filter_radius: Maximum distance from axis (default: 1.0 meters)

    Returns:
        filtered_points: Filtered point cloud array (M, 3) where M <= N
        mask: Boolean mask indicating which points were kept
    """
    # Compute vectors from axis point to each point
    diffs = points - axis_point

    # Project onto axis (axial component)
    axial_proj = np.outer(diffs @ axis, axis)

    # Remove axial component to get perpendicular component
    perp = diffs - axial_proj

    # Compute perpendicular distance from axis
    distances = np.linalg.norm(perp, axis=1)

    # Create mask for points within filter_radius
    mask = distances <= filter_radius

    filtered_points = points[mask]

    return filtered_points, mask


# -------------------------------------------------------
# 10. Main analysis: Loop over point clouds
# -------------------------------------------------------
for name, params in params_dict.items():
    print(f"\n{'=' * 60}")
    print(f"=== PROCESSING: {name} ===")
    print(f"{'=' * 60}\n")

    # Extract parameters
    n_slices = params["n_slice"]
    filter_radius = params["filter_radius"]
    measured_radius = params["measured_radius"]

    print(
        f"Parameters: n_slice={n_slices}, filter_radius={filter_radius}, measured_radius={measured_radius}"
    )

    # Load point cloud
    ply_path = f"output/point_clouds/{name}.ply"
    try:
        cloud = trimesh.load(ply_path)
        if not hasattr(cloud, "vertices"):
            print(f"ERROR: {ply_path} does not contain vertices. Skipping.")
            continue
        points = np.asarray(cloud.vertices)
        print(f"Loaded {points.shape[0]} points from {ply_path}")
    except Exception as e:
        print(f"ERROR: Failed to load {ply_path}: {e}. Skipping.")
        continue

    # Apply 3D statistical outlier removal before finding principal axis
    # This removes outliers that could bias the PCA computation
    print("\n--- Step 1: Remove 3D outliers before principal axis detection ---")
    points_cleaned, inlier_mask_3d = remove_statistical_outliers_3d(
        points, n_neighbors=20, std_ratio=2.0
    )

    # Find principal axis using cleaned points
    print("\n--- Step 2: Find principal axis using cleaned points ---")
    axis_direction, axis_point = find_principal_axis(points_cleaned)
    print(f"Axis direction (normalized): {axis_direction}")
    print(f"Axis point (centroid): {axis_point}")

    # Filter points by distance from axis (before slicing)
    # Use original points here, not cleaned points, for consistency with later visualizations
    print("\n--- Step 3: Filter points by distance from axis ---")
    points_filtered, filter_mask = filter_points_by_axis_distance(
        points, axis_direction, axis_point, filter_radius=filter_radius
    )
    print(
        f"Filtered points: {len(points)} -> {len(points_filtered)} "
        f"(removed {len(points) - len(points_filtered)} points "
        f"beyond {filter_radius}m from axis)"
    )

    # Slice along axis
    print("\n--- Step 4: Slice point cloud along the axis ---")
    slices, slice_positions = slice_along_axis(
        points_filtered, axis_direction, axis_point, n_slices=n_slices
    )

    print(f"Created {len(slices)} slices along the axis")
    for i, (slice_pts, pos) in enumerate(zip(slices, slice_positions, strict=False)):
        print(f"  Slice {i + 1}: {len(slice_pts)} points, axial position = {pos:.3f}")

    # Process each slice
    print("\n--- Step 5: Fit circles to each slice and compute camber ---")
    radii = []
    radius_sigmas = []  # Store uncertainty (sigma) for each radius
    cambers = []  # Store camber values
    camber_sigmas = []  # Store uncertainty (sigma) for each camber
    slice_centers = []
    valid_slices = []
    circle_centers_2d = []  # Store circle centers for visualization
    constrained_centers_2d = []  # Store constrained-radius circle centers
    mean_circle_error_mm = []  # Mean distance free vs constrained circle in bbox (mm)
    sigma_point_to_green_mm = []  # Std of point-to-green-circle distances per slice (mm)
    sigma_red_green_bbox_mm = []  # Std of red-green circle distance in bbox per slice (mm)
    valid_slice_points = []  # Store original slice points for all valid slices

    # Track previous slice results for use as priors
    prev_center_2d = None
    prev_radius = None

    for i, (slice_pts, pos) in enumerate(zip(slices, slice_positions, strict=False)):
        if len(slice_pts) < 3:
            print(f"\nSlice {i + 1}: Skipped (too few points: {len(slice_pts)})")
            continue

        # Project to orthogonal plane
        points_2d = project_to_orthogonal_plane(slice_pts, axis_direction, axis_point)

        # Fit circle with geometric constraints (use measured radius and filter radius as guidance)
        # Use previous slice results as starting point if available
        center_2d, radius, sigma = fit_circle_2d(
            points_2d,
            expected_radius=measured_radius,
            max_center_offset=filter_radius,
            prior_center=prev_center_2d,
            prior_radius=prev_radius,
        )

        if radius is None or radius <= 0:
            print(f"\nSlice {i + 1}: Failed to fit circle")
            continue

        # Compute camber
        camber, chord_length, sagitta, intersection_points = compute_camber(
            center_2d, radius, filter_radius
        )

        if camber is None:
            print(
                f"\nSlice {i + 1}: Failed to compute camber (circles don't intersect properly)"
            )
            continue

        # Propagate uncertainty from radius to camber
        # Use finite difference to estimate sensitivity
        camber_plus = compute_camber(center_2d, radius + sigma, filter_radius)[0]
        camber_minus = compute_camber(center_2d, radius - sigma, filter_radius)[0]

        if camber_plus is not None and camber_minus is not None:
            # Approximate derivative: df/dr ≈ (f(r+sigma) - f(r-sigma)) / (2*sigma)
            camber_sigma = abs(camber_plus - camber_minus) / 2.0
        else:
            # Fallback: use a simple approximation
            camber_sigma = abs(camber) * (sigma / radius) if radius > 0 else 0.0

        # Constrained-radius fit: optimize center only with R = measured_radius
        constr_center, constr_std = fit_circle_fixed_radius(
            points_2d, measured_radius, initial_center=center_2d
        )

        radii.append(radius)
        radius_sigmas.append(sigma)
        cambers.append(camber)
        camber_sigmas.append(camber_sigma)
        slice_centers.append(pos)
        circle_centers_2d.append(center_2d)
        constrained_centers_2d.append(constr_center)
        err_mm = mean_circle_error_in_bbox_mm(
            points_2d, center_2d, radius, constr_center, measured_radius
        )
        mean_circle_error_mm.append(err_mm if err_mm is not None else np.nan)
        # Sigma of point-to-green-circle distance (green = ground truth, R = measured_radius)
        dist_to_green = np.abs(
            np.linalg.norm(points_2d - constr_center, axis=1) - measured_radius
        )
        sigma_green_mm = float(np.std(dist_to_green)) * 1000.0  # m -> mm
        sigma_point_to_green_mm.append(sigma_green_mm)
        # Sigma of red-green circle distance in bbox (along arc inside bbox only)
        sigma_rg = sigma_circle_error_in_bbox_mm(
            points_2d, center_2d, radius, constr_center, measured_radius
        )
        sigma_red_green_bbox_mm.append(sigma_rg if sigma_rg is not None else np.nan)
        valid_slices.append(i + 1)
        valid_slice_points.append(slice_pts)  # Store for visualization

        # Update priors for next slice
        prev_center_2d = center_2d
        prev_radius = radius

        print(f"\nSlice {i + 1} (axial position = {pos:.3f}):")
        print(f"  Circle center (2D): {center_2d}")
        print(f"  Radius: {radius:.6f} ± {2 * sigma:.6f} (2*sigma)")
        print(f"  Camber: {camber:.6f} ± {2 * camber_sigma:.6f} (2*sigma)")
        if err_mm is not None and not np.isnan(err_mm):
            print(f"  Mean circle error (free vs constrained in bbox): {err_mm:.3f} mm")

    # Summary: mean circle error in mm (over valid slices)
    if mean_circle_error_mm:
        valid_err = np.array(mean_circle_error_mm)
        valid_err = valid_err[~(np.isnan(valid_err))]
        if len(valid_err) > 0:
            print(
                f"\n--- Mean circle error (free vs constrained, in bbox): "
                f"mean = {np.mean(valid_err):.3f} mm, max = {np.max(valid_err):.3f} mm over {len(valid_err)} slices ---"
            )

    # -------------------------------------------------------
    # 7. Visualizations
    # -------------------------------------------------------

    # Create plots directory
    SUBFOLDER_NAME = f"plots_{name}"
    plots_dir = Path(SUBFOLDER_NAME)
    plots_dir.mkdir(exist_ok=True)
    print("\n=== GENERATING VISUALIZATIONS ===")
    print(f"Saving all figures to: {plots_dir.absolute()}")
    fig = plt.figure(figsize=(12, 6))

    # Plot 1: 3D point cloud with axis (showing filtered points)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # Show original points in light gray (if not too many)
    if len(points) <= 20000:
        ax1.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c="lightgray",
            s=0.5,
            alpha=0.1,
            label="All Points",
        )

    # Sample filtered points for faster rendering (if too many)
    if len(points_filtered) > 10000:
        sample_indices = np.random.choice(len(points_filtered), 10000, replace=False)
        points_vis = points_filtered[sample_indices]
    else:
        points_vis = points_filtered

    ax1.scatter(
        points_vis[:, 0],
        points_vis[:, 1],
        points_vis[:, 2],
        c=points_vis[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.5,
        label="Filtered Points",
    )

    # Draw axis line
    axial_coords_all = (points_filtered - axis_point) @ axis_direction
    t_min_vis = axial_coords_all.min()
    t_max_vis = axial_coords_all.max()
    axis_line_points = axis_point + np.outer([t_min_vis, t_max_vis], axis_direction)
    ax1.plot(
        axis_line_points[:, 0],
        axis_line_points[:, 1],
        axis_line_points[:, 2],
        "r-",
        linewidth=3,
        label="Detected Axis",
    )
    ax1.scatter(*axis_point, color="red", s=100, marker="o", label="Axis Point")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(
        f"Step 1: Point Cloud with Detected Axis\n(Filtered: {filter_radius}m radius)"
    )
    ax1.legend()

    # Plot 2: Sliced point cloud (colored by slice)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(slices)))

    # Compute slice boundaries for visualization
    axial_coords_all = (points_filtered - axis_point) @ axis_direction
    t_min_all = axial_coords_all.min()
    t_max_all = axial_coords_all.max()
    slice_width = (t_max_all - t_min_all) / n_slices
    slice_boundaries = [t_min_all + i * slice_width for i in range(n_slices + 1)]

    # Draw slice boundary planes (as lines perpendicular to axis)
    for i, boundary_t in enumerate(slice_boundaries):
        boundary_point = axis_point + boundary_t * axis_direction

        # Create a small circle in the plane perpendicular to axis to visualize boundary
        # Use two perpendicular vectors in the plane
        if np.linalg.norm(axis_direction - np.array([1, 0, 0])) > 0.1:
            u_vec = np.cross(axis_direction, np.array([1, 0, 0]))
        else:
            u_vec = np.cross(axis_direction, np.array([0, 1, 0]))
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(axis_direction, u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)

        # Draw a small circle at the boundary (radius = filter_radius for visibility)
        theta = np.linspace(0, 2 * np.pi, 20)
        circle_radius = filter_radius * 0.8  # Slightly smaller than filter radius
        boundary_circle = boundary_point + circle_radius * (
            np.outer(np.cos(theta), u_vec) + np.outer(np.sin(theta), v_vec)
        )
        ax2.plot(
            boundary_circle[:, 0],
            boundary_circle[:, 1],
            boundary_circle[:, 2],
            "k--",
            linewidth=1,
            alpha=0.4,
            label="Slice Boundary" if i == 0 else "",
        )

    for i, (slice_pts, color) in enumerate(zip(slices, colors, strict=False)):
        if len(slice_pts) > 5000:
            sample_idx = np.random.choice(len(slice_pts), 5000, replace=False)
            slice_vis = slice_pts[sample_idx]
        else:
            slice_vis = slice_pts
        ax2.scatter(
            slice_vis[:, 0],
            slice_vis[:, 1],
            slice_vis[:, 2],
            c=[color],
            s=1,
            alpha=0.5,
            label=f"Slice {i + 1}",
        )

    # Draw axis
    ax2.plot(
        axis_line_points[:, 0],
        axis_line_points[:, 1],
        axis_line_points[:, 2],
        "k-",
        linewidth=2,
        alpha=0.5,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Step 2: Sliced Point Cloud\n(After filtering, boundaries shown)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    output_path = plots_dir / "01_cylinder_analysis_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Visualization 2: All successful projected circle fittings (zoomed to data)
    if len(valid_slices) > 0:
        n_cols = min(4, len(valid_slices))
        n_rows = int(np.ceil(len(valid_slices) / n_cols))
        fig_all = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for idx, (slice_num, slice_pts, pos) in enumerate(
            zip(valid_slices, valid_slice_points, slice_centers, strict=False)
        ):
            ax = fig_all.add_subplot(n_rows, n_cols, idx + 1)

            # Project to 2D
            points_2d = project_to_orthogonal_plane(
                slice_pts, axis_direction, axis_point
            )

            # Get fitted circle parameters for this slice
            center_2d = circle_centers_2d[idx]
            radius_val = radii[idx]
            constr_center = constrained_centers_2d[idx]

            # Plot projected points
            ax.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                s=5,
                alpha=0.6,
                c="blue",
                label="Points",
            )

            # Plot free-fit circle (red)
            theta = np.linspace(0, 2 * np.pi, 360)
            free_x = center_2d[0] + radius_val * np.cos(theta)
            free_y = center_2d[1] + radius_val * np.sin(theta)
            ax.plot(
                free_x,
                free_y,
                color="red",
                linewidth=1.5,
                label=f"Free R={radius_val:.3f}",
            )

            # Plot constrained circle (green, R = measured_radius)
            if constr_center is not None:
                constr_x = constr_center[0] + measured_radius * np.cos(theta)
                constr_y = constr_center[1] + measured_radius * np.sin(theta)
                ax.plot(
                    constr_x,
                    constr_y,
                    color="green",
                    linewidth=1.5,
                    label=f"Constr R={measured_radius:.3f}",
                )

            # Zoom to data region: bbox of points with margin
            margin_frac = 0.15
            x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
            y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
            dx = x_max - x_min
            dy = y_max - y_min
            pad = max(dx, dy) * margin_frac
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_xlabel("u", fontsize=10)
            ax.set_ylabel("v", fontsize=10)
            ax.set_title(
                f"Slice {slice_num}\nPos: {pos:.3f}, R_free: {radius_val:.3f}",
                fontsize=9,
            )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="best")

        plt.tight_layout()
        output_path = plots_dir / "02_cylinder_all_projections.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig_all)
        print(f"Saved: {output_path} ({len(valid_slices)} successful slices)")

    # Visualization 3: Radius plot
    if len(radii) > 0:
        fig2 = plt.figure(figsize=(8, 5))

        # Radius vs position with error bars (±2*sigma)
        ax2 = fig2.add_subplot(1, 1, 1)
        # Convert sigmas to 2-sigma error bars
        error_bars = [2 * sigma for sigma in radius_sigmas]
        ax2.errorbar(
            slice_centers,
            radii,
            yerr=error_bars,
            fmt="-",
            linewidth=2,
            capsize=5,
            capthick=2,
            color="orange",
            ecolor="black",
            alpha=0.7,
            label=r"Fitted radius (±2$\sigma$)",
        )
        ax2.axhline(
            measured_radius,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Measured radius",
        )
        ax2.set_xlabel("Position along axis", fontsize=12)
        ax2.set_ylabel("Radius (m)", fontsize=12)
        ax2.set_ylim(2, 6)
        ax2.set_title("Radius vs position along axis", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        output_path = plots_dir / "03_cylinder_curvature_radius.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {output_path}")

        print("\n=== SUMMARY ===")
        print(f"Mean radius: {np.mean(radii):.6f}")
        print(f"Std radius: {np.std(radii):.6f}")
        print(
            f"Mean 2*sigma uncertainty: {np.mean([2 * s for s in radius_sigmas]):.6f}"
        )
        print(f"Max 2*sigma uncertainty: {max([2 * s for s in radius_sigmas]):.6f}")
        print(f"Measured radius: {measured_radius:.6f}")
    else:
        print("\nNo valid slices found for radius analysis")

    # Visualization 4: Camber plot
    if len(cambers) > 0:
        # Filter camber values to be in [0, 0.3]
        valid_mask = np.array([0 <= c <= 0.15 for c in cambers])
        filtered_cambers = [c for c, m in zip(cambers, valid_mask, strict=False) if m]
        filtered_camber_sigmas = [
            s for s, m in zip(camber_sigmas, valid_mask, strict=False) if m
        ]
        filtered_slice_centers = [
            sc for sc, m in zip(slice_centers, valid_mask, strict=False) if m
        ]

        if len(filtered_cambers) == 0:
            print("\nNo valid camber values in range [0, 0.3] for plotting")
        else:
            # Compute measured camber from measured radius
            # Use a representative center (average of all centers) for measured camber calculation
            if len(circle_centers_2d) > 0:
                avg_center = np.mean(circle_centers_2d, axis=0)
                measured_camber, _, _, _ = compute_camber(
                    avg_center, measured_radius, filter_radius
                )
            else:
                measured_camber = None

            fig3 = plt.figure(figsize=(8, 5))

            # Camber vs position with error bars (±2*sigma)
            ax3 = fig3.add_subplot(1, 1, 1)
            # Convert sigmas to 2-sigma error bars
            camber_error_bars = [2 * sigma for sigma in filtered_camber_sigmas]
            ax3.errorbar(
                filtered_slice_centers,
                filtered_cambers,
                yerr=camber_error_bars,
                fmt="-",
                linewidth=2,
                capsize=5,
                capthick=2,
                color="blue",
                ecolor="black",
                alpha=0.7,
                label="Fitted Camber (±2*sigma)",
            )
            if measured_camber is not None and 0 <= measured_camber <= 0.3:
                ax3.axhline(
                    measured_camber,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Measured Camber",
                )
            ax3.set_xlabel("Position along axis", fontsize=12)
            ax3.set_ylabel("Camber", fontsize=12)
            ax3.set_title(
                "Camber vs Position Along Axis", fontsize=14, fontweight="bold"
            )
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            plt.tight_layout()
            output_path = plots_dir / "04_cylinder_camber.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            print(f"Saved: {output_path}")

            print("\n=== CAMBER SUMMARY (filtered to [0, 0.3]) ===")
            print(f"Mean camber: {np.mean(filtered_cambers):.6f}")
            print(f"Std camber: {np.std(filtered_cambers):.6f}")
            print(
                f"Mean 2*sigma uncertainty: {np.mean([2 * s for s in filtered_camber_sigmas]):.6f}"
            )
            print(
                f"Max 2*sigma uncertainty: {max([2 * s for s in filtered_camber_sigmas]):.6f}"
            )
            if measured_camber is not None:
                print(f"Measured camber: {measured_camber:.6f}")
    else:
        print("\nNo valid slices found for camber analysis")

    # Visualization 5: Sigma of point-to-green-circle distance per slice
    # The green (constrained) circle is ground truth: fixed radius = measured_radius from physical measurement.
    # This plot shows the spread (std) of point-to-ground-truth distances per slice.
    if len(sigma_point_to_green_mm) > 0:
        fig5 = plt.figure(figsize=(8, 5))
        ax5 = fig5.add_subplot(1, 1, 1)
        ax5.plot(
            slice_centers,
            sigma_point_to_green_mm,
            "o-",
            color="green",
            linewidth=2,
            markersize=6,
            label=r"$\sigma$-raw-distribution (point cloud to ground-truth circle)",
        )
        # Red curve: sigma of red-green circle distance in bbox only
        if len(sigma_red_green_bbox_mm) == len(slice_centers):
            red_sigma = np.array(sigma_red_green_bbox_mm)
            valid_red = ~np.isnan(red_sigma)
            if np.any(valid_red):
                ax5.plot(
                    np.array(slice_centers)[valid_red],
                    red_sigma[valid_red],
                    "s-",
                    color="red",
                    linewidth=2,
                    markersize=5,
                    label=r"$\sigma$ (red-green circle distance, bbox only)",
                )
        ax5.set_xlabel("Position along principal axis", fontsize=12)
        ax5.set_ylabel("Std of distance (mm)", fontsize=12)

        ax5.grid(True, alpha=0.3)
        ax5.legend()
        if "c2f_720" in name:
            ax5.set_ylim(top=150, bottom=0)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        output_path = plots_dir / "05_cylinder_sigma_point_cloud_to_ground_truth.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig5)
        print(f"Saved: {output_path}")
