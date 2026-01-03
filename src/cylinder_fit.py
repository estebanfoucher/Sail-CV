import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from sklearn.decomposition import PCA

# -------------------------------------------------------
# 0. Load parameters from JSON
# -------------------------------------------------------
params_json_path = "output/point_clouds/post_process_parameters.json"
with open(params_json_path) as f:
    params_dict = json.load(f)

print(f"\n=== LOADED PARAMETERS FROM {params_json_path} ===")
print(f"Found {len(params_dict)} point cloud configurations\n")


# -------------------------------------------------------
# 2. Find best long axis using PCA
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
# 3. Slice point cloud along axis
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
# 4. Project points to orthogonal plane
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
# 5. Fit circle to 2D points
# -------------------------------------------------------
def fit_circle_2d(points_2d):
    """
    Fit a circle to 2D points using least-squares.

    Args:
        points_2d: 2D point array (N, 2)

    Returns:
        center: Circle center (2D)
        radius: Circle radius
        sigma: Uncertainty (standard deviation) of the radius estimate
    """
    if len(points_2d) < 3:
        return None, None, None

    # Algebraic circle fitting (Kasa method)
    # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
    # Expanding: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
    # Let A = -2*cx, B = -2*cy, C = cx^2 + cy^2 - r^2
    # Then: x^2 + y^2 + A*x + B*y + C = 0

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # Build system of equations
    A_matrix = np.column_stack([x, y, np.ones(len(x))])
    b_vector = -(x**2 + y**2)

    # Solve least-squares
    try:
        params, _residuals, _rank, _s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

        # Extract center and radius
        cx = -params[0] / 2
        cy = -params[1] / 2
        center = np.array([cx, cy])

        # Radius from center and constant term
        r_squared = cx**2 + cy**2 - params[2]
        if r_squared > 0:
            radius = np.sqrt(r_squared)
        else:
            # Fallback: use mean distance from center
            distances = np.linalg.norm(points_2d - center, axis=1)
            radius = np.mean(distances)

        # Calculate residuals: distance from each point to the fitted circle
        distances_to_circle = np.abs(
            np.linalg.norm(points_2d - center, axis=1) - radius
        )

        # Standard deviation of residuals gives uncertainty in radius
        # Use sample standard deviation (ddof=1 for unbiased estimate)
        if len(distances_to_circle) > 1:
            sigma = np.std(distances_to_circle, ddof=1)
        else:
            sigma = 0.0

        return center, radius, sigma
    except:
        # Fallback: use centroid and mean distance
        center = points_2d.mean(axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        radius = np.mean(distances)
        # Estimate sigma from spread of distances
        sigma = np.std(distances, ddof=1) if len(distances) > 1 else 0.0
        return center, radius, sigma


# -------------------------------------------------------
# 5a. Compute camber from fitted circle and filter radius
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
# 5b. Filter points by distance from axis
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
# 6. Main analysis: Loop over point clouds
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

    # Find principal axis
    axis_direction, axis_point = find_principal_axis(points)
    print(f"Axis direction (normalized): {axis_direction}")
    print(f"Axis point (centroid): {axis_point}")

    # Filter points by distance from axis (before slicing)
    points_filtered, filter_mask = filter_points_by_axis_distance(
        points, axis_direction, axis_point, filter_radius=filter_radius
    )
    print(
        f"\nFiltered points: {len(points)} -> {len(points_filtered)} "
        f"(removed {len(points) - len(points_filtered)} points "
        f"beyond {filter_radius}m from axis)"
    )

    # Slice along axis
    slices, slice_positions = slice_along_axis(
        points_filtered, axis_direction, axis_point, n_slices=n_slices
    )

    print(f"\nCreated {len(slices)} slices along the axis")
    for i, (slice_pts, pos) in enumerate(zip(slices, slice_positions, strict=False)):
        print(f"  Slice {i + 1}: {len(slice_pts)} points, axial position = {pos:.3f}")

    # Process each slice
    radii = []
    radius_sigmas = []  # Store uncertainty (sigma) for each radius
    cambers = []  # Store camber values
    camber_sigmas = []  # Store uncertainty (sigma) for each camber
    slice_centers = []
    valid_slices = []
    circle_centers_2d = []  # Store circle centers for visualization
    valid_slice_points = []  # Store original slice points for all valid slices

    for i, (slice_pts, pos) in enumerate(zip(slices, slice_positions, strict=False)):
        if len(slice_pts) < 3:
            print(f"\nSlice {i + 1}: Skipped (too few points: {len(slice_pts)})")
            continue

        # Project to orthogonal plane
        points_2d = project_to_orthogonal_plane(slice_pts, axis_direction, axis_point)

        # Fit circle
        center_2d, radius, sigma = fit_circle_2d(points_2d)

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

        radii.append(radius)
        radius_sigmas.append(sigma)
        cambers.append(camber)
        camber_sigmas.append(camber_sigma)
        slice_centers.append(pos)
        circle_centers_2d.append(center_2d)
        valid_slices.append(i + 1)
        valid_slice_points.append(slice_pts)  # Store for visualization

        print(f"\nSlice {i + 1} (axial position = {pos:.3f}):")
        print(f"  Circle center (2D): {center_2d}")
        print(f"  Radius: {radius:.6f} ± {2 * sigma:.6f} (2*sigma)")
        print(f"  Camber: {camber:.6f} ± {2 * camber_sigma:.6f} (2*sigma)")

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

    # Visualization 2: All successful projected circle fittings
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

            # Plot projected points
            ax.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                s=5,
                alpha=0.6,
                c="blue",
                label="Projected Points",
            )

            # Plot fitted circle
            circle = plt.Circle(
                center_2d,
                radius_val,
                fill=False,
                color="red",
                linewidth=2,
                label="Fitted Circle",
            )
            ax.add_patch(circle)
            ax.scatter(*center_2d, color="red", s=20, marker="o", label="Circle Center")

            ax.set_xlabel("u (plane coordinate)", fontsize=10)
            ax.set_ylabel("v (plane coordinate)", fontsize=10)
            ax.set_title(
                f"Slice {slice_num}\nPos: {pos:.3f}, R: {radius_val:.4f}", fontsize=9
            )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

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
            label="Fitted Radius (±2*sigma)",
        )
        ax2.axhline(
            measured_radius,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Measured Radius",
        )
        ax2.set_xlabel("Position along axis", fontsize=12)
        ax2.set_ylabel("Radius", fontsize=12)
        ax2.set_title("Radius vs Position Along Axis", fontsize=14, fontweight="bold")
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
