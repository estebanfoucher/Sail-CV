"""
3D Visualization module for sail tracking debugging.

This module provides interactive 3D plotting for visualizing:
- Sail geometry at different angles and twist
- Camera position and viewing direction
- Telltale positions
- Projection from 3D to 2D

Supports both 1-DOF (angle only) and 2-DOF (angle + twist) visualization.

Requires matplotlib with mplot3d support.
"""

from typing import TYPE_CHECKING

import numpy as np
from projection import (
    get_sail_corners_world,
    get_sail_mesh_world,
    get_telltales_world,
    project_telltales,
)

from models import Detection
from models.sail_3d import Sail3DConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D


def plot_sail_camera_setup(
    config: Sail3DConfig,
    angle_deg: float = 0.0,
    twist_deg: float = 0.0,
    detections: list[Detection] | None = None,
    show_projection_rays: bool = False,
    ax: "Axes3D | None" = None,
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """
    Plot the complete 3D scene showing sail, camera, and telltales.

    Supports both flat sail (no twist) and twisted sail visualization.

    Args:
        config: Sail3DConfig with geometry and camera settings
        angle_deg: Base sail angle in degrees at the foot
        twist_deg: Twist angle in degrees (added at the head)
        detections: Optional list of detections (not used in 3D plot, reserved for future)
        show_projection_rays: If True, draw lines from camera to telltales
        ax: Optional existing Axes3D to plot on
        figsize: Figure size if creating new figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # --- Plot coordinate axes at origin ---
    axis_length = 1.0
    ax.quiver(
        0, 0, 0, axis_length, 0, 0, color="red", arrow_length_ratio=0.1, label="X (bow)"
    )
    ax.quiver(
        0,
        0,
        0,
        0,
        axis_length,
        0,
        color="green",
        arrow_length_ratio=0.1,
        label="Y (starboard)",
    )
    ax.quiver(
        0, 0, 0, 0, 0, axis_length, color="blue", arrow_length_ratio=0.1, label="Z (up)"
    )

    # --- Plot sail surface ---
    if abs(twist_deg) < 0.1:
        # No twist: draw as single polygon
        corners = get_sail_corners_world(config.sail, angle_deg, twist_deg)
        sail_poly = Poly3DCollection(
            [corners],
            alpha=0.3,
            facecolor="cyan",
            edgecolor="darkblue",
            linewidth=2,
        )
        ax.add_collection3d(sail_poly)
    else:
        # With twist: draw as mesh of quadrilaterals
        mesh = get_sail_mesh_world(config.sail, angle_deg, twist_deg, num_strips=8)
        for quad in mesh:
            sail_poly = Poly3DCollection(
                [quad],
                alpha=0.3,
                facecolor="cyan",
                edgecolor="darkblue",
                linewidth=1,
            )
            ax.add_collection3d(sail_poly)

    # --- Plot telltales on sail ---
    telltales = get_telltales_world(config.sail, angle_deg, twist_deg)
    for tell_id, world_pos in telltales:
        ax.scatter(
            world_pos[0],
            world_pos[1],
            world_pos[2],
            c="red",
            s=50,
            marker="o",
            label=f"Telltale {tell_id}" if tell_id == telltales[0][0] else None,
        )
        ax.text(
            world_pos[0],
            world_pos[1],
            world_pos[2] + 0.2,
            tell_id,
            fontsize=8,
            ha="center",
        )

    # --- Plot camera ---
    cam_pos = np.array(config.camera.position)
    look_at = np.array(config.camera.look_at)

    # Camera position
    ax.scatter(
        cam_pos[0],
        cam_pos[1],
        cam_pos[2],
        c="orange",
        s=100,
        marker="^",
        label="Camera",
    )

    # Camera viewing direction
    view_dir = look_at - cam_pos
    view_dir = view_dir / np.linalg.norm(view_dir) * 1.5  # Scale for visibility
    ax.quiver(
        cam_pos[0],
        cam_pos[1],
        cam_pos[2],
        view_dir[0],
        view_dir[1],
        view_dir[2],
        color="orange",
        arrow_length_ratio=0.1,
        linewidth=2,
    )

    # Camera FOV cone (simplified as lines to corners of image)
    if show_projection_rays:
        for _tell_id, world_pos in telltales:
            ax.plot(
                [cam_pos[0], world_pos[0]],
                [cam_pos[1], world_pos[1]],
                [cam_pos[2], world_pos[2]],
                "g--",
                alpha=0.3,
                linewidth=1,
            )

    # --- Set axis properties ---
    ax.set_xlabel("X (boat axis)")
    ax.set_ylabel("Y (lateral)")
    ax.set_zlabel("Z (vertical)")

    if abs(twist_deg) < 0.1:
        ax.set_title(f"Sail at {angle_deg:.1f}° angle")
    else:
        ax.set_title(f"Sail at {angle_deg:.1f}° angle, {twist_deg:.1f}° twist")

    # Set equal aspect ratio
    max_range = max(config.sail.width, config.sail.height, 3.0)
    ax.set_xlim(-1, max_range + 1)
    ax.set_ylim(-max_range / 2 - 1, max_range / 2 + 1)
    ax.set_zlim(0, config.sail.height + 1)

    # Add legend (only unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    return fig


def plot_sail_angle_sweep(
    config: Sail3DConfig,
    angles: list[float] | None = None,
    twist_deg: float = 0.0,
    figsize: tuple[int, int] = (16, 4),
) -> "Figure":
    """
    Plot sail at multiple angles side-by-side.

    Args:
        config: Sail3DConfig with geometry and camera settings
        angles: List of angles to plot (default: evenly spaced in range)
        twist_deg: Twist angle in degrees (same for all plots)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if angles is None:
        angles = [0.0, 15.0, 30.0, 45.0]

    n_angles = len(angles)
    fig = plt.figure(figsize=figsize)

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(1, n_angles, i + 1, projection="3d")

        # Plot sail (as mesh if twisted)
        if abs(twist_deg) < 0.1:
            corners = get_sail_corners_world(config.sail, angle, twist_deg)
            sail_poly = Poly3DCollection(
                [corners],
                alpha=0.4,
                facecolor="cyan",
                edgecolor="darkblue",
                linewidth=2,
            )
            ax.add_collection3d(sail_poly)
        else:
            mesh = get_sail_mesh_world(config.sail, angle, twist_deg, num_strips=6)
            for quad in mesh:
                sail_poly = Poly3DCollection(
                    [quad],
                    alpha=0.4,
                    facecolor="cyan",
                    edgecolor="darkblue",
                    linewidth=1,
                )
                ax.add_collection3d(sail_poly)

        # Plot camera
        cam_pos = np.array(config.camera.position)
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c="orange", s=50, marker="^")

        # Plot telltales
        telltales = get_telltales_world(config.sail, angle, twist_deg)
        for _tell_id, world_pos in telltales:
            ax.scatter(world_pos[0], world_pos[1], world_pos[2], c="red", s=20)

        # Settings
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if abs(twist_deg) < 0.1:
            ax.set_title(f"{angle:.0f}°")
        else:
            ax.set_title(f"{angle:.0f}° / {twist_deg:.0f}° twist")

        max_range = max(config.sail.width, config.sail.height, 3.0)
        ax.set_xlim(-1, max_range + 1)
        ax.set_ylim(-max_range / 2 - 1, max_range / 2 + 1)
        ax.set_zlim(0, config.sail.height + 1)

    plt.tight_layout()
    return fig


def plot_2d_projection_overlay(
    config: Sail3DConfig,
    angle_deg: float,
    twist_deg: float = 0.0,
    image: np.ndarray | None = None,
    detections: list[Detection] | None = None,
    assignments: list[tuple[int, str]] | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """
    Plot 2D image with projected telltale positions and detections.

    Args:
        config: Sail3DConfig with geometry and camera settings
        angle_deg: Base sail angle in degrees at the foot
        twist_deg: Twist angle in degrees (added at the head)
        image: Optional background image (BGR or RGB numpy array)
        detections: Optional list of detections to overlay
        assignments: Optional list of (detection_idx, telltale_id) matches
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches

    fig, ax = plt.subplots(figsize=figsize)

    # Plot background image if provided
    img_width, img_height = config.camera.image_size
    if image is not None:
        # Convert BGR to RGB if needed (3 channels, assume BGR from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        ax.imshow(image_rgb, extent=[0, img_width, img_height, 0])
    else:
        # Set background to light gray
        ax.set_facecolor("#f0f0f0")
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Invert Y for image coordinates

    # Project telltales
    projected = project_telltales(config.sail, config.camera, angle_deg, twist_deg)
    telltales = config.sail.telltales

    # Plot projected telltale positions
    for (proj_x, proj_y), telltale in zip(projected, telltales, strict=False):
        ax.scatter(proj_x, proj_y, c="blue", s=100, marker="x", linewidths=2, zorder=5)
        ax.annotate(
            telltale.id,
            (proj_x, proj_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="blue",
        )

    # Plot detections if provided
    if detections:
        for _i, det in enumerate(detections):
            xyxy = det.bbox.xyxy
            cx = (xyxy.x1 + xyxy.x2) / 2
            cy = (xyxy.y1 + xyxy.y2) / 2

            # Draw bounding box
            rect = patches.Rectangle(
                (xyxy.x1, xyxy.y1),
                xyxy.x2 - xyxy.x1,
                xyxy.y2 - xyxy.y1,
                linewidth=1,
                edgecolor="green",
                facecolor="none",
                zorder=3,
            )
            ax.add_patch(rect)

            # Draw center point
            ax.scatter(cx, cy, c="green", s=50, marker="o", zorder=4)

    # Draw assignment lines if provided
    if assignments and detections:
        telltale_map = {t.id: i for i, t in enumerate(telltales)}
        for det_idx, tell_id in assignments:
            if det_idx < len(detections) and tell_id in telltale_map:
                det = detections[det_idx]
                xyxy = det.bbox.xyxy
                det_cx = (xyxy.x1 + xyxy.x2) / 2
                det_cy = (xyxy.y1 + xyxy.y2) / 2

                tell_idx = telltale_map[tell_id]
                proj_x, proj_y = projected[tell_idx]

                ax.plot(
                    [det_cx, proj_x],
                    [det_cy, proj_y],
                    "r-",
                    linewidth=1,
                    alpha=0.7,
                    zorder=2,
                )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if abs(twist_deg) < 0.1:
        ax.set_title(f"2D Projection at {angle_deg:.1f}° angle")
    else:
        ax.set_title(f"2D Projection at {angle_deg:.1f}° angle, {twist_deg:.1f}° twist")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    ax.scatter([], [], c="blue", marker="x", label="Projected telltales")
    if detections:
        ax.scatter([], [], c="green", marker="o", label="Detections")
    ax.legend(loc="upper right")

    return fig


def create_interactive_angle_slider(
    config: Sail3DConfig,
    initial_angle: float = 0.0,
    initial_twist: float = 0.0,
) -> None:
    """
    Create matplotlib figure with sliders to interactively adjust sail angle and twist.

    The 3D view updates in real-time as the sliders are moved.

    Args:
        config: Sail3DConfig with geometry and camera settings
        initial_angle: Initial base angle to display
        initial_twist: Initial twist to display
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 9))

    # Create 3D axis with space for sliders
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)

    # Storage for current plot elements
    plot_elements = {"sail_polys": [], "telltales": [], "texts": []}
    current_params = {"angle": initial_angle, "twist": initial_twist}

    def update_plot(_=None) -> None:
        """Update the plot for current angle and twist."""
        angle = current_params["angle"]
        twist = current_params["twist"]

        # Remove old sail polygons
        for poly in plot_elements["sail_polys"]:
            poly.remove()
        plot_elements["sail_polys"] = []

        # Remove old telltale markers and texts
        for scatter in plot_elements["telltales"]:
            scatter.remove()
        for text in plot_elements["texts"]:
            text.remove()
        plot_elements["telltales"] = []
        plot_elements["texts"] = []

        # Draw new sail (as mesh if twisted)
        if abs(twist) < 0.1:
            corners = get_sail_corners_world(config.sail, angle, twist)
            sail_poly = Poly3DCollection(
                [corners],
                alpha=0.3,
                facecolor="cyan",
                edgecolor="darkblue",
                linewidth=2,
            )
            ax.add_collection3d(sail_poly)
            plot_elements["sail_polys"].append(sail_poly)
        else:
            mesh = get_sail_mesh_world(config.sail, angle, twist, num_strips=8)
            for quad in mesh:
                sail_poly = Poly3DCollection(
                    [quad],
                    alpha=0.3,
                    facecolor="cyan",
                    edgecolor="darkblue",
                    linewidth=1,
                )
                ax.add_collection3d(sail_poly)
                plot_elements["sail_polys"].append(sail_poly)

        # Draw new telltales
        telltales = get_telltales_world(config.sail, angle, twist)
        for tell_id, world_pos in telltales:
            scatter = ax.scatter(
                world_pos[0], world_pos[1], world_pos[2], c="red", s=50, marker="o"
            )
            plot_elements["telltales"].append(scatter)

            text = ax.text(
                world_pos[0],
                world_pos[1],
                world_pos[2] + 0.2,
                tell_id,
                fontsize=8,
                ha="center",
            )
            plot_elements["texts"].append(text)

        if abs(twist) < 0.1:
            ax.set_title(f"Sail at {angle:.1f}° angle")
        else:
            ax.set_title(f"Sail at {angle:.1f}° angle, {twist:.1f}° twist")
        fig.canvas.draw_idle()

    def on_angle_change(val: float) -> None:
        current_params["angle"] = val
        update_plot()

    def on_twist_change(val: float) -> None:
        current_params["twist"] = val
        update_plot()

    # Initial plot setup (static elements)
    # Coordinate axes
    axis_length = 1.0
    ax.quiver(0, 0, 0, axis_length, 0, 0, color="red", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color="green", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color="blue", arrow_length_ratio=0.1)

    # Camera
    cam_pos = np.array(config.camera.position)
    look_at = np.array(config.camera.look_at)
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c="orange", s=100, marker="^")
    view_dir = look_at - cam_pos
    view_dir = view_dir / np.linalg.norm(view_dir) * 1.5
    ax.quiver(
        cam_pos[0],
        cam_pos[1],
        cam_pos[2],
        view_dir[0],
        view_dir[1],
        view_dir[2],
        color="orange",
        arrow_length_ratio=0.1,
        linewidth=2,
    )

    # Axis labels and limits
    ax.set_xlabel("X (boat axis)")
    ax.set_ylabel("Y (lateral)")
    ax.set_zlabel("Z (vertical)")

    max_range = max(config.sail.width, config.sail.height, 3.0)
    ax.set_xlim(-1, max_range + 1)
    ax.set_ylim(-max_range / 2 - 1, max_range / 2 + 1)
    ax.set_zlim(0, config.sail.height + 1)

    # Create sliders
    angle_slider_ax = fig.add_axes((0.2, 0.1, 0.6, 0.03))
    angle_slider = Slider(
        angle_slider_ax,
        "Base Angle (°)",
        config.angle_min,
        config.angle_max,
        valinit=initial_angle,
        valstep=0.5,
    )
    angle_slider.on_changed(on_angle_change)

    twist_slider_ax = fig.add_axes((0.2, 0.05, 0.6, 0.03))
    twist_slider = Slider(
        twist_slider_ax,
        "Twist (°)",
        config.twist_min,
        config.twist_max,
        valinit=initial_twist,
        valstep=0.5,
    )
    twist_slider.on_changed(on_twist_change)

    # Initial plot
    update_plot()

    plt.show()


def plot_projection_comparison(
    config: Sail3DConfig,
    angle_deg: float,
    twist_deg: float = 0.0,
    figsize: tuple[int, int] = (14, 6),
) -> "Figure":
    """
    Create a side-by-side plot showing 3D scene and 2D projection.

    Args:
        config: Sail3DConfig with geometry and camera settings
        angle_deg: Base sail angle in degrees at the foot
        twist_deg: Twist angle in degrees (added at the head)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)

    # --- Left: 3D view ---
    ax3d = fig.add_subplot(121, projection="3d")

    # Coordinate axes
    axis_length = 1.0
    ax3d.quiver(0, 0, 0, axis_length, 0, 0, color="red", arrow_length_ratio=0.1)
    ax3d.quiver(0, 0, 0, 0, axis_length, 0, color="green", arrow_length_ratio=0.1)
    ax3d.quiver(0, 0, 0, 0, 0, axis_length, color="blue", arrow_length_ratio=0.1)

    # Sail (as mesh if twisted)
    if abs(twist_deg) < 0.1:
        corners = get_sail_corners_world(config.sail, angle_deg, twist_deg)
        sail_poly = Poly3DCollection(
            [corners],
            alpha=0.3,
            facecolor="cyan",
            edgecolor="darkblue",
            linewidth=2,
        )
        ax3d.add_collection3d(sail_poly)
    else:
        mesh = get_sail_mesh_world(config.sail, angle_deg, twist_deg, num_strips=8)
        for quad in mesh:
            sail_poly = Poly3DCollection(
                [quad],
                alpha=0.3,
                facecolor="cyan",
                edgecolor="darkblue",
                linewidth=1,
            )
            ax3d.add_collection3d(sail_poly)

    # Telltales
    telltales = get_telltales_world(config.sail, angle_deg, twist_deg)
    for tell_id, world_pos in telltales:
        ax3d.scatter(world_pos[0], world_pos[1], world_pos[2], c="red", s=50)
        ax3d.text(world_pos[0], world_pos[1], world_pos[2] + 0.2, tell_id, fontsize=8)

    # Camera
    cam_pos = np.array(config.camera.position)
    look_at = np.array(config.camera.look_at)
    ax3d.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c="orange", s=100, marker="^")
    view_dir = look_at - cam_pos
    view_dir = view_dir / np.linalg.norm(view_dir) * 1.5
    ax3d.quiver(
        cam_pos[0],
        cam_pos[1],
        cam_pos[2],
        view_dir[0],
        view_dir[1],
        view_dir[2],
        color="orange",
        arrow_length_ratio=0.1,
    )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    if abs(twist_deg) < 0.1:
        ax3d.set_title(f"3D Scene (angle={angle_deg:.1f}°)")
    else:
        ax3d.set_title(f"3D Scene (angle={angle_deg:.1f}°, twist={twist_deg:.1f}°)")

    max_range = max(config.sail.width, config.sail.height, 3.0)
    ax3d.set_xlim(-1, max_range + 1)
    ax3d.set_ylim(-max_range / 2 - 1, max_range / 2 + 1)
    ax3d.set_zlim(0, config.sail.height + 1)

    # --- Right: 2D projection ---
    ax2d = fig.add_subplot(122)

    img_width, img_height = config.camera.image_size
    ax2d.set_xlim(0, img_width)
    ax2d.set_ylim(img_height, 0)
    ax2d.set_facecolor("#f0f0f0")
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    # Project and plot telltales
    projected = project_telltales(config.sail, config.camera, angle_deg, twist_deg)
    for (proj_x, proj_y), telltale in zip(
        projected, config.sail.telltales, strict=False
    ):
        ax2d.scatter(proj_x, proj_y, c="blue", s=100, marker="x", linewidths=2)
        ax2d.annotate(
            telltale.id,
            (proj_x, proj_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="blue",
        )

    ax2d.set_xlabel("X (pixels)")
    ax2d.set_ylabel("Y (pixels)")
    if abs(twist_deg) < 0.1:
        ax2d.set_title(f"2D Projection (angle={angle_deg:.1f}°)")
    else:
        ax2d.set_title(
            f"2D Projection (angle={angle_deg:.1f}°, twist={twist_deg:.1f}°)"
        )

    plt.tight_layout()
    return fig
