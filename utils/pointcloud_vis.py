import numpy as np
import torch


def visualize_pointcloud_distribution(gaussians, save_path, cameras=None, iteration=None):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(18, 6))
    points = gaussians.get_xyz.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy().flatten()

    max_range = np.array(
        [
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        ]
    ).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    cam_data = []
    if cameras is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
        for idx, cam in enumerate(cameras):
            if hasattr(cam, "world_view_transform"):
                w2c = cam.world_view_transform.detach().cpu().numpy() if torch.is_tensor(cam.world_view_transform) else np.array(cam.world_view_transform)
                w2c = w2c.transpose(1, 0)
                c2w = np.linalg.inv(w2c)
                cam_center = c2w[:3, 3]
                c2w_rot = c2w[:3, :3]
            elif hasattr(cam, "R") and hasattr(cam, "T"):
                r = cam.R.detach().cpu().numpy() if torch.is_tensor(cam.R) else np.array(cam.R)
                t = cam.T.detach().cpu().numpy() if torch.is_tensor(cam.T) else np.array(cam.T)
                c2w_rot = r.T
                cam_center = -r.T @ t
            elif hasattr(cam, "camera_center"):
                cam_center = cam.camera_center.detach().cpu().numpy() if torch.is_tensor(cam.camera_center) else np.array(cam.camera_center)
                continue
            else:
                continue

            cam_id = getattr(cam, "uid", getattr(cam, "colmap_id", getattr(cam, "image_name", idx)))
            cam_data.append(
                {
                    "center": cam_center,
                    "direction": c2w_rot @ np.array([0, 0, 1]),
                    "right": c2w_rot @ np.array([1, 0, 0]),
                    "up": c2w_rot @ np.array([0, 1, 0]),
                    "id": cam_id,
                    "color": colors[idx],
                }
            )

        if cam_data:
            cam_centers = np.array([cam["center"] for cam in cam_data])
            all_points = np.vstack([points, cam_centers])
            max_range = np.array(
                [
                    all_points[:, 0].max() - all_points[:, 0].min(),
                    all_points[:, 1].max() - all_points[:, 1].min(),
                    all_points[:, 2].max() - all_points[:, 2].min(),
                ]
            ).max() / 2.0
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    arrow_length = max_range * 0.15
    views = [
        {"title": "Front View (X-Y plane)", "x": 0, "y": 1, "xlabel": "X", "ylabel": "Y"},
        {"title": "Side View (Y-Z plane)", "x": 1, "y": 2, "xlabel": "Y", "ylabel": "Z"},
        {"title": "Top View (X-Z plane)", "x": 0, "y": 2, "xlabel": "X", "ylabel": "Z"},
    ]

    for view_idx, view in enumerate(views):
        ax = fig.add_subplot(1, 3, view_idx + 1)
        scatter = ax.scatter(
            points[:, view["x"]],
            points[:, view["y"]],
            c=opacities,
            s=2,
            alpha=0.6,
            cmap="viridis",
            vmin=0,
            vmax=1,
            label="Point Cloud",
        )

        for cam_info in cam_data:
            center = cam_info["center"]
            color = cam_info["color"]
            center_2d = np.array([center[view["x"]], center[view["y"]]])
            ax.scatter(center_2d[0], center_2d[1], c=[color], s=150, marker="o", edgecolors="black", linewidths=1.5)
            for direction, arrow_color, scale in (
                (cam_info["direction"], "blue", 1.0),
                (cam_info["right"], "red", 0.6),
                (cam_info["up"], "green", 0.6),
            ):
                direction_2d = np.array([direction[view["x"]], direction[view["y"]]])
                direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-8)
                ax.arrow(
                    center_2d[0],
                    center_2d[1],
                    direction_2d[0] * arrow_length * scale,
                    direction_2d[1] * arrow_length * scale,
                    head_width=arrow_length * 0.12,
                    head_length=arrow_length * 0.16,
                    fc=arrow_color,
                    ec=arrow_color,
                    linewidth=2,
                    alpha=0.8,
                )

        ax.set_xlabel(view["xlabel"], fontsize=12)
        ax.set_ylabel(view["ylabel"], fontsize=12)
        title = view["title"]
        if iteration is not None:
            title += f" (Iter {iteration})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        if view["x"] == 0 and view["y"] == 1:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
        elif view["x"] == 1 and view["y"] == 2:
            ax.set_xlim(mid_y - max_range, mid_y + max_range)
            ax.set_ylim(mid_z - max_range, mid_z + max_range)
        else:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_z - max_range, mid_z + max_range)

        if view_idx == 2:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Opacity", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
