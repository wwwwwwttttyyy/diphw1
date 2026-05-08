import argparse
import math
import os
from pathlib import Path

# My dip environment may load two OpenMP runtimes when importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


IMAGE_SIZE = 1024.0
INIT_FOCAL = 880.0
INIT_DEPTH = 2.5
INIT_YAW_DEG = -70.0
LR_POINTS = 0.01
LR_CAMERA = 0.005
LR_FOCAL = 0.002
PRINT_EVERY = 50
EVAL_CHUNK = 200000


def load_data(data_dir):
    data_dir = Path(data_dir)
    points2d_file = data_dir / "points2d.npz"
    colors_file = data_dir / "points3d_colors.npy"

    pack = np.load(points2d_file)
    keys = sorted(pack.files)
    points2d = np.stack([pack[k] for k in keys], axis=0).astype(np.float32)
    colors = np.load(colors_file).astype(np.float32)
    return points2d, colors, keys


def make_observations(points2d):
    visible = points2d[:, :, 2] > 0.5
    view_id, point_id = np.nonzero(visible)
    xy = points2d[view_id, point_id, :2]
    return (
        view_id.astype(np.int64),
        point_id.astype(np.int64),
        xy.astype(np.float32),
    )


def init_parameters(points2d, focal, depth, image_size, yaw_deg, seed):
    rng = np.random.default_rng(seed)
    n_views, n_points, _ = points2d.shape
    cx = image_size / 2.0
    cy = image_size / 2.0

    visible = points2d[:, :, 2] > 0.5
    visible_count = np.maximum(visible.sum(axis=0), 1)

    mean_x = (points2d[:, :, 0] * visible).sum(axis=0) / visible_count
    mean_y = (points2d[:, :, 1] * visible).sum(axis=0) / visible_count

    # Put the initial points on a rough fronto-parallel plane.
    x = (mean_x - cx) * depth / focal
    y = (cy - mean_y) * depth / focal
    z = rng.normal(0.0, 0.04, size=n_points)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    points -= points.mean(axis=0, keepdims=True)

    view_count = np.maximum(visible.sum(axis=1), 1)
    view_mean_x = (points2d[:, :, 0] * visible).sum(axis=1) / view_count
    view_mean_y = (points2d[:, :, 1] * visible).sum(axis=1) / view_count

    translations = np.zeros((n_views, 3), dtype=np.float32)
    translations[:, 0] = (view_mean_x - cx) * depth / focal
    translations[:, 1] = (cy - view_mean_y) * depth / focal
    translations[:, 2] = -depth

    eulers = np.zeros((n_views, 3), dtype=np.float32)
    center_view = n_views // 2
    max_offset = max(center_view, n_views - center_view - 1)
    yaw = np.deg2rad(yaw_deg)
    for i in range(n_views):
        eulers[i, 1] = (i - center_view) / max_offset * yaw

    eulers[center_view] = 0.0
    translations[center_view] = np.array([0.0, 0.0, -depth], dtype=np.float32)
    return points, eulers, translations, center_view


def euler_to_matrix(angles):
    rx = angles[:, 0]
    ry = angles[:, 1]
    rz = angles[:, 2]

    sx, cx = torch.sin(rx), torch.cos(rx)
    sy, cy = torch.sin(ry), torch.cos(ry)
    sz, cz = torch.sin(rz), torch.cos(rz)

    zero = torch.zeros_like(rx)
    one = torch.ones_like(rx)

    row0 = torch.stack([one, zero, zero], dim=1)
    row1 = torch.stack([zero, cx, -sx], dim=1)
    row2 = torch.stack([zero, sx, cx], dim=1)
    rot_x = torch.stack([row0, row1, row2], dim=1)

    row0 = torch.stack([cy, zero, sy], dim=1)
    row1 = torch.stack([zero, one, zero], dim=1)
    row2 = torch.stack([-sy, zero, cy], dim=1)
    rot_y = torch.stack([row0, row1, row2], dim=1)

    row0 = torch.stack([cz, -sz, zero], dim=1)
    row1 = torch.stack([sz, cz, zero], dim=1)
    row2 = torch.stack([zero, zero, one], dim=1)
    rot_z = torch.stack([row0, row1, row2], dim=1)

    return rot_z @ rot_y @ rot_x


def euler_to_matrix_np(angles):
    angles = np.asarray(angles)
    mats = []
    for rx, ry, rz in angles:
        sx, cx = math.sin(rx), math.cos(rx)
        sy, cy = math.sin(ry), math.cos(ry)
        sz, cz = math.sin(rz), math.cos(rz)

        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        mats.append(rot_z @ rot_y @ rot_x)
    return np.stack(mats, axis=0)


def camera_values(eulers, translations, fixed_view, depth):
    if fixed_view is None:
        return eulers, translations

    fixed_eulers = eulers.clone()
    fixed_translations = translations.clone()
    fixed_eulers[fixed_view] = 0.0
    fixed_translations[fixed_view] = fixed_translations.new_tensor([0.0, 0.0, -depth])
    return fixed_eulers, fixed_translations


def project(points, eulers, translations, view_id, point_id, log_focal, image_size):
    cx = image_size / 2.0
    cy = image_size / 2.0

    rotations = euler_to_matrix(eulers)
    pts = points[point_id]
    rot = rotations[view_id]
    trans = translations[view_id]

    cam = (rot @ pts.unsqueeze(-1)).squeeze(-1) + trans
    z = cam[:, 2]
    z = torch.where(z.abs() < 1e-4, torch.full_like(z, -1e-4), z)

    focal = torch.exp(log_focal)
    u = -focal * cam[:, 0] / z + cx
    v = focal * cam[:, 1] / z + cy
    return torch.stack([u, v], dim=1), focal


def calc_full_rmse(points, eulers, translations, view_id, point_id, xy, log_focal, image_size, chunk):
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, xy.shape[0], chunk):
            end = min(start + chunk, xy.shape[0])
            pred, _ = project(
                points,
                eulers,
                translations,
                view_id[start:end],
                point_id[start:end],
                log_focal,
                image_size,
            )
            diff = pred - xy[start:end]
            total += diff.pow(2).sum().item()
            count += diff.shape[0]
    return math.sqrt(total / max(count, 1))


def save_obj(path, points, colors):
    colors = np.clip(colors, 0.0, 1.0)
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(
                f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
            )


def save_point_preview(path, points, colors, seed):
    rng = np.random.default_rng(seed)
    n = min(12000, len(points))
    ids = rng.choice(len(points), size=n, replace=False)

    plt.figure(figsize=(5, 7))
    plt.scatter(
        points[ids, 0],
        points[ids, 1],
        c=np.clip(colors[ids], 0.0, 1.0),
        s=0.7,
        linewidths=0,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("reconstructed point cloud - front view")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def set_equal_3d(ax, points):
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    center = (p_min + p_max) * 0.5
    radius = (p_max - p_min).max() * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_point_cloud_views(path, points, colors, seed):
    rng = np.random.default_rng(seed)
    n = min(14000, len(points))
    ids = rng.choice(len(points), size=n, replace=False)
    p = points[ids]
    c = np.clip(colors[ids], 0.0, 1.0)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(p[:, 0], p[:, 1], c=c, s=0.45, linewidths=0)
    ax.set_title("front view")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(p[:, 2], p[:, 1], c=c, s=0.45, linewidths=0)
    ax.set_title("side view")
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(p[:, 0], p[:, 2], c=c, s=0.45, linewidths=0)
    ax.set_title("top view")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=0.35, linewidths=0)
    ax.set_title("3D view")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=15, azim=-65)
    set_equal_3d(ax, points)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_point_cloud_gif(path, points, colors, seed):
    rng = np.random.default_rng(seed)
    n = min(20000, len(points))
    ids = rng.choice(len(points), size=n, replace=False)
    p = points[ids]
    c = np.clip(colors[ids] * 0.55, 0.0, 1.0)
    p_show = np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=1)
    all_show = np.stack([points[:, 0], points[:, 2], points[:, 1]], axis=1)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    frames = []

    for azim in np.linspace(0, 360, 36, endpoint=False):
        ax.clear()
        ax.scatter(
            p_show[:, 0],
            p_show[:, 1],
            p_show[:, 2],
            c=c,
            s=1.1,
            linewidths=0,
            depthshade=False,
        )
        ax.view_init(elev=0, azim=azim)
        set_equal_3d(ax, all_show)
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(Image.fromarray(frame))

    plt.close(fig)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )


def project_np(points, eulers, translations, focal, view_index, image_size):
    rotations = euler_to_matrix_np(eulers)
    cam = points @ rotations[view_index].T + translations[view_index]
    z = cam[:, 2]
    z = np.where(np.abs(z) < 1e-4, -1e-4, z)
    cx = image_size / 2.0
    cy = image_size / 2.0
    u = -focal * cam[:, 0] / z + cx
    v = focal * cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def save_reprojection_check(path, points, eulers, translations, focal, points2d, image_size):
    views = [0, points2d.shape[0] // 2, points2d.shape[0] - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, view_index in zip(axes, views):
        obs = points2d[view_index]
        visible = obs[:, 2] > 0.5
        pred = project_np(points, eulers, translations, focal, view_index, image_size)
        ax.scatter(obs[visible, 0], obs[visible, 1], s=0.4, c="black", alpha=0.35, label="observed")
        ax.scatter(pred[visible, 0], pred[visible, 1], s=0.4, c="red", alpha=0.35, label="projected")
        ax.set_title(f"view {view_index:03d}")
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    axes[0].legend(markerscale=5, loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_camera_plot(path, eulers, translations):
    rotations = euler_to_matrix_np(eulers)
    centers = []
    for r, t in zip(rotations, translations):
        centers.append(-(r.T @ t))
    centers = np.stack(centers, axis=0)

    plt.figure(figsize=(6, 5))
    plt.plot(centers[:, 0], centers[:, 2], marker="o", markersize=3)
    for i in range(0, len(centers), 5):
        plt.text(centers[i, 0], centers[i, 2], str(i), fontsize=8)
    plt.scatter([0], [0], c="red", s=25, label="object center")
    plt.xlabel("camera center X")
    plt.ylabel("camera center Z")
    plt.title("estimated camera centers")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_loss_plot(path, history):
    steps = [x[0] for x in history]
    rmses = [x[1] for x in history]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, rmses)
    plt.xlabel("iteration")
    plt.ylabel("batch reprojection RMSE / px")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points2d, colors, keys = load_data(args.data_dir)
    view_id_np, point_id_np, xy_np = make_observations(points2d)
    init_points, init_eulers, init_trans, center_view = init_parameters(
        points2d,
        INIT_FOCAL,
        INIT_DEPTH,
        IMAGE_SIZE,
        INIT_YAW_DEG,
        args.seed,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")
    print(f"views: {len(keys)}, points: {points2d.shape[1]}, observations: {len(xy_np)}")

    view_id = torch.from_numpy(view_id_np).to(device)
    point_id = torch.from_numpy(point_id_np).to(device)
    xy = torch.from_numpy(xy_np).to(device)

    points = torch.nn.Parameter(torch.from_numpy(init_points).to(device))
    eulers = torch.nn.Parameter(torch.from_numpy(init_eulers).to(device))
    translations = torch.nn.Parameter(torch.from_numpy(init_trans).to(device))
    log_focal = torch.nn.Parameter(torch.tensor(math.log(INIT_FOCAL), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": LR_POINTS},
            {"params": [eulers, translations], "lr": LR_CAMERA},
            {"params": [log_focal], "lr": LR_FOCAL},
        ]
    )

    fixed_view = center_view
    print(f"fixed reference view: {fixed_view:03d}")

    n_obs = xy.shape[0]
    batch_size = n_obs if args.batch_size <= 0 else min(args.batch_size, n_obs)
    history = []

    for it in range(1, args.iters + 1):
        if batch_size == n_obs:
            batch = torch.arange(n_obs, device=device)
        else:
            batch = torch.randint(0, n_obs, (batch_size,), device=device)

        cur_eulers, cur_trans = camera_values(eulers, translations, fixed_view, INIT_DEPTH)
        pred, focal = project(
            points,
            cur_eulers,
            cur_trans,
            view_id[batch],
            point_id[batch],
            log_focal,
            IMAGE_SIZE,
        )

        diff = pred - xy[batch]
        reproj_mse = diff.pow(2).mean()
        center_penalty = points.mean(dim=0).pow(2).sum()
        loss = reproj_mse + 1e-4 * center_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 1 or it % PRINT_EVERY == 0 or it == args.iters:
            rmse = math.sqrt(diff.pow(2).sum(dim=1).mean().item())
            history.append((it, rmse))
            print(f"iter {it:5d} | batch rmse {rmse:8.3f} px | focal {focal.item():8.2f}")

    cur_eulers, cur_trans = camera_values(eulers, translations, fixed_view, INIT_DEPTH)
    full_rmse = calc_full_rmse(
        points,
        cur_eulers,
        cur_trans,
        view_id,
        point_id,
        xy,
        log_focal,
        IMAGE_SIZE,
        EVAL_CHUNK,
    )
    print(f"final full rmse: {full_rmse:.3f} px")
    print(f"final focal: {torch.exp(log_focal).item():.3f}")

    points_out = points.detach().cpu().numpy()
    eulers_out = cur_eulers.detach().cpu().numpy()
    trans_out = cur_trans.detach().cpu().numpy()
    focal_out = float(torch.exp(log_focal).detach().cpu())

    np.savez(
        out_dir / "ba_params.npz",
        points=points_out,
        eulers=eulers_out,
        translations=trans_out,
        focal=focal_out,
        keys=np.array(keys),
        final_rmse=full_rmse,
    )
    save_obj(out_dir / "ba_points.obj", points_out, colors)
    save_loss_plot(out_dir / "loss.png", history)
    save_point_preview(out_dir / "point_cloud_preview.png", points_out, colors, args.seed)
    save_point_cloud_views(out_dir / "point_cloud_views.png", points_out, colors, args.seed)
    save_point_cloud_gif(out_dir / "point_cloud.gif", points_out, colors, args.seed)
    save_reprojection_check(
        out_dir / "reprojection_check.png",
        points_out,
        eulers_out,
        trans_out,
        focal_out,
        points2d,
        IMAGE_SIZE,
    )
    save_camera_plot(out_dir / "camera_centers.png", eulers_out, trans_out)

    print(f"saved: {out_dir / 'ba_points.obj'}")
    print(f"saved: {out_dir / 'loss.png'}")
    print(f"saved: {out_dir / 'point_cloud_preview.png'}")
    print(f"saved: {out_dir / 'point_cloud_views.png'}")
    print(f"saved: {out_dir / 'point_cloud.gif'}")
    print(f"saved: {out_dir / 'reprojection_check.png'}")
    print(f"saved: {out_dir / 'camera_centers.png'}")
    print(f"saved: {out_dir / 'ba_params.npz'}")


if __name__ == "__main__":
    main()
