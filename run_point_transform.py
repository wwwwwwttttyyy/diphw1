import cv2
import numpy as np
import gradio as gr
from typing import Union, List, Tuple

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


# Point-guided image deformation
def _mls_affine_map_points(
    points: np.ndarray,
    control_from: np.ndarray,
    control_to: np.ndarray,
    alpha: float,
    eps: float,
    chunk_size: int = 65536,
) -> np.ndarray:
    displacement_all = control_to - control_from
    mapped_points = np.empty_like(points, dtype=np.float32)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        v = points[start:end]

        diff = v[:, None, :] - control_from[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(v)), nearest_idx]
        exact_mask = nearest_dist < eps

        weights = 1.0 / (np.maximum(dists, eps) ** alpha)
        sum_w = np.sum(weights, axis=1, keepdims=True)
        weighted_disp = (weights @ displacement_all) / sum_w

        if len(control_from) < 3:
            mapped_chunk = v + weighted_disp
            mapped_chunk[exact_mask] = control_to[nearest_idx[exact_mask]]
            mapped_points[start:end] = mapped_chunk
            continue

        p_star = (weights @ control_from) / sum_w
        q_star = (weights @ control_to) / sum_w
        p_hat = control_from[None, :, :] - p_star[:, None, :]
        q_hat = control_to[None, :, :] - q_star[:, None, :]

        A = np.einsum("bn,bni,bnj->bij", weights, p_hat, p_hat, optimize=True)
        B = np.einsum("bn,bni,bnj->bij", weights, p_hat, q_hat, optimize=True)

        det_A = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        fallback_mask = np.abs(det_A) < 1e-8
        mapped_chunk = np.empty((len(v), 2), dtype=np.float32)

        valid_mask = ~fallback_mask
        if np.any(valid_mask):
            M = np.linalg.solve(A[valid_mask], B[valid_mask])
            mapped_chunk[valid_mask] = (
                np.einsum("bi,bij->bj", v[valid_mask] - p_star[valid_mask], M, optimize=True)
                + q_star[valid_mask]
            )

        if np.any(fallback_mask):
            mapped_chunk[fallback_mask] = v[fallback_mask] + weighted_disp[fallback_mask]

        if np.any(exact_mask):
            mapped_chunk[exact_mask] = control_to[nearest_idx[exact_mask]]

        mapped_points[start:end] = mapped_chunk

    return mapped_points


def point_guided_deformation(
    image: np.ndarray,
    source_pts: Union[List[List[float]], List[Tuple[float, float]], np.ndarray],
    target_pts: Union[List[List[float]], List[Tuple[float, float]], np.ndarray],
    alpha: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Return
    ------
        A deformed image.

    修改了一下函数结构,引入了typehint
    """

    warped_image = np.array(image)
    h, width = image.shape[:2]

    source_pts = np.array(source_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)

    if len(source_pts) == 0 or len(target_pts) == 0:
        return warped_image
    if len(source_pts) != len(target_pts):
        n = min(len(source_pts), len(target_pts))
        source_pts = source_pts[:n]
        target_pts = target_pts[:n]
        if n == 0:
            return warped_image

    # Anchor image corners to keep global shape stable. added by gpt - codex 5.3 
    corners = np.array(
        [[0, 0], [width - 1, 0], [0, h - 1], [width - 1, h - 1]],
        dtype=np.float32,
    )

    
    source_all = np.vstack([source_pts, corners])
    target_all = np.vstack([target_pts, corners])

    # Inverse mapping: compute all output pixels in target space and sample from source space.
    grid_y, grid_x = np.indices((h, width), dtype=np.float32)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    source_coords = _mls_affine_map_points(grid_points, target_all, source_all, alpha, eps)
    map_x = source_coords[:, 0].reshape(h, width)
    map_y = source_coords[:, 1].reshape(h, width)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return warped_image


def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))
    return warped_image


# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)


if __name__ == "__main__":
    demo.launch()
