"""
Generate training labels from 3D boxes in camera space.

Camera convention (OpenCV):
  +X: right
  +Y: down
  +Z: forward (depth); points with Z > 0 are in front of the camera

Projection:
  u = fx * X / Z + cx
  v = fy * Y / Z + cy

Box layout in camera space:
  center (cx, cy, cz); sizes (w, l, h) along local (x, y, z) before rotation;
  quaternion (qx, qy, qz, qw) maps local box frame -> camera frame.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Data layout: data/images/, data/labels/
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path("../data")
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "images"
DEFAULT_LABELS_DIR = DEFAULT_DATA_ROOT / "labels"
TRAIN_LABEL_SUFFIX = "_train"
INSTANCE_MASK_SUFFIX = "_instance_mask"


def is_raw_label_json(path: Union[str, Path]) -> bool:
    """True for capture JSON, excluding generated ``*_train.json`` / ``index.json``."""
    path = Path(path)
    if path.suffix != ".json" or path.stem.endswith(TRAIN_LABEL_SUFFIX):
        return False
    return path.name != "index.json"


def list_raw_label_jsons(
    labels_dir: Union[str, Path],
    pattern: str = "frame_*.json",
    *,
    recursive: bool = False,
) -> List[Path]:
    """List original annotation JSON files under ``labels_dir``."""
    root = Path(labels_dir)
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(p for p in iterator if is_raw_label_json(p))


def train_label_path(stem: str, labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR) -> Path:
    return Path(labels_dir) / f"{stem}{TRAIN_LABEL_SUFFIX}.json"


def instance_mask_path(stem: str, labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR) -> Path:
    return Path(labels_dir) / f"{stem}{INSTANCE_MASK_SUFFIX}.png"


def resolve_images_dir(
    json_path: Union[str, Path],
    images_dir: Union[str, Path, None] = None,
) -> Path:
    """Default ``data/images`` when JSON lives under ``data/labels``."""
    if images_dir is not None:
        return Path(images_dir)
    json_path = Path(json_path)
    if json_path.parent.name == "labels":
        return json_path.parent.parent / "images"
    if (json_path.parent.parent / "images").is_dir():
        return json_path.parent.parent / "images"
    return json_path.parent

# Local box axes: w -> x, l -> y, h -> z (camera frame when unrotated)
# Corner order: bottom face (z=-h/2) 0-3 CCW, top face (z=+h/2) 4-7 CCW
_BOX_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float64,
)

# 6 faces, each 4 vertex indices (CCW when viewed from outside)
_BOX_FACES = (
    (0, 1, 2, 3),  # bottom  (z-)
    (4, 7, 6, 5),  # top     (z+)
    (0, 4, 5, 1),  # -Y face (toward image top when unrotated)
    (2, 6, 7, 3),  # +Y face (toward image bottom when unrotated)
    (0, 3, 7, 4),  # left    (x-)
    (1, 5, 6, 2),  # right   (x+)
)


# ---------------------------------------------------------------------------
# Instance mask rendering (OpenGL + CPU fallback)
# ---------------------------------------------------------------------------
_GL_RENDERER: Any | None = None
_GL_UNAVAILABLE = False


def _opencv_to_gl(points: np.ndarray) -> np.ndarray:
    """OpenCV camera (+Y down, +Z fwd) -> OpenGL (+Y up, -Z fwd)."""
    out = np.empty_like(points, dtype=np.float32)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 1]
    out[..., 2] = -points[..., 2]
    return out


def intrinsics_to_gl_projection(
    K: np.ndarray,
    W: int,
    H: int,
    near: float,
    far: float,
) -> np.ndarray:
    """Row-major 4x4 projection matching pinhole ``_project_points``."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    n, f = float(near), float(far)
    return np.array(
        [
            [2.0 * fx / W, 0.0, 0.0, 0.0],
            [0.0, 2.0 * fy / H, 0.0, 0.0],
            [1.0 - 2.0 * cx / W, 2.0 * cy / H - 1.0, -(f + n) / (f - n), -1.0],
            [0.0, 0.0, -2.0 * f * n / (f - n), 0.0],
        ],
        dtype=np.float32,
    )


def _id_to_rgb(label: int) -> tuple[float, float, float]:
    r = (label & 0xFF) / 255.0
    g = ((label >> 8) & 0xFF) / 255.0
    b = ((label >> 16) & 0xFF) / 255.0
    return r, g, b


def _rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return r | (g << 8) | (b << 16)


def _box_triangles_gl(corners_cam: np.ndarray) -> np.ndarray:
    """(12, 3, 3) triangle vertices in OpenGL camera coordinates."""
    corners_gl = _opencv_to_gl(corners_cam)
    box_center = corners_gl.mean(axis=0)
    tris: list[np.ndarray] = []

    for face in _BOX_FACES:
        face_gl = corners_gl[list(face)]
        face_center = face_gl.mean(axis=0)
        outward = face_center - box_center
        for i0, i1, i2 in ((0, 1, 2), (0, 2, 3)):
            tri = face_gl[[i0, i1, i2]]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if float(np.dot(normal, outward)) < 0.0:
                tri = tri[[0, 2, 1]]
            tris.append(tri.astype(np.float32))

    return np.stack(tris, axis=0)


def box_triangles_cam(corners_cam: np.ndarray) -> np.ndarray:
    """(12, 3, 3) OpenCV-camera triangles matching the OpenGL mesh."""
    tris_gl = _box_triangles_gl(corners_cam)
    tris_cv = tris_gl.copy()
    tris_cv[..., 1] *= -1.0
    tris_cv[..., 2] *= -1.0
    return tris_cv.astype(np.float64)


def _project_cam_points(K: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project (..., 3) OpenCV camera points -> (..., 3) as u, v, z."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = pts[..., 2]
    z_safe = np.maximum(z, 1e-8)
    u = fx * pts[..., 0] / z_safe + cx
    v = fy * pts[..., 1] / z_safe + cy
    return np.stack([u, v, z], axis=-1)


def _render_instance_mask_cpu(
    corners_cam: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    bg_label: int = 255,
) -> np.ndarray:
    """Vectorized software z-buffer rasterizer (no OpenGL)."""
    n = corners_cam.shape[0]
    mask_buf = np.full((H, W), bg_label, dtype=np.uint8)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    if n == 0:
        return mask_buf

    splats: list[tuple[float, int, np.ndarray]] = []
    for box_i in range(n):
        tris = box_triangles_cam(corners_cam[box_i])
        label = box_i + 1
        for tri in tris:
            if np.any(tri[:, 2] <= 1e-6):
                continue
            splats.append((float(tri[:, 2].mean()), label, tri))

    splats.sort(key=lambda item: item[0], reverse=True)

    for _mean_z, label, tri in splats:
        uvz = _project_cam_points(K, tri)
        xs = uvz[:, 0]
        ys = uvz[:, 1]
        zs = uvz[:, 2]
        if np.any(zs <= 1e-6):
            continue

        x_min = max(int(np.floor(xs.min())), 0)
        x_max = min(int(np.ceil(xs.max())), W - 1)
        y_min = max(int(np.floor(ys.min())), 0)
        y_max = min(int(np.ceil(ys.max())), H - 1)
        if x_min > x_max or y_min > y_max:
            continue

        x0, y0, z0 = xs[0], ys[0], zs[0]
        x1, y1, z1 = xs[1], ys[1], zs[1]
        x2, y2, z2 = xs[2], ys[2], zs[2]
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-12:
            continue

        uu = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
        vv = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
        px, py = np.meshgrid(uu, vv)
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
        w2 = 1.0 - w0 - w1
        z = w0 * z0 + w1 * z1 + w2 * z2
        closer = (
            (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            & (z > 1e-6)
            & (z < depth_buf[y_min:y_max + 1, x_min:x_max + 1])
        )
        if not np.any(closer):
            continue
        depth_buf[y_min:y_max + 1, x_min:x_max + 1][closer] = z[closer]
        mask_buf[y_min:y_max + 1, x_min:x_max + 1][closer] = label

    return mask_buf


class InstanceMaskRenderer:
    """Render per-pixel box ids with OpenGL depth testing."""

    def __init__(self) -> None:
        import moderngl

        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 u_proj;
                in vec3 in_pos;
                void main() {
                    gl_Position = u_proj * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 u_id_rgb;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(u_id_rgb, 1.0);
                }
            """,
        )
        self._fbo = None
        self._fbo_size: tuple[int, int] = (0, 0)

    def _ensure_fbo(self, W: int, H: int) -> None:
        if self._fbo is not None and self._fbo_size == (W, H):
            return
        if self._fbo is not None:
            self._fbo.release()
        color = self.ctx.texture((W, H), 4, dtype="f4")
        depth = self.ctx.depth_texture((W, H))
        self._fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
        self._fbo_size = (W, H)

    def render(
        self,
        corners_cam: np.ndarray,
        K: np.ndarray,
        W: int,
        H: int,
        bg_label: int = 255,
    ) -> np.ndarray:
        n = corners_cam.shape[0]
        mask_buf = np.full((H, W), bg_label, dtype=np.uint8)
        if n == 0:
            return mask_buf

        z_cv = corners_cam[..., 2]
        valid = z_cv > 1e-6
        if not np.any(valid):
            return mask_buf

        near = max(0.05, float(z_cv[valid].min()) * 0.5)
        far = max(float(z_cv[valid].max()) * 2.0, near + 50.0)
        proj = intrinsics_to_gl_projection(K, W, H, near, far)

        self._ensure_fbo(W, H)
        assert self._fbo is not None
        self._fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.CULL_FACE)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        self.prog["u_proj"].write(np.ascontiguousarray(proj).tobytes())

        for box_i in range(n):
            if not np.any(z_cv[box_i] > 0):
                continue
            tris = _box_triangles_gl(corners_cam[box_i]).reshape(-1, 3)
            vbo = self.ctx.buffer(tris.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f", "in_pos")])
            label = box_i + 1
            self.prog["u_id_rgb"].value = _id_to_rgb(label)
            vao.render(mode=self.ctx.TRIANGLES)
            vao.release()
            vbo.release()

        data = self._fbo.read(components=4)
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4)
        rgba = rgba[::-1, :, :3]
        ids = _rgb_to_id(rgba)
        visible = ids > 0
        mask_buf[visible] = np.clip(ids[visible], 0, 255).astype(np.uint8)
        return mask_buf


def reset_gl_renderer() -> None:
    """Drop the process-local EGL context (needed after DataLoader fork)."""
    global _GL_RENDERER, _GL_UNAVAILABLE
    _GL_RENDERER = None
    _GL_UNAVAILABLE = False


def render_instance_mask_buf(
    corners_cam: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    bg_label: int = 255,
) -> np.ndarray:
    global _GL_RENDERER, _GL_UNAVAILABLE
    if not _GL_UNAVAILABLE:
        try:
            if _GL_RENDERER is None:
                _GL_RENDERER = InstanceMaskRenderer()
            return _GL_RENDERER.render(corners_cam, K, W, H, bg_label=bg_label)
        except Exception as exc:
            _GL_RENDERER = None
            _GL_UNAVAILABLE = True
            print(f"OpenGL instance-mask renderer unavailable ({exc}); using CPU rasterizer")
    return _render_instance_mask_cpu(corners_cam, K, W, H, bg_label=bg_label)


def _box_corners_cam(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: (n, 10) -> cx, cy, cz, w, l, h, qx, qy, qz, qw
    returns: (n, 8, 3) corner coordinates in camera frame
    """
    n = boxes.shape[0]
    corners = np.zeros((n, 8, 3), dtype=np.float64)
    half = boxes[:, 3:6] * 0.5  # w, l, h
    local = _BOX_CORNER_SIGNS[None, :, :] * half[:, None, :]
    quats = boxes[:, 6:10]
    centers = boxes[:, 0:3]
    for i in range(n):
        rot = Rotation.from_quat(quats[i]).as_matrix()
        corners[i] = local[i] @ rot.T + centers[i]
    return corners


def _project_points(K: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    pts: (..., 3) camera coordinates
    returns:
        uv: (..., 2) pixel coordinates
        depth: (...) positive depth in camera frame (Z)
    """
    z = pts[..., 2]
    x = pts[..., 0] / np.maximum(z, 1e-8)
    y = pts[..., 1] / np.maximum(z, 1e-8)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return np.stack([u, v], axis=-1), z


def _face_visible(face_idx: np.ndarray, corners_cam: np.ndarray) -> bool:
    """True if the outward face normal points toward the camera at the origin."""
    face_pts = corners_cam[face_idx]
    centroid = face_pts.mean(axis=0)
    if centroid[2] <= 0:
        return False

    v0, v1, v2, _v3 = face_pts
    normal = np.cross(v1 - v0, v2 - v0)
    # Orient normal outward (away from box center), independent of vertex winding.
    box_center = corners_cam.mean(axis=0)
    outward = centroid - box_center
    if float(np.dot(normal, outward)) < 0:
        normal = -normal
    return float(np.dot(normal, -centroid)) > 0


def _ray_triangle_t(
    origin: np.ndarray,
    target: np.ndarray,
    tri: np.ndarray,
) -> float | None:
    """Return t on ray origin + t*(target-origin) where the segment hits the triangle."""
    direction = target - origin
    edge1 = tri[1] - tri[0]
    edge2 = tri[2] - tri[0]
    h = np.cross(direction, edge2)
    det = float(np.dot(edge1, h))
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det  # signed: hits both triangle sides (no back-face cull)
    s = origin - tri[0]
    u = inv_det * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = inv_det * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return None
    t = inv_det * float(np.dot(edge2, q))
    return t


def _ray_triangles_t(
    origin: np.ndarray,
    target: np.ndarray,
    tris: np.ndarray,
) -> np.ndarray:
    """Batched ``_ray_triangle_t``; ``nan`` where the ray misses."""
    if tris.size == 0:
        return np.zeros((0,), dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    edge1 = tris[:, 1] - tris[:, 0]
    edge2 = tris[:, 2] - tris[:, 0]
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    parallel = np.abs(det) < 1e-12
    inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=~parallel)
    s = np.asarray(origin, dtype=np.float64) - tris[:, 0]
    u = inv_det * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv_det * (q @ direction)
    t = inv_det * np.einsum("ij,ij->i", edge2, q)
    hit = (~parallel) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
    out = np.full(tris.shape[0], np.nan, dtype=np.float64)
    out[hit] = t[hit]
    return out


def _all_box_triangles(all_corners_cam: np.ndarray) -> np.ndarray:
    if all_corners_cam.shape[0] == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.concatenate(
        [box_triangles_cam(box_corners) for box_corners in all_corners_cam],
        axis=0,
    )


def _point_visible_from_camera(
    point: np.ndarray,
    all_corners_cam: np.ndarray,
    eps: float = 1e-6,
    *,
    tris: np.ndarray | None = None,
) -> bool:
    """
    A 3D point is visible iff the segment camera->point does not pierce any
    box triangle (either side) strictly before reaching the point.

    Uses the same triangle mesh as OpenGL instance-mask rendering.
    """
    if point[2] <= eps:
        return False
    origin = np.zeros(3, dtype=np.float64)
    if tris is None:
        tris = _all_box_triangles(all_corners_cam)
    t = _ray_triangles_t(origin, point, tris)
    return not bool(np.any((t > eps) & (t < 1.0 - eps)))


_BG_LABEL = 255

# Keypoint layout: (k, 8, 3) with channels (x, y, v); v=0 valid, v=1 invalid.
_KP_DIM = 3
_KP_VALID = 0.0
_KP_INVALID = 1.0


def kp_corner_valid(kp: np.ndarray) -> bool:
    """True when the corner is valid (v == 0). Accepts legacy (x, y) arrays."""
    kp = np.asarray(kp, dtype=np.float32).reshape(-1)
    if kp.size >= 3:
        return float(kp[2]) == _KP_VALID
    return float(kp[0]) >= 0.0 and float(kp[1]) >= 0.0


def kp_xy(kp: np.ndarray) -> tuple[float, float]:
    """Return pixel (x, y) from a corner row."""
    kp = np.asarray(kp, dtype=np.float32).reshape(-1)
    return float(kp[0]), float(kp[1])


def ensure_kps_xyv(kps: np.ndarray) -> np.ndarray:
    """Normalize keypoints to (k, 8, 3); upgrade legacy (k, 8, 2) on load."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.size == 0:
        return np.zeros((0, 8, _KP_DIM), dtype=np.float32)
    if kps.ndim == 3 and kps.shape[-1] == _KP_DIM:
        return kps
    if kps.ndim == 3 and kps.shape[-1] == 2:
        out = np.zeros((kps.shape[0], 8, _KP_DIM), dtype=np.float32)
        out[..., 2] = _KP_INVALID
        for i in range(kps.shape[0]):
            for c in range(8):
                x, y = kps[i, c]
                if x >= 0.0 and y >= 0.0:
                    out[i, c, 0] = x
                    out[i, c, 1] = y
                    out[i, c, 2] = _KP_VALID
        return out
    raise ValueError(f"expected kps shape (k, 8, 2|3), got {kps.shape}")

BOX_TYPE_TO_ID: Dict[str, int] = {
    "A1": 0,
    "A2": 1,
    "A3": 2,
    "B1": 3,
    "B2": 4,
    "B3": 5,
}


def intrinsics_from_camera(camera: Mapping[str, Any]) -> Tuple[np.ndarray, int, int]:
    """Build K and image size from a frame JSON ``camera`` block."""
    if "intrinsic_matrix" in camera:
        K = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
        W = int(camera["image_width"])
        H = int(camera["image_height"])
        return K, W, H

    W = int(camera["width"])
    H = int(camera["height"])
    K = np.array(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K, W, H


def nyx_capture_fov_degrees(K: np.ndarray, W: int, H: int) -> float:
    """
    Horizontal FOV (degrees) passed to ``nyx.camera_options(fov=...)`` during capture.

    Matches ``box_tetris_truck_nyx_mujoco.py``:
    ``fov_x = 2 * atan(width / (2 * fx))``.
    """
    fx = float(K[0, 0])
    return math.degrees(2.0 * math.atan(W / (2.0 * fx)))


def nyx_vertical_fov_degrees(K: np.ndarray, W: int, H: int) -> float:
    """
    Vertical FOV (degrees) — only if Nyx ``fov`` were vertical (not used in capture).
    """
    fy = float(K[1, 1])
    return math.degrees(2.0 * math.atan(H / (2.0 * fy)))


def world_to_camera_from_pose(
    position: Union[Sequence[float], np.ndarray],
    lookat: Union[Sequence[float], np.ndarray],
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    4x4 world-to-camera matrix matching ``box_tetris_truck_nyx_mujoco`` capture.

    Camera axes: x=right, y=up, z=forward (before OpenCV y-flip).
    """
    position = np.asarray(position, dtype=np.float64).reshape(3)
    target = np.asarray(lookat, dtype=np.float64).reshape(3)
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_ref = np.asarray(world_up, dtype=np.float64)
    right = np.cross(forward, up_ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.stack((right, up, forward), axis=0)
    mat[:3, 3] = -mat[:3, :3] @ position
    return mat


def _world_to_camera_matrix(frame: Mapping[str, Any]) -> np.ndarray:
    """Prefer pose-built matrix (capture logic); fall back to stored JSON."""
    camera = frame.get("camera") or {}
    if "position" in camera and "lookat" in camera:
        return world_to_camera_from_pose(camera["position"], camera["lookat"])
    return np.asarray(camera["world_to_camera"], dtype=np.float64).reshape(4, 4)


def intrinsics_matching_nyx_render(
    K: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Pinhole K that matches Nyx-rendered RGB.

    Capture passes ``fov_x = 2*atan(W/(2*fx))`` to ``nyx.camera_options(fov=...)``,
    but Nyx treats ``fov`` as vertical. The effective focal lengths both become
    ``f_eff = fx * H / W`` (aspect ratio H:W); see Nyx vertical-FOV geometry.
    """
    fx = float(K[0, 0])
    f_eff = fx * H / W
    K_out = np.asarray(K, dtype=np.float64).copy()
    K_out[0, 0] = f_eff
    K_out[1, 1] = f_eff
    return K_out


def _center_principal_point(K: np.ndarray, W: int, H: int) -> np.ndarray:
    """Set ``cx, cy`` to the image center (pixel coordinates)."""
    K_out = np.asarray(K, dtype=np.float64).copy()
    K_out[0, 2] = W / 2.0
    K_out[1, 2] = H / 2.0
    return K_out


def intrinsics_for_projection(
    camera: Mapping[str, Any],
    frame: Mapping[str, Any] | None = None,
    *,
    projection_intrinsics: str | None = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Pinhole K for projecting labels onto captured RGB.

    ``projection_intrinsics``:
      - ``"pinhole"``: use JSON ``intrinsic_matrix`` as-is (centered principal point)
      - ``"nyx"``: apply ``intrinsics_matching_nyx_render`` (legacy Nyx vertical-FOV capture)
      - ``None``: use ``camera["nyx_fov"]`` when set; otherwise ``"nyx"`` for genesis
        frames (legacy default) and ``"pinhole"`` otherwise.
    """
    K, W, H = intrinsics_from_camera(camera)
    mode = projection_intrinsics
    if mode is None:
        explicit = str(camera.get("nyx_fov", "")).lower()
        if explicit in ("pinhole", "nyx"):
            mode = explicit
        elif frame is not None and _is_genesis_frame(frame):
            mode = "nyx"
        else:
            mode = "pinhole"
    mode = str(mode).lower()
    if mode == "nyx":
        K = intrinsics_matching_nyx_render(K, W, H)
    elif mode != "pinhole":
        raise ValueError(f"unknown projection_intrinsics={projection_intrinsics!r}")
    K = _center_principal_point(K, W, H)
    return K, W, H


def _is_genesis_frame(frame: Mapping[str, Any]) -> bool:
    boxes = frame.get("boxes") or []
    if not boxes:
        return "world_to_camera" in frame.get("camera", {})
    first = boxes[0]
    return "cx" in first and "world_to_camera" in frame.get("camera", {})


def _box_corners_world(box: Mapping[str, Any]) -> np.ndarray:
    """8 corners of one box in world coordinates."""
    center = np.array([box["cx"], box["cy"], box["cz"]], dtype=np.float64)
    half = np.array([box["w"], box["l"], box["h"]], dtype=np.float64) * 0.5
    local = _BOX_CORNER_SIGNS * half
    rot = Rotation.from_quat([
        float(box["qx"]), float(box["qy"]), float(box["qz"]), float(box["qw"]),
    ]).as_matrix()
    return local @ rot.T + center


def _apply_world_to_camera(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 world_to_camera to (N, 3) points."""
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (pts_h @ M.T)[:, :3]


def _genesis_to_opencv(pts: np.ndarray) -> np.ndarray:
    """
    Convert genesis camera coords to OpenCV (+X right, +Y down, +Z forward).

    JSON ``world_to_camera`` uses +Y up; OpenCV uses +Y down.
    """
    out = np.asarray(pts, dtype=np.float64).copy()
    out[..., 1] *= -1.0
    return out


def _world_to_opencv_cam(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """World points -> OpenCV camera frame."""
    return _genesis_to_opencv(_apply_world_to_camera(M, pts))


def corners_cam_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    """(n, 8, 3) box corners in OpenCV camera frame."""
    M = _world_to_camera_matrix(frame)
    corners = [_world_to_opencv_cam(M, _box_corners_world(box)) for box in frame["boxes"]]
    if not corners:
        return np.zeros((0, 8, 3), dtype=np.float64)
    return np.stack(corners, axis=0)


def boxes_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    """
    Convert frame JSON boxes to (n, 10): cx,cy,cz,w,l,h,qx,qy,qz,qw in camera frame.

    Supports:
      - genesis format: world boxes + ``camera.world_to_camera`` (corner transform)
      - legacy format: camera-frame ``position`` / ``sizes`` / ``quaternion``
    """
    boxes = frame.get("boxes") or []
    if not boxes:
        return np.zeros((0, 10), dtype=np.float64)

    if _is_genesis_frame(frame):
        M = _world_to_camera_matrix(frame)
        rows: List[List[float]] = []
        for box in boxes:
            corners_c = _world_to_opencv_cam(M, _box_corners_world(box))
            center = corners_c.mean(axis=0)
            half = np.array([box["w"], box["l"], box["h"]], dtype=np.float64) * 0.5
            rows.append([
                float(center[0]), float(center[1]), float(center[2]),
                float(half[0] * 2), float(half[2] * 2), float(half[1] * 2),
                float(box["qx"]), float(box["qy"]), float(box["qz"]), float(box["qw"]),
            ])
        return np.asarray(rows, dtype=np.float64)

    rows = []
    for box in boxes:
        center = box["position"]
        sizes = box["sizes"]
        quat = box["quaternion"]
        rows.append([
            float(center[0]), float(center[1]), float(center[2]),
            float(sizes[0]), float(sizes[1]), float(sizes[2]),
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        ])
    return np.asarray(rows, dtype=np.float64)


def labels_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    """Map box ``type`` strings to class ids; unknown / missing -> 0."""
    return np.asarray(
        [BOX_TYPE_TO_ID.get(str(box.get("type", "")), 0) for box in frame.get("boxes", [])],
        dtype=np.int64,
    )


def load_frame_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_path_for_frame(
    json_path: Union[str, Path],
    frame: Mapping[str, Any],
    *,
    images_dir: Union[str, Path, None] = None,
) -> Path:
    json_path = Path(json_path)
    images_dir = resolve_images_dir(json_path, images_dir)
    name = frame.get("image") or frame.get("frame") or f"{json_path.stem}.png"
    candidate = images_dir / Path(name).name
    if candidate.is_file():
        return candidate
    legacy = json_path.with_name(Path(name).name)
    if legacy.is_file():
        return legacy
    png = json_path.with_suffix(".png")
    if png.is_file():
        return png
    raise FileNotFoundError(
        f"No image found for {json_path} (tried {candidate}, {legacy}, {png})"
    )


def _frame_index_from_path(json_path: Path, frame: Mapping[str, Any]) -> int:
    if "frame_index" in frame:
        return int(frame["frame_index"])
    stem = json_path.stem
    if stem.startswith("frame_") and stem[6:].isdigit():
        return int(stem[6:]) - 1
    return -1


def gen_train_sample_from_json(
    json_path: Union[str, Path],
    *,
    image_path: Union[str, Path, None] = None,
    images_dir: Union[str, Path, None] = None,
    projection_intrinsics: str | None = None,
) -> Dict[str, Any]:
    """
    Load a frame JSON (+ optional RGB path) and generate aligned training labels.

    Returns a dict with:
      instance_mask: PIL 'P' image (bg=255, instances=0..k-1)
      kps: (k, 8, 3) projected corners as (x, y, v); v=0 valid, v=1 invalid
      box_indices: (k,) source box index per visible instance
      labels: (n,) class id for every input box
      inst_labels: (k,) class id per visible instance
      boxes_cam: (n, 10) camera-frame 3D boxes
      K, W, H, image_path, frame
    """
    json_path = Path(json_path)
    frame = load_frame_json(json_path)
    K, W, H = intrinsics_for_projection(
        frame["camera"], frame, projection_intrinsics=projection_intrinsics,
    )
    boxes = boxes_from_frame(frame)
    labels = labels_from_frame(frame)
    corners_cam = corners_cam_from_frame(frame) if _is_genesis_frame(frame) else None

    instance_mask, kps, box_indices = gen_train_sample(
        K, W, H, boxes, corners_cam=corners_cam,
    )
    inst_labels = labels[box_indices] if box_indices.size else np.zeros(0, dtype=np.int64)

    resolved_image = (
        Path(image_path) if image_path is not None
        else image_path_for_frame(json_path, frame, images_dir=images_dir)
    )
    image_name = frame.get("image") or frame.get("frame", json_path.stem + ".png")

    return {
        "instance_mask": instance_mask,
        "kps": kps,
        "box_indices": box_indices,
        "labels": labels,
        "inst_labels": inst_labels,
        "boxes_cam": boxes,
        "K": K,
        "W": W,
        "H": H,
        "image_path": resolved_image,
        "frame": image_name,
        "frame_index": _frame_index_from_path(json_path, frame),
        "projection_intrinsics": projection_intrinsics,
    }


def gen_train_samples_from_dir(
    labels_dir: Union[str, Path],
    pattern: str = "frame_*.json",
) -> List[Dict[str, Any]]:
    """Generate training labels for every raw annotation JSON under ``labels_dir``."""
    return [
        gen_train_sample_from_json(path)
        for path in list_raw_label_jsons(labels_dir, pattern)
    ]


def load_train_sample_from_labels(
    stem: str,
    labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR,
    *,
    images_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Load a generated training sample from ``labels/{stem}_train.json``."""
    labels_dir = Path(labels_dir)
    train_path = train_label_path(stem, labels_dir)
    if not train_path.is_file():
        raise FileNotFoundError(train_path)

    with train_path.open(encoding="utf-8") as f:
        train = json.load(f)

    mask_name = train.get("instance_mask") or f"{stem}{INSTANCE_MASK_SUFFIX}.png"
    mask_path = labels_dir / mask_name
    mask = Image.open(mask_path)

    kps = ensure_kps_xyv(np.asarray(train["kps"], dtype=np.float32))
    box_indices = np.asarray(train["box_indices"], dtype=np.int64)
    inst_labels = np.asarray(train.get("inst_labels", train.get("labels", [])), dtype=np.int64)
    K = np.asarray(train["K"], dtype=np.float64)
    W, H = int(train["W"]), int(train["H"])

    raw_json = labels_dir / f"{stem}.json"
    frame = load_frame_json(raw_json) if raw_json.is_file() else {}
    image_name = train.get("image") or frame.get("image") or f"{stem}.png"
    if images_dir is None:
        images_dir = resolve_images_dir(raw_json if raw_json.is_file() else labels_dir / f"{stem}.json")
    image_path = Path(images_dir) / Path(image_name).name

    return {
        "instance_mask": mask,
        "kps": kps,
        "box_indices": box_indices,
        "inst_labels": inst_labels,
        "labels": inst_labels,
        "K": K,
        "W": W,
        "H": H,
        "image_path": image_path,
        "frame": image_name,
        "frame_index": int(train.get("frame_index", -1)),
        "train_json": train_path,
        "source_json": raw_json if raw_json.is_file() else None,
    }


def mask_id_to_boxes_xyxy(mask: np.ndarray, num_instances: int) -> List[List[int]]:
    """Inclusive pixel xyxy tight AABB for instance ids ``0 .. k-1``."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    boxes: List[List[int]] = []
    for i in range(int(num_instances)):
        ys, xs = np.where(mask == i)
        if xs.size == 0:
            boxes.append([0, 0, 0, 0])
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


def copy_image_as_jpg(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    quality: int = 95,
) -> Path:
    """Write an RGB JPEG copy of ``src`` to ``dst``."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.save(dst, format="JPEG", quality=int(quality))
    return dst


def save_train_sample(
    sample: Mapping[str, Any],
    labels_dir: Union[str, Path],
    *,
    stem: str | None = None,
    source_json: Union[str, Path, None] = None,
) -> Path:
    """
    Persist generated training labels under ``labels_dir``.

    Files:
      {stem}_instance_mask.png  P-mode id map (bg=255)
      {stem}_train.json         det / seg / kp payloads plus ``has_*`` flags
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = Path(str(sample.get("frame", "sample"))).stem

    mask_filename = f"{stem}{INSTANCE_MASK_SUFFIX}.png"
    sample["instance_mask"].save(labels_dir / mask_filename)

    mask_arr = np.asarray(sample["instance_mask"], dtype=np.uint8)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    kps = ensure_kps_xyv(np.asarray(sample["kps"], dtype=np.float32))
    k = int(kps.shape[0])
    inst_labels = np.asarray(sample["inst_labels"], dtype=np.int64).tolist()
    box_indices = np.asarray(sample["box_indices"], dtype=np.int64).tolist()
    boxes_xyxy = mask_id_to_boxes_xyxy(mask_arr, k)
    has_seg = k > 0 and bool(np.any(mask_arr != _BG_LABEL))
    has_det = any((b[2] > b[0]) and (b[3] > b[1]) for b in boxes_xyxy)
    has_kp = k > 0
    image_name = Path(sample["image_path"]).name

    train = {
        "frame": sample.get("frame") or image_name,
        "frame_index": int(sample.get("frame_index", -1)),
        "image": image_name,
        "W": int(sample["W"]),
        "H": int(sample["H"]),
        "K": np.asarray(sample["K"], dtype=np.float64).tolist(),
        "num_instances": k,
        "has_det": bool(has_det),
        "has_seg": bool(has_seg),
        "has_kp": bool(has_kp),
        "det": {
            "boxes": boxes_xyxy,
            "labels": inst_labels,
        },
        "seg": {
            "instance_mask": mask_filename,
        },
        "kp": {
            "kps": kps.tolist(),
        },
        "instance_mask": mask_filename,
        "kps": kps.tolist(),
        "inst_labels": inst_labels,
        "box_indices": box_indices,
        "source_json": Path(source_json).name if source_json is not None else f"{stem}.json",
    }
    if sample.get("projection_intrinsics"):
        train["projection_intrinsics"] = sample["projection_intrinsics"]
    train_path = train_label_path(stem, labels_dir)
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
        f.write("\n")

    return train_path


def save_train_samples(
    samples: List[Mapping[str, Any]],
    labels_dir: Union[str, Path],
    *,
    source_jsons: List[Union[str, Path]] | None = None,
) -> Path:
    """Save multiple samples and write ``labels_dir/index.json`` manifest."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        src = source_jsons[i] if source_jsons is not None else None
        stem = Path(str(sample.get("frame", f"sample_{i}"))).stem
        train_path = save_train_sample(sample, labels_dir, stem=stem, source_json=src)
        entries.append({
            "stem": stem,
            "train_json": train_path.name,
            "instance_mask": f"{stem}{INSTANCE_MASK_SUFFIX}.png",
            "source_json": Path(src).name if src is not None else f"{stem}.json",
            "frame": sample.get("frame"),
            "frame_index": int(sample.get("frame_index", -1)),
            "num_instances": int(sample["kps"].shape[0]),
        })

    index_path = labels_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"samples": entries}, f, indent=2)
        f.write("\n")
    return index_path


def gen_and_save_from_json(
    json_path: Union[str, Path],
    labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR,
    *,
    image_path: Union[str, Path, None] = None,
) -> Path:
    """Generate labels from one JSON file and save to ``labels_dir``."""
    json_path = Path(json_path)
    sample = gen_train_sample_from_json(json_path, image_path=image_path)
    return save_train_sample(sample, labels_dir, stem=json_path.stem, source_json=json_path)


def gen_and_save_from_dir(
    labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR,
    pattern: str = "frame_*.json",
) -> Path:
    """Generate and save all raw frame JSON annotations under ``labels_dir``."""
    labels_dir = Path(labels_dir)
    json_paths = list_raw_label_jsons(labels_dir, pattern)
    samples = [gen_train_sample_from_json(path) for path in json_paths]
    return save_train_samples(samples, labels_dir, source_jsons=json_paths)


def _even_indices(n: int, limit: int) -> set[int]:
    """Evenly spaced indices in ``[0, n)``, at most ``limit`` of them."""
    if n <= 0 or limit <= 0:
        return set()
    if limit >= n:
        return set(range(n))
    if limit == 1:
        return {0}
    return {int(round(i * (n - 1) / (limit - 1))) for i in range(limit)}


def _write_viz_grid(saved: Sequence[Path], out_dir: Path) -> Path | None:
    """Thumbnail grid of viz PNGs under ``out_dir/all_frames_grid.png``."""
    from PIL import ImageDraw, ImageFont

    if not saved:
        return None
    thumb_w, thumb_h, cols = 480, 384, 6
    rows = (len(saved) + cols - 1) // cols
    cell_w, cell_h = thumb_w, thumb_h + 24
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (32, 32, 32))
    draw_font = ImageFont.load_default()
    for i, p in enumerate(saved):
        r, c = divmod(i, cols)
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), (24, 24, 24))
        ox = (cell_w - im.width) // 2
        canvas.paste(im, (ox, 0))
        d = ImageDraw.Draw(canvas)
        d.text((8, thumb_h + 4), p.stem.replace("_viz", ""), fill=(220, 220, 220), font=draw_font)
        grid.paste(canvas, (c * cell_w, r * cell_h))
    grid_path = out_dir / "all_frames_grid.png"
    grid.save(grid_path)
    return grid_path


def _sample_flags_from_mask(mask_arr: np.ndarray, k: int) -> Tuple[bool, bool, bool]:
    boxes_xyxy = mask_id_to_boxes_xyxy(mask_arr, k)
    has_seg = k > 0 and bool(np.any(mask_arr != _BG_LABEL))
    has_det = any((b[2] > b[0]) and (b[3] > b[1]) for b in boxes_xyxy)
    has_kp = k > 0
    return bool(has_det), bool(has_seg), bool(has_kp)


def _entry_from_existing(
    json_path: Path,
    in_root: Path,
    out_root: Path,
) -> Dict[str, Any] | None:
    """Build an index entry from an already-written train sample, or None."""
    rel = json_path.relative_to(in_root)
    sample_dir = out_root / rel.parent
    stem = json_path.stem
    train_path = train_label_path(stem, sample_dir)
    jpg_path = sample_dir / f"{stem}.jpg"
    mask_path = sample_dir / f"{stem}{INSTANCE_MASK_SUFFIX}.png"
    if not (train_path.is_file() and jpg_path.is_file() and mask_path.is_file()):
        return None
    with train_path.open(encoding="utf-8") as f:
        train = json.load(f)
    return {
        "stem": stem,
        "rel_dir": str(rel.parent).replace("\\", "/"),
        "image": jpg_path.name,
        "train_json": train_path.name,
        "instance_mask": mask_path.name,
        "source_json": json_path.name,
        "num_instances": int(train.get("num_instances", 0)),
        "has_det": bool(train.get("has_det", False)),
        "has_seg": bool(train.get("has_seg", False)),
        "has_kp": bool(train.get("has_kp", False)),
    }


def _process_tree_sample(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker: generate one mirrored train sample (+ optional viz)."""
    (
        index,
        json_path_s,
        in_root_s,
        out_root_s,
        jpg_quality,
        do_viz,
        viz_dir_s,
        skip_existing,
        projection_intrinsics,
    ) = args
    json_path = Path(json_path_s)
    in_root = Path(in_root_s)
    out_root = Path(out_root_s)
    rel = json_path.relative_to(in_root)
    sample_dir = out_root / rel.parent
    stem = json_path.stem

    try:
        if skip_existing:
            entry = _entry_from_existing(json_path, in_root, out_root)
            if entry is not None:
                viz_path_s = None
                if do_viz and viz_dir_s:
                    viz_name = f"{rel.parts[0]}_{stem}_viz.png" if rel.parts else f"{stem}_viz.png"
                    viz_path = Path(viz_dir_s) / viz_name
                    if not viz_path.is_file():
                        sample = load_train_sample_from_labels(
                            stem, sample_dir, images_dir=sample_dir,
                        )
                        sample["projection_intrinsics"] = projection_intrinsics
                        save_sample_visualization(
                            json_path,
                            viz_path,
                            image_path=sample_dir / f"{stem}.jpg",
                            sample=sample,
                            projection_intrinsics=projection_intrinsics,
                        )
                    viz_path_s = str(viz_path)
                return {
                    "index": index,
                    "entry": entry,
                    "viz_path": viz_path_s,
                    "skipped": True,
                    "error": None,
                    "rel": str(rel),
                }

        sample = gen_train_sample_from_json(
            json_path, projection_intrinsics=projection_intrinsics,
        )
        jpg_path = copy_image_as_jpg(
            sample["image_path"], sample_dir / f"{stem}.jpg", quality=jpg_quality,
        )
        sample["image_path"] = jpg_path
        sample["frame"] = jpg_path.name
        save_train_sample(sample, sample_dir, stem=stem, source_json=json_path)

        k = int(sample["kps"].shape[0])
        mask_arr = np.asarray(sample["instance_mask"], dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        has_det, has_seg, has_kp = _sample_flags_from_mask(mask_arr, k)
        entry = {
            "stem": stem,
            "rel_dir": str(rel.parent).replace("\\", "/"),
            "image": jpg_path.name,
            "train_json": f"{stem}{TRAIN_LABEL_SUFFIX}.json",
            "instance_mask": f"{stem}{INSTANCE_MASK_SUFFIX}.png",
            "source_json": json_path.name,
            "num_instances": k,
            "has_det": has_det,
            "has_seg": has_seg,
            "has_kp": has_kp,
        }

        viz_path_s = None
        if do_viz and viz_dir_s:
            viz_name = f"{rel.parts[0]}_{stem}_viz.png" if rel.parts else f"{stem}_viz.png"
            viz_path = Path(viz_dir_s) / viz_name
            save_sample_visualization(
                json_path,
                viz_path,
                image_path=jpg_path,
                sample=sample,
                projection_intrinsics=projection_intrinsics,
            )
            viz_path_s = str(viz_path)

        return {
            "index": index,
            "entry": entry,
            "viz_path": viz_path_s,
            "skipped": False,
            "error": None,
            "rel": str(rel),
        }
    except Exception as exc:
        return {
            "index": index,
            "entry": None,
            "viz_path": None,
            "skipped": False,
            "error": f"{type(exc).__name__}: {exc}",
            "rel": str(rel),
        }


def gen_and_save_from_tree(
    in_root: Union[str, Path],
    out_root: Union[str, Path],
    *,
    pattern: str = "*.json",
    jpg_quality: int = 95,
    viz_dir: Union[str, Path, None] = None,
    viz_limit: int = 24,
    workers: int = 1,
    skip_existing: bool = True,
    projection_intrinsics: str = "pinhole",
) -> Path:
    """
    Mirror a nested capture tree into train samples.

    For each raw JSON under ``in_root``, writes next to the mirrored path:
      {stem}.jpg, {stem}_instance_mask.png, {stem}_train.json
    and a root ``index.json`` listing every sample.

    ``workers > 1`` uses a process pool (CPU rasterization / I/O). Already-written
    samples are skipped when ``skip_existing`` is True.

    Default ``projection_intrinsics="pinhole"`` matches fill_container RGB captures;
    use ``"nyx"`` only for legacy Nyx vertical-FOV renders.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    in_root = Path(in_root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    json_paths = list_raw_label_jsons(in_root, pattern, recursive=True)
    if not json_paths:
        raise FileNotFoundError(f"No raw label JSON files under {in_root} (pattern={pattern!r})")

    viz_indices = _even_indices(len(json_paths), int(viz_limit)) if viz_dir else set()
    viz_out = Path(viz_dir) if viz_dir else None
    if viz_out is not None:
        viz_out.mkdir(parents=True, exist_ok=True)

    n = len(json_paths)
    workers = max(1, int(workers))
    if workers > 1:
        workers = min(workers, n, os.cpu_count() or workers)

    jobs = [
        (
            i,
            str(json_path),
            str(in_root),
            str(out_root),
            int(jpg_quality),
            i in viz_indices,
            str(viz_out) if viz_out is not None else None,
            bool(skip_existing),
            str(projection_intrinsics),
        )
        for i, json_path in enumerate(json_paths)
    ]

    results: List[Dict[str, Any] | None] = [None] * n
    done = 0
    skipped = 0
    errors: List[Dict[str, str]] = []

    def _consume(result: Dict[str, Any]) -> None:
        nonlocal done, skipped
        results[int(result["index"])] = result
        done += 1
        if result.get("skipped"):
            skipped += 1
        if result.get("error"):
            errors.append({"rel": str(result.get("rel", "")), "error": str(result["error"])})
            print(f"ERROR [{done}/{n}] {result.get('rel')}: {result['error']}")
            return
        entry = result["entry"]
        if result.get("viz_path"):
            print(f"Viz: {result['viz_path']} (k={entry['num_instances']})")
        if done % 50 == 0 or done == 1 or done == n:
            print(
                f"[{done}/{n}] {result['rel']} k={entry['num_instances']} "
                f"has_det={entry['has_det']} has_seg={entry['has_seg']} "
                f"has_kp={entry['has_kp']} skipped={skipped} workers={workers}"
            )

    print(
        f"Processing {n} samples with {workers} workers "
        f"(skip_existing={skip_existing}, projection_intrinsics={projection_intrinsics})"
    )
    if workers == 1:
        for job in jobs:
            _consume(_process_tree_sample(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_tree_sample, job) for job in jobs]
            for fut in as_completed(futures):
                _consume(fut.result())

    entries = [r["entry"] for r in results if r is not None and r.get("entry") is not None]
    viz_saved = [
        Path(r["viz_path"]) for r in results if r is not None and r.get("viz_path")
    ]

    index_path = out_root / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": entries,
                "errors": errors,
                "projection_intrinsics": projection_intrinsics,
            },
            f,
            indent=2,
        )
        f.write("\n")

    if viz_saved and viz_out is not None:
        grid_path = _write_viz_grid(viz_saved, viz_out)
        if grid_path is not None:
            print(f"Grid: {grid_path}")

    print(
        f"Saved {len(entries)} samples (skipped {skipped}, errors {len(errors)}), "
        f"index: {index_path}"
    )
    return index_path


_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# edge -> the two box faces that share it (for front/back classification)
_EDGE_TO_FACES: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
for _face in _BOX_FACES:
    _face_arr = np.asarray(_face, dtype=np.int32)
    for _a, _b in (
        (_face[0], _face[1]), (_face[1], _face[2]),
        (_face[2], _face[3]), (_face[3], _face[0]),
    ):
        _key = (_a, _b) if _a < _b else (_b, _a)
        if _key not in _EDGE_TO_FACES:
            _EDGE_TO_FACES[_key] = (_face_arr,)
        else:
            _EDGE_TO_FACES[_key] = (_EDGE_TO_FACES[_key][0], _face_arr)


def _point_visible_excluding_box(
    point: np.ndarray,
    all_corners_cam: np.ndarray,
    exclude_box_idx: int | None,
    eps: float = 1e-6,
) -> bool:
    """Like ``_point_visible_from_camera`` but ignore triangles from one box."""
    if point[2] <= eps:
        return False
    origin = np.zeros(3, dtype=np.float64)
    for bi, box_corners in enumerate(all_corners_cam):
        if bi == exclude_box_idx:
            continue
        for tri in box_triangles_cam(box_corners):
            t = _ray_triangle_t(origin, point, tri)
            if t is not None and eps < t < 1.0 - eps:
                return False
    return True


def _edge_is_solid(
    i: int,
    j: int,
    box_corners: np.ndarray,
    all_corners_cam: np.ndarray,
    box_index: int,
) -> bool:
    """True when the edge is front-facing and not occluded by other boxes."""
    key = (i, j) if i < j else (j, i)
    faces = _EDGE_TO_FACES.get(key, ())
    if not any(_face_visible(f, box_corners) for f in faces):
        return False
    mid = 0.5 * (box_corners[i] + box_corners[j])
    return _point_visible_excluding_box(mid, all_corners_cam, exclude_box_idx=box_index)


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_len: int = 10,
    gap_len: int = 7,
) -> None:
    """Draw a dashed segment with OpenCV."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1.0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        sx = int(round(x1 + dx * pos))
        sy = int(round(y1 + dy * pos))
        end = min(pos + dash_len, length)
        ex = int(round(x1 + dx * end))
        ey = int(round(y1 + dy * end))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += dash_len + gap_len


def _draw_visible_kps_on_bgr(
    bgr: np.ndarray,
    kps: np.ndarray,
    inst_id: int,
    *,
    W: int,
    H: int,
    color: tuple[int, int, int] = (0, 0, 255),
    point_radius: int = 7,
) -> np.ndarray:
    """Draw numbered keypoints for corners marked valid (v == 0)."""
    import cv2

    out = np.asarray(bgr)
    kps = ensure_kps_xyv(kps)
    for ci in range(8):
        if not kp_corner_valid(kps[inst_id, ci]):
            continue
        x, y = kp_xy(kps[inst_id, ci])
        c = (int(round(x)), int(round(y)))
        if not (0 <= c[0] < W and 0 <= c[1] < H):
            continue
        cv2.circle(out, c, point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, c, point_radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out, str(ci), (c[0] + 8, c[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, str(ci), (c[0] + 8, c[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
    return out

_VIZ_BOX_COLORS_BGR = (
    (0, 200, 255),
    (255, 80, 80),
    (80, 180, 255),
    (255, 200, 0),
    (220, 80, 255),
    (0, 220, 120),
)

_VIZ_MASK_COLORS_RGB = (
    (0, 220, 80),
    (255, 80, 80),
    (80, 160, 255),
    (255, 200, 0),
    (220, 80, 255),
    (0, 200, 200),
)


def overlay_wireframes_on_rgb(
    rgb: np.ndarray,
    frame: Mapping[str, Any],
    *,
    line_thickness: int = 2,
    point_radius: int = 5,
    projection_intrinsics: str | None = None,
) -> np.ndarray:
    """
    Draw 3D box wireframes on RGB (same projection as ``replay_box_capture``).

    Uses ``corners_cam_from_frame`` + ``intrinsics_for_projection``.
    """
    import cv2

    corners = corners_cam_from_frame(frame)
    if corners.shape[0] == 0:
        return cv2.cvtColor(np.asarray(rgb)[..., :3], cv2.COLOR_RGB2BGR)

    K, W, H = intrinsics_for_projection(
        frame["camera"], frame, projection_intrinsics=projection_intrinsics,
    )
    vis = np.asarray(rgb)
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    if vis.shape[-1] == 4:
        vis = vis[..., :3]
    out = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    for bi, box_corners in enumerate(corners):
        color = _VIZ_BOX_COLORS_BGR[bi % len(_VIZ_BOX_COLORS_BGR)]
        uv, depth = _project_points(K, box_corners)
        valid = depth > 1e-6
        for i, j in _BOX_EDGES:
            if not (valid[i] and valid[j]):
                continue
            p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
            if _edge_is_solid(i, j, box_corners, corners, bi):
                cv2.line(out, p1, p2, color, line_thickness, cv2.LINE_AA)
            else:
                _draw_dashed_line(out, p1, p2, color, line_thickness)
        for ci, ((u, v), ok) in enumerate(zip(uv, valid)):
            if not ok:
                continue
            if not _corner_visible(box_corners[ci], u, v, corners, W, H):
                continue
            c = (int(round(u)), int(round(v)))
            cv2.circle(out, c, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, c, point_radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                out, str(ci), (c[0] + 6, c[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return out


def overlay_mask_kps_on_bgr(
    bgr: np.ndarray,
    sample: Mapping[str, Any],
    *,
    mask_alpha: float = 0.45,
    instance_id: int | None = None,
) -> np.ndarray:
    """Tint ``instance_mask`` and draw ``kps`` on a BGR canvas."""
    import cv2

    out = np.asarray(bgr, dtype=np.float32).copy()
    mask = np.asarray(sample["instance_mask"], dtype=np.uint8)
    kps = ensure_kps_xyv(sample["kps"])
    W, H = int(sample["W"]), int(sample["H"])
    inst_ids = range(int(kps.shape[0]))
    if instance_id is not None:
        inst_ids = [instance_id]

    for inst_id in inst_ids:
        color = np.array(_VIZ_MASK_COLORS_RGB[inst_id % len(_VIZ_MASK_COLORS_RGB)], dtype=np.float32)
        region = mask == inst_id
        if np.any(region):
            out[region] = out[region] * (1.0 - mask_alpha) + color[::-1] * mask_alpha
            ys, xs = np.where(region)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            bgr_c = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(out, (x0, y0), (x1, y1), bgr_c, 2, cv2.LINE_AA)
        for ci in range(8):
            if not kp_corner_valid(kps[inst_id, ci]):
                continue
            x, y = kp_xy(kps[inst_id, ci])
            c = (int(round(x)), int(round(y)))
            if 0 <= c[0] < W and 0 <= c[1] < H:
                cv2.circle(out, c, 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(out, c, 6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    out, str(ci), (c[0] + 8, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
                )

    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_wireframe_for_box(
    bgr: np.ndarray,
    frame: Mapping[str, Any],
    box_index: int,
    *,
    line_thickness: int = 2,
    point_radius: int = 6,
    color: tuple[int, int, int] | None = None,
    kps: np.ndarray | None = None,
    inst_id: int | None = None,
) -> np.ndarray:
    """Draw wireframe + numbered corners for a single JSON box index."""
    import cv2

    all_corners = corners_cam_from_frame(frame)
    if box_index < 0 or box_index >= all_corners.shape[0]:
        return bgr
    K, W, H = intrinsics_for_projection(frame["camera"], frame)
    out = np.asarray(bgr)
    box_corners = all_corners[box_index]
    draw_color = color or _VIZ_BOX_COLORS_BGR[box_index % len(_VIZ_BOX_COLORS_BGR)]
    uv, depth = _project_points(K, box_corners)
    valid = depth > 1e-6
    for i, j in _BOX_EDGES:
        if not (valid[i] and valid[j]):
            continue
        p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        if _edge_is_solid(i, j, box_corners, all_corners, box_index):
            cv2.line(out, p1, p2, draw_color, line_thickness, cv2.LINE_AA)
        else:
            _draw_dashed_line(out, p1, p2, draw_color, line_thickness)
    if kps is not None and inst_id is not None:
        out = _draw_visible_kps_on_bgr(
            out, kps, inst_id, W=W, H=H, color=draw_color, point_radius=point_radius + 1,
        )
    else:
        for ci, ((u, v), ok) in enumerate(zip(uv, valid)):
            if not ok:
                continue
            if not _corner_visible(box_corners[ci], u, v, all_corners, W, H):
                continue
            c = (int(round(u)), int(round(v)))
            cv2.circle(out, c, point_radius, draw_color, -1, cv2.LINE_AA)
            cv2.circle(out, c, point_radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                out, str(ci), (c[0] + 6, c[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return out


def render_instance_mask_panel(
    sample: Mapping[str, Any],
    inst_id: int,
) -> np.ndarray:
    """Black background + single-instance mask (BGR)."""
    import cv2

    mask = np.asarray(sample["instance_mask"], dtype=np.uint8)
    H, W = mask.shape
    color = np.array(_VIZ_MASK_COLORS_RGB[inst_id % len(_VIZ_MASK_COLORS_RGB)], dtype=np.uint8)
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    region = mask == inst_id
    panel[region] = color[::-1]
    return panel


def render_single_instance_visualization(
    json_path: Union[str, Path],
    inst_id: int,
    *,
    image_path: Union[str, Path, None] = None,
    sample: Mapping[str, Any] | None = None,
    dim_factor: float = 0.68,
) -> np.ndarray:
    """
    Three-panel BGR image for one instance: RGB | mask | RGB+mask+kps+wireframe.

    Panel 3 dims other objects; only ``inst_id`` mask, kps, and wireframe are shown.
    """
    import cv2

    json_path = Path(json_path)
    frame = load_frame_json(json_path)
    if sample is None:
        sample = gen_train_sample_from_json(json_path, image_path=image_path)
    rgb_path = Path(image_path) if image_path is not None else Path(sample["image_path"])
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    k = int(sample["kps"].shape[0])
    if inst_id < 0 or inst_id >= k:
        raise ValueError(f"instance id {inst_id} out of range [0, {k})")

    box_index = int(sample["box_indices"][inst_id])
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel_rgb = rgb_bgr.copy()
    panel_mask = render_instance_mask_panel(sample, inst_id)

    base = (rgb_bgr.astype(np.float32) * dim_factor).astype(np.uint8)
    panel_overlay = overlay_mask_kps_on_bgr(base, sample, mask_alpha=0.45, instance_id=inst_id)
    panel_overlay = overlay_wireframe_for_box(
        panel_overlay, frame, box_index,
        kps=np.asarray(sample["kps"], dtype=np.float32),
        inst_id=inst_id,
    )

    title = f"{json_path.stem}  inst={inst_id:02d}  json_box={box_index:02d}"
    for panel, label in (
        (panel_rgb, "RGB"),
        (panel_mask, "mask"),
        (panel_overlay, "overlay"),
    ):
        cv2.putText(
            panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            panel, title, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA,
        )

    return np.concatenate([panel_rgb, panel_mask, panel_overlay], axis=1)


def save_single_instance_visualization(
    json_path: Union[str, Path],
    inst_id: int,
    output: Union[str, Path],
    *,
    image_path: Union[str, Path, None] = None,
    sample: Mapping[str, Any] | None = None,
) -> Path:
    import cv2

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render_single_instance_visualization(
        json_path, inst_id, image_path=image_path, sample=sample,
    ))
    return output


def _load_sample_for_viz(
    json_path: Path,
    *,
    labels_dir: Path | None = None,
    images_dir: Path | None = None,
) -> Dict[str, Any]:
    """Load sample for visualization; prefer cached ``labels/{stem}_train.json``."""
    stem = json_path.stem
    if labels_dir is not None:
        train_path = train_label_path(stem, labels_dir)
        if train_path.is_file():
            return load_train_sample_from_labels(
                stem, labels_dir, images_dir=images_dir,
            )
    return gen_train_sample_from_json(json_path, images_dir=images_dir)


def visualize_instances_from_dir(
    labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR,
    out_dir: Union[str, Path] = DEFAULT_DATA_ROOT / "viz" / "per_instance",
    pattern: str = "frame_*.json",
    *,
    images_dir: Union[str, Path, None] = None,
) -> List[Path]:
    """Write ``{stem}/{stem}_instXX_boxYY.png`` for every visible instance."""
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(images_dir) if images_dir is not None else labels_dir.parent / "images"

    json_paths = list_raw_label_jsons(labels_dir, pattern)
    saved: List[Path] = []
    total_inst = 0

    for json_path in json_paths:
        sample = _load_sample_for_viz(json_path, labels_dir=labels_dir, images_dir=img_dir)
        stem = json_path.stem
        frame_dir = out_dir / stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        k = int(sample["kps"].shape[0])
        for inst_id in range(k):
            box_index = int(sample["box_indices"][inst_id])
            out_path = frame_dir / f"{stem}_inst{inst_id:02d}_box{box_index:02d}.png"
            save_single_instance_visualization(
                json_path, inst_id, out_path, sample=sample,
            )
            saved.append(out_path)
            total_inst += 1
        print(f"{stem}: {k} instances -> {frame_dir}")

    index = {
        "frames": len(json_paths),
        "instances": total_inst,
        "out_dir": str(out_dir),
    }
    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"Saved {total_inst} instance images under {out_dir}")
    return saved


def render_sample_visualization(
    json_path: Union[str, Path],
    *,
    image_path: Union[str, Path, None] = None,
    sample: Mapping[str, Any] | None = None,
    projection_intrinsics: str | None = None,
) -> np.ndarray:
    """RGB + wireframe + instance mask + kps (BGR uint8)."""
    json_path = Path(json_path)
    frame = load_frame_json(json_path)
    if sample is None:
        sample = gen_train_sample_from_json(
            json_path, image_path=image_path, projection_intrinsics=projection_intrinsics,
        )
    elif projection_intrinsics is None:
        projection_intrinsics = sample.get("projection_intrinsics")
    rgb_path = Path(image_path) if image_path is not None else Path(sample["image_path"])
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    bgr = overlay_wireframes_on_rgb(
        rgb, frame, projection_intrinsics=projection_intrinsics,
    )
    return overlay_mask_kps_on_bgr(bgr, sample)


def save_sample_visualization(
    json_path: Union[str, Path],
    output: Union[str, Path],
    *,
    image_path: Union[str, Path, None] = None,
    sample: Mapping[str, Any] | None = None,
    projection_intrinsics: str | None = None,
) -> Path:
    import cv2

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render_sample_visualization(
        json_path,
        image_path=image_path,
        sample=sample,
        projection_intrinsics=projection_intrinsics,
    ))
    return output


def visualize_samples_from_dir(
    labels_dir: Union[str, Path] = DEFAULT_LABELS_DIR,
    out_dir: Union[str, Path] = DEFAULT_DATA_ROOT / "viz",
    pattern: str = "frame_*.json",
    *,
    images_dir: Union[str, Path, None] = None,
) -> List[Path]:
    """Write ``{stem}_viz.png`` for every matching frame JSON."""
    from PIL import ImageDraw, ImageFont

    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(images_dir) if images_dir is not None else labels_dir.parent / "images"

    json_paths = list_raw_label_jsons(labels_dir, pattern)
    saved: List[Path] = []

    for json_path in json_paths:
        sample = _load_sample_for_viz(json_path, labels_dir=labels_dir, images_dir=img_dir)
        stem = json_path.stem
        out_path = out_dir / f"{stem}_viz.png"
        save_sample_visualization(json_path, out_path, sample=sample)
        saved.append(out_path)
        print(f"Viz: {out_path} (k={sample['kps'].shape[0]})")

    if saved:
        thumb_w, thumb_h, cols = 480, 384, 6
        rows = (len(saved) + cols - 1) // cols
        cell_w, cell_h = thumb_w, thumb_h + 24
        grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (32, 32, 32))
        draw_font = ImageFont.load_default()
        for i, p in enumerate(saved):
            r, c = divmod(i, cols)
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_w, cell_h), (24, 24, 24))
            ox = (cell_w - im.width) // 2
            canvas.paste(im, (ox, 0))
            d = ImageDraw.Draw(canvas)
            d.text((8, thumb_h + 4), p.stem.replace("_viz", ""), fill=(220, 220, 220), font=draw_font)
            grid.paste(canvas, (c * cell_w, r * cell_h))
        grid_path = out_dir / "all_frames_grid.png"
        grid.save(grid_path)
        print(f"Grid: {grid_path}")

    return saved


def _corner_visible(
    point: np.ndarray,
    u: float,
    v: float,
    all_corners_cam: np.ndarray,
    W: int,
    H: int,
    *,
    tris: np.ndarray | None = None,
) -> bool:
    """Visible iff unobstructed along camera ray and inside the image."""
    if not _point_visible_from_camera(point, all_corners_cam, tris=tris):
        return False
    if u < 0 or u >= W or v < 0 or v >= H:
        return False
    return True


def _make_palette(num_instances: int, bg_index: int = 255) -> list[int]:
    """Build a 256-color RGB palette."""
    palette = [0] * (256 * 3)
    palette[bg_index * 3 + 0] = 0
    palette[bg_index * 3 + 1] = 0
    palette[bg_index * 3 + 2] = 0
    if num_instances == 0:
        return palette
    rng = np.random.RandomState(42)
    for i in range(num_instances):
        r, g, b = rng.randint(40, 255, size=3)
        palette[i * 3 + 0] = int(r)
        palette[i * 3 + 1] = int(g)
        palette[i * 3 + 2] = int(b)
    return palette


def gen_train_sample(
    K: np.ndarray,
    W: int,
    H: int,
    boxes: np.ndarray,
    *,
    corners_cam: np.ndarray | None = None,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Generate instance mask and projected 3D box corners for one training sample.

    Args:
        K: (3, 3) camera intrinsic matrix
        W, H: image width and height
        boxes: (n, 10) each row is
            (cx, cy, cz, w, l, h, qx, qy, qz, qw)
            in the OpenCV camera frame (+X right, +Y down, +Z forward)
        corners_cam: optional (n, 8, 3) precomputed corners (overrides ``boxes``)

    Returns:
        instance_mask: PIL Image mode 'P'. Background = 255; visible instances use
            ids 0..k-1 (k = number of visible boxes).
        kps: (k, 8, 3) projected corners as (x, y, v); v=0 valid, v=1 occluded / OOB.
            Row i corresponds to instance id i in the mask.
        box_indices: (k,) input box index for each output instance id.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 10)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n = boxes.shape[0]

    if n == 0:
        mask = np.full((H, W), _BG_LABEL, dtype=np.uint8)
        img = Image.fromarray(mask, mode="P")
        img.putpalette(_make_palette(0, _BG_LABEL))
        return img, np.zeros((0, 8, _KP_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if corners_cam is not None:
        corners_cam = np.asarray(corners_cam, dtype=np.float64).reshape(n, 8, 3)
    else:
        corners_cam = _box_corners_cam(boxes)
    corners_uv, corners_z = _project_points(K, corners_cam)

    mask_buf = render_instance_mask_buf(corners_cam, K, W, H, bg_label=_BG_LABEL)

    visible_labels = sorted(int(x) for x in np.unique(mask_buf) if x != _BG_LABEL)
    k = len(visible_labels)
    if k == 0:
        img = Image.fromarray(mask_buf, mode="P")
        img.putpalette(_make_palette(0, _BG_LABEL))
        return img, np.zeros((0, 8, _KP_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)

    # Remap temporary labels to contiguous ids 0..k-1
    remapped = np.full((H, W), _BG_LABEL, dtype=np.uint8)
    for new_id, old_label in enumerate(visible_labels):
        remapped[mask_buf == old_label] = new_id

    box_indices = np.array([label - 1 for label in visible_labels], dtype=np.int64)

    kps = np.zeros((k, 8, _KP_DIM), dtype=np.float32)
    kps[..., 2] = _KP_INVALID
    all_tris = _all_box_triangles(corners_cam)
    # (depth, out_i, ui, vi) — paint nearer corners last so they win overlaps.
    corner_splats: list[tuple[float, int, int, int]] = []
    for out_i, src_label in enumerate(visible_labels):
        box_idx = src_label - 1
        for c in range(8):
            point = corners_cam[box_idx, c]
            u, v = corners_uv[box_idx, c]
            z = float(corners_z[box_idx, c])
            if z > 1e-6:
                kps[out_i, c, 0] = float(u)
                kps[out_i, c, 1] = float(v)
            if _corner_visible(point, u, v, corners_cam, W, H, tris=all_tris):
                kps[out_i, c, 2] = _KP_VALID
                ui = int(round(u))
                vi = int(round(v))
                if 0 <= ui < W and 0 <= vi < H:
                    corner_splats.append((float(point[2]), out_i, ui, vi))

    # GL mask covers triangle interiors; corner vertices often fall on 1px gaps
    # or adjacent instances. Stamp visible corners so mask and kps agree.
    corner_splats.sort(key=lambda item: item[0], reverse=True)
    for _depth, out_i, ui, vi in corner_splats:
        remapped[vi, ui] = out_i

    img = Image.fromarray(remapped, mode="P")
    img.putpalette(_make_palette(k, _BG_LABEL))
    return img, kps, box_indices


def make_random_boxes(n: int, seed: int = 0) -> np.ndarray:
    """Synthetic camera-frame boxes for debugging / synthetic training."""
    rng = np.random.RandomState(seed)
    boxes = np.zeros((n, 10), dtype=np.float64)
    for i in range(n):
        boxes[i, 0] = rng.uniform(-2.5, 2.5)
        boxes[i, 1] = rng.uniform(6.0, 12.0)
        boxes[i, 2] = rng.uniform(0.5, 3.5)
        w, l, h = rng.uniform(0.8, 2.0, size=3)
        boxes[i, 3:6] = (w, l, h)
        yaw = math.radians(rng.uniform(-40.0, 40.0))
        boxes[i, 6:10] = Rotation.from_euler("z", yaw).as_quat()
    return boxes


def make_demo_boxes(seed: int = 0) -> np.ndarray:
    """Fixed demo boxes for quick visualization (``seed`` ignored)."""
    del seed
    specs = (
        dict(center=(-1.2, 8.0, 1.2), size=(1.2, 0.9, 0.8), euler=(0, 0, 15)),
        dict(center=(1.0, 9.0, 1.5), size=(1.0, 1.1, 0.7), euler=(0, 0, -20)),
        dict(center=(-2.6, 0.3, 7.5), size=(1.8, 1.2, 0.9), euler=(0, 0, -20)),
        dict(center=(2.6, -0.2, 8.5), size=(2.0, 1.4, 1.1), euler=(5, 10, 30)),
    )
    rows: list[list[float]] = []
    for spec in specs:
        cx, cy, cz = spec["center"]
        w, l, h = spec["size"]
        qx, qy, qz, qw = Rotation.from_euler("xyz", spec["euler"], degrees=True).as_quat()
        rows.append([cx, cy, cz, w, l, h, qx, qy, qz, qw])
    return np.asarray(rows, dtype=np.float64)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate instance mask + keypoints from frame JSON annotations",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Raw label JSON file(s); default: all under --labels-dir or --in-root",
    )
    parser.add_argument(
        "--in-root",
        type=str,
        default=None,
        help="Source tree of capture JSON+PNG; mirrored into --out-root",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output tree for jpg + *_train.json + instance masks (requires --in-root)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=str(DEFAULT_LABELS_DIR),
        help="Directory for raw + generated labels (default: data/labels)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory of RGB images (default: data/images)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.json",
        help="Glob pattern for raw annotation files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse when listing JSON under --labels-dir",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality when copying RGB into --out-root",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="If set, write RGB+wireframe+mask+kps+2D-box PNGs under this directory",
    )
    parser.add_argument(
        "--viz-limit",
        type=int,
        default=24,
        help="Max viz frames when --viz-dir is set (evenly sampled; default 24)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool size for --in-root generation (default 1)",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default="pinhole",
        choices=("pinhole", "nyx"),
        help="Projection intrinsics for --in-root (default pinhole; nyx=legacy FOV fix)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Regenerate even if jpg/mask/train json already exist",
    )
    parser.add_argument(
        "--viz-inst-dir",
        type=str,
        default=None,
        help="If set, write per-instance 3-panel PNGs under {dir}/{stem}/",
    )
    args = parser.parse_args()

    if args.in_root:
        if not args.out_root:
            raise SystemExit("--out-root is required with --in-root")
        pattern = args.pattern if args.pattern != "frame_*.json" else "*.json"
        gen_and_save_from_tree(
            args.in_root,
            args.out_root,
            pattern=pattern,
            jpg_quality=args.jpg_quality,
            viz_dir=args.viz_dir,
            viz_limit=args.viz_limit,
            workers=args.workers,
            skip_existing=not args.no_skip_existing,
            projection_intrinsics=args.intrinsics,
        )
        return

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)

    if args.inputs:
        json_paths = [Path(p) for p in args.inputs if is_raw_label_json(p)]
    else:
        json_paths = list_raw_label_jsons(
            labels_dir, args.pattern, recursive=args.recursive,
        )

    if not json_paths:
        raise SystemExit(f"No raw label JSON files found under {labels_dir}")

    samples = [
        gen_train_sample_from_json(path, images_dir=images_dir)
        for path in json_paths
    ]
    index_path = save_train_samples(samples, labels_dir, source_jsons=json_paths)

    for json_path, sample in zip(json_paths, samples):
        stem = json_path.stem
        k = int(sample["kps"].shape[0])
        mask_arr = np.asarray(sample["instance_mask"])
        n_kp_on_mask = 0
        for inst_id in range(k):
            for c in range(8):
                kp = ensure_kps_xyv(sample["kps"])[inst_id, c]
                if not kp_corner_valid(kp):
                    continue
                x, y = kp_xy(kp)
                ui, vi = int(round(x)), int(round(y))
                if 0 <= ui < sample["W"] and 0 <= vi < sample["H"]:
                    if mask_arr[vi, ui] == inst_id:
                        n_kp_on_mask += 1

        print(
            f"{json_path.name}: k={k} -> {train_label_path(stem, labels_dir)} "
            f"kp_on_mask={n_kp_on_mask}"
        )

    print(f"Saved {len(samples)} samples, index: {index_path}")

    if args.viz_dir:
        visualize_samples_from_dir(
            labels_dir, args.viz_dir, pattern=args.pattern, images_dir=images_dir,
        )

    if args.viz_inst_dir:
        visualize_instances_from_dir(
            labels_dir, args.viz_inst_dir, pattern=args.pattern, images_dir=images_dir,
        )


if __name__ == "__main__":
    _main()
