"""Planar camera-to-robot calibration math and versioned persistence."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np


TransformType = Literal["affine", "homography"]
ImageSize = tuple[int, int]
CURRENT_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, CURRENT_SCHEMA_VERSION)
DEFAULT_COORDINATE_CONVENTION = (
    "pixel origin top-left; u right, v down; robot X/Y in entered units"
)


class CalibrationError(ValueError):
    """Base error for invalid calibration input or stored data."""


class CalibrationGeometryMismatchError(CalibrationError):
    """Raised when a calibration is applied to a different image geometry."""


def _points(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise CalibrationError(f"{name} must contain four finite (x, y) points.")
    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    if span <= 1e-10:
        raise CalibrationError(f"{name} points are too close together.")
    for first in range(4):
        for second in range(first + 1, 4):
            for third in range(second + 1, 4):
                ab = points[second] - points[first]
                ac = points[third] - points[first]
                twice_area = abs(ab[0] * ac[1] - ab[1] * ac[0])
                if twice_area <= 1e-8 * span * span:
                    raise CalibrationError(
                        f"{name} has three nearly collinear points."
                    )
    return points


def _normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    shifted = points - center
    rms_distance = np.sqrt(np.mean(np.sum(shifted * shifted, axis=1)))
    if rms_distance <= 1e-12:
        raise CalibrationError("Calibration points cannot be normalized safely.")
    scale = np.sqrt(2.0) / rms_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return shifted * scale, transform


def fit_homography(
    camera_points: Sequence[Sequence[float]],
    robot_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Fit a 3x3 normalized DLT map from camera pixels to robot X/Y."""

    source = _points(camera_points, "Camera")
    destination = _points(robot_points, "Robot")
    source_normalized, source_transform = _normalize(source)
    destination_normalized, destination_transform = _normalize(destination)
    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(source_normalized, destination_normalized):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    system = np.asarray(rows, dtype=np.float64)
    _, singular_values, vh = np.linalg.svd(system, full_matrices=True)
    if (
        singular_values[0] <= 1e-14
        or singular_values[-1] / singular_values[0] < 1e-10
    ):
        raise CalibrationError(
            "The four pairs cannot define a stable mapping. Spread the points out."
        )

    matrix = (
        np.linalg.inv(destination_transform)
        @ vh[-1].reshape(3, 3)
        @ source_transform
    )
    scale = matrix[2, 2]
    if abs(scale) > 1e-12:
        matrix = matrix / scale
    else:
        norm = np.linalg.norm(matrix)
        if norm <= 1e-12:
            raise CalibrationError("The calculated mapping is singular.")
        matrix = matrix / norm
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise CalibrationError("The calculated mapping is singular.")
    _assert_fit(matrix, source, destination)
    return matrix


def fit_affine(
    camera_points: Sequence[Sequence[float]],
    robot_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Fit a 2x3 affine map from camera pixels to robot X/Y."""

    source = _points(camera_points, "Camera")
    destination = _points(robot_points, "Robot")
    source_normalized, source_transform = _normalize(source)
    destination_normalized, destination_transform = _normalize(destination)
    design = np.column_stack((source_normalized, np.ones(4)))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, destination_normalized, rcond=None
    )
    if rank != 3:
        raise CalibrationError("The camera points do not define a stable affine mapping.")
    normalized_matrix = np.vstack((coefficients.T, [0.0, 0.0, 1.0]))
    matrix = (
        np.linalg.inv(destination_transform)
        @ normalized_matrix
        @ source_transform
    )
    matrix = matrix / matrix[2, 2]
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise CalibrationError("The calculated affine mapping is singular.")
    # Four correspondences overdetermine a six-parameter affine transform. A
    # least-squares fit normally has a small residual from pixel selection,
    # even when the points describe the same plane.
    mapped = np.asarray(
        [transform_point(matrix[:2, :], u, v) for u, v in source],
        dtype=np.float64,
    )
    max_error = float(np.max(np.linalg.norm(mapped - destination, axis=1)))
    robot_span = float(np.linalg.norm(np.ptp(destination, axis=0)))
    allowed_error = max(1e-6, 0.01 * robot_span)
    if max_error > allowed_error:
        raise CalibrationError(
            f"Affine point fit error is {max_error:.2f} robot units "
            f"(limit {allowed_error:.2f}). Check the matching points or "
            "use 3 x 3 Perspective."
        )
    return matrix[:2, :]


def transform_point(matrix: Sequence[Sequence[float]], u: float, v: float) -> tuple[float, float]:
    """Apply an affine or homography matrix to one camera pixel."""

    array = np.asarray(matrix, dtype=np.float64)
    vector = array @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    if array.shape == (2, 3):
        if not np.isfinite(vector).all():
            raise CalibrationError("The transformed point is invalid.")
        return float(vector[0]), float(vector[1])
    if array.shape != (3, 3):
        raise CalibrationError("The transform must be a 2x3 or 3x3 matrix.")
    if (
        not np.isfinite(vector).all()
        or abs(vector[2]) <= 1e-12 * max(1.0, np.linalg.norm(vector))
    ):
        raise CalibrationError("This pixel is at or near the transform's horizon.")
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def _assert_fit(
    matrix: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
) -> None:
    mapped = np.asarray(
        [transform_point(matrix, u, v) for u, v in source], dtype=np.float64
    )
    tolerance = 1e-6 * max(1.0, np.ptp(destination, axis=0).max())
    if not np.allclose(mapped, destination, rtol=0.0, atol=tolerance):
        raise CalibrationError("The point pairs cannot be mapped consistently.")


def _image_size(value: Sequence[int]) -> ImageSize:
    if len(value) != 2:
        raise CalibrationError("Image size must contain width and height.")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise CalibrationError("Image width and height must be positive.")
    return width, height


def _tuple_points(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in values)


@dataclass(frozen=True)
class CalibrationRecord:
    transform: TransformType
    camera_points_uv: tuple[tuple[float, float], ...]
    robot_points_xy: tuple[tuple[float, float], ...]
    matrix: tuple[tuple[float, ...], ...]
    image_width: int
    image_height: int
    coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION
    camera_metadata: dict[str, Any] | None = None
    roi: dict[str, Any] | None = None
    snapshot_png: bytes | None = None
    created_at: str | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION
    source_schema_version: int = CURRENT_SCHEMA_VERSION

    @property
    def image_size(self) -> ImageSize:
        return self.image_width, self.image_height

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "transform": self.transform,
            "coordinate_convention": self.coordinate_convention,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_points_uv": [list(point) for point in self.camera_points_uv],
            "robot_points_xy": [list(point) for point in self.robot_points_xy],
            "matrix": [list(row) for row in self.matrix],
            "camera_metadata": self.camera_metadata or {},
            "roi": self.roi,
            "created_at": self.created_at or _utc_now_iso(),
        }
        if self.snapshot_png is not None:
            data["snapshot_png_base64"] = base64.b64encode(self.snapshot_png).decode(
                "ascii"
            )
        return data


class CalibrationService:
    """Stateless calibration operations shared by UI and future downstream modules."""

    @staticmethod
    def fit(
        transform: TransformType,
        camera_points_uv: Sequence[Sequence[float]],
        robot_points_xy: Sequence[Sequence[float]],
        image_size: Sequence[int],
        *,
        coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION,
        camera_metadata: dict[str, Any] | None = None,
        roi: dict[str, Any] | None = None,
        snapshot_png: bytes | None = None,
    ) -> CalibrationRecord:
        width, height = _image_size(image_size)
        if transform == "affine":
            fitted = fit_affine(camera_points_uv, robot_points_xy)
        elif transform == "homography":
            fitted = fit_homography(camera_points_uv, robot_points_xy)
        else:
            raise CalibrationError(f"Unknown transform type: {transform!r}")
        source = _points(camera_points_uv, "Camera")
        destination = _points(robot_points_xy, "Robot")
        return CalibrationRecord(
            transform=transform,
            camera_points_uv=_tuple_points(source),
            robot_points_xy=_tuple_points(destination),
            matrix=tuple(tuple(float(value) for value in row) for row in fitted),
            image_width=width,
            image_height=height,
            coordinate_convention=coordinate_convention,
            camera_metadata=dict(camera_metadata or {}),
            roi=dict(roi) if roi is not None else None,
            snapshot_png=snapshot_png,
            created_at=_utc_now_iso(),
        )

    @staticmethod
    def apply(
        record: CalibrationRecord,
        point_uv: Sequence[float],
        current_image_size: Sequence[int],
    ) -> tuple[float, float]:
        CalibrationService.validate_geometry(record, current_image_size)
        if len(point_uv) != 2:
            raise CalibrationError("A pixel must contain u and v coordinates.")
        return transform_point(record.matrix, float(point_uv[0]), float(point_uv[1]))

    @staticmethod
    def validate_geometry(
        record: CalibrationRecord, current_image_size: Sequence[int]
    ) -> None:
        actual = _image_size(current_image_size)
        if actual != record.image_size:
            raise CalibrationGeometryMismatchError(
                "Calibration geometry mismatch: saved for "
                f"{record.image_width}x{record.image_height}, current image is "
                f"{actual[0]}x{actual[1]}. Load a matching image or recalibrate."
            )

    @staticmethod
    def save(path: Path, record: CalibrationRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> CalibrationRecord:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise CalibrationError("Unsupported calibration file version.")
        transform: TransformType = (
            "homography" if version == 1 else data.get("transform")
        )
        if transform not in ("affine", "homography"):
            raise CalibrationError("Unknown transform type.")
        image_size = _image_size((data["image_width"], data["image_height"]))
        camera_points = _points(data["camera_points_uv"], "Camera")
        robot_points = _points(data["robot_points_xy"], "Robot")
        fitted = (
            fit_homography(camera_points, robot_points)
            if transform == "homography"
            else fit_affine(camera_points, robot_points)
        )
        matrix_key = "matrix_3x3" if version == 1 else "matrix"
        stored = np.asarray(data[matrix_key], dtype=np.float64)
        if stored.shape != fitted.shape or not np.allclose(
            fitted, stored, rtol=1e-7, atol=1e-7
        ):
            raise CalibrationError("Saved matrix does not match the saved points.")
        snapshot = data.get("snapshot_png_base64")
        snapshot_bytes = (
            base64.b64decode(snapshot, validate=True)
            if snapshot is not None
            else None
        )
        return CalibrationRecord(
            transform=transform,
            camera_points_uv=_tuple_points(camera_points),
            robot_points_xy=_tuple_points(robot_points),
            matrix=tuple(tuple(float(value) for value in row) for row in stored),
            image_width=image_size[0],
            image_height=image_size[1],
            coordinate_convention=data.get(
                "coordinate_convention", DEFAULT_COORDINATE_CONVENTION
            ),
            camera_metadata=dict(data.get("camera_metadata", {})),
            roi=dict(data["roi"]) if data.get("roi") is not None else None,
            snapshot_png=snapshot_bytes,
            created_at=data.get("created_at"),
            schema_version=CURRENT_SCHEMA_VERSION,
            source_schema_version=int(version),
        )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
