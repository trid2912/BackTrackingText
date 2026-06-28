"""Core polygon estimation.

A core polygon is a fixed-vertex, equiangular circumscribing polygon fitted to a
binary mask. Compared to a convex hull (whose vertex count varies frame to
frame), it provides a stable, topologically-consistent set of vertices, which is
what makes robust homography estimation possible across frames.

This module only contains the geometry needed by the back-tracking tracker
(``core/tracking.py``): convex-hull computation, polygon construction at fixed
edge orientations, and the optimal core-polygon search over the starting angle.
"""

import logging
import os

import numpy as np


def extract_white_pixels(mask):
    """Return the (x, y) coordinates of foreground pixels (value > 0)."""
    ys, xs = np.where(mask > 0)
    return np.column_stack((xs, ys))  # (x, y) order


def cross_product(p1, p2, p3):
    """Cross product (p2 - p1) x (p3 - p1); its sign gives the turn direction."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def monotone_chain(points):
    """Convex hull via the Andrew monotone-chain algorithm."""
    if len(points) < 3:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def find_supporting_line(hull, start_point, angle):
    """Return a supporting line of given ``angle`` with all hull points on one side."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([direction[1], -direction[0]])
    dots = np.dot(hull, normal)
    max_dot = np.dot(start_point, normal)
    offset = np.max(np.maximum(dots - max_dot, 0))  # push the line outward
    adjusted_point = start_point + offset * normal
    return adjusted_point, direction


def construct_polygon(hull_points, num_edges, start_angle):
    """Build an equiangular circumscribing polygon containing all hull points.

    Edges are placed at orientations ``start_angle + k * 2*pi/num_edges``; the
    vertices are the intersections of consecutive supporting lines.
    """
    hull = monotone_chain(hull_points)
    angles = [start_angle + k * (2 * np.pi / num_edges) for k in range(num_edges)]
    polygon_vertices = []
    current_point = hull[np.argmin(hull[:, 0])]  # start from the leftmost point

    for angle in angles:
        result = find_supporting_line(hull, current_point, angle)
        if result is None:
            current_idx = np.where((hull == current_point).all(axis=1))[0][0]
            next_idx = (current_idx + 1) % len(hull)
            current_point = hull[next_idx]
            result = find_supporting_line(hull, current_point, angle)
            if result is None:
                raise ValueError("Cannot construct polygon with given constraints")
        point, _ = result
        polygon_vertices.append(point)
        current_point = point

    # Intersect consecutive supporting lines to obtain the vertices.
    vertices = []
    for i in range(len(polygon_vertices)):
        p1 = polygon_vertices[i]
        d1 = np.array([np.cos(angles[i]), np.sin(angles[i])])
        p2 = polygon_vertices[(i + 1) % len(polygon_vertices)]
        d2 = np.array([np.cos(angles[(i + 1) % num_edges]), np.sin(angles[(i + 1) % num_edges])])
        n1 = np.array([d1[1], -d1[0]])
        n2 = np.array([d2[1], -d2[0]])
        c1 = np.dot(p1, n1)
        c2 = np.dot(p2, n2)
        A = np.vstack([n1, n2])
        b = np.array([c1, c2])
        vertices.append(np.linalg.solve(A, b))

    return np.array(vertices)


def compute_polygon_area(vertices):
    """Polygon area via the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def find_best_core_polygon(hull_points, num_edges, step_size):
    """Search the starting angle and return the tightest (smallest-area) core polygon.

    Parameters
    ----------
    hull_points : ndarray (m, 2)
        Hull / contour points of the mask.
    num_edges : int
        Number of polygon edges (``n``).
    step_size : float
        Sampling step for the starting angle over ``[0, 2*pi/num_edges)``.

    Returns
    -------
    ndarray
        Vertices of the smallest-area valid polygon.
    """
    best_area = np.inf
    best_vertices = None
    angles = np.arange(0, 2 * np.pi / num_edges, step_size)

    for start_angle in angles:
        try:
            vertices = construct_polygon(hull_points, num_edges, start_angle)
        except (ValueError, np.linalg.LinAlgError):
            continue
        area = compute_polygon_area(vertices)
        if area < best_area:
            best_area = area
            best_vertices = vertices

    if best_vertices is None:
        raise ValueError("No valid polygon could be constructed for any starting angle")

    return best_vertices


def parse_txt_file(txt_file, min_bbox_size=1.0, min_area=1.0):
    """Parse a polygon annotation file.

    Each line is ``instance_id x1 y1 x2 y2 ... xn yn``. Negative coordinates are
    clamped to 0; degenerate or too-small boxes are skipped.

    Returns a list of dicts with ``instance_id``, ``polygon`` and ``bbox``
    (``[x_min, y_min, width, height]``).
    """
    instances = []
    if not os.path.exists(txt_file):
        logging.warning("Annotation file %s does not exist. Skipping.", txt_file)
        return instances

    with open(txt_file, "r") as f:
        for line_num, line in enumerate(f):
            data = list(map(float, line.strip().split()))
            if len(data) < 9:
                logging.warning("Line %d in %s is malformed. Skipping.", line_num + 1, txt_file)
                continue

            instance_id = int(data[0])
            points = data[1:]
            x_coords = [max(0.0, x) for x in points[0::2]]
            y_coords = [max(0.0, y) for y in points[1::2]]

            if min(x_coords) == max(x_coords) or min(y_coords) == max(y_coords):
                logging.warning("Line %d in %s is a degenerate polygon. Skipping.", line_num + 1, txt_file)
                continue

            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            if width < min_bbox_size or height < min_bbox_size or width * height < min_area:
                logging.warning("Line %d in %s is too small. Skipping.", line_num + 1, txt_file)
                continue

            interleaved = [c for pair in zip(x_coords, y_coords) for c in pair]
            instances.append({
                "instance_id": instance_id,
                "polygon": interleaved,
                "bbox": [x_min, y_min, width, height],
            })
    return instances
