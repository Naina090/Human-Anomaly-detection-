"""Validation and spatial association for optional normalized surveillance zones."""
import json
import math
import re

import cv2
import numpy as np


class ZoneValidationError(ValueError):
    """Raised when client-supplied ROI coordinates are malformed."""


_ZONE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")


def parse_zones(raw, max_zones=8, max_points=100):
    """Parse canonical zones from a form JSON value; an omitted value means no ROI."""
    if raw in (None, ""):
        return []
    try:
        zones = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        raise ZoneValidationError("Zone data is not valid JSON. Clear the zone and draw it again.") from exc
    if not isinstance(zones, list) or len(zones) > max_zones:
        raise ZoneValidationError(f"Submit a list with no more than {max_zones} zones.")
    normalized = []
    for index, zone in enumerate(zones, 1):
        if not isinstance(zone, dict):
            raise ZoneValidationError("Each zone must contain a name, enabled setting, and points.")
        zone_id = str(zone.get("zone_id") or f"zone_{index}")
        if not _ZONE_ID.fullmatch(zone_id):
            raise ZoneValidationError("A zone has an invalid identifier.")
        name = str(zone.get("name") or f"Zone {index}").strip()[:80]
        if not name:
            name = f"Zone {index}"
        enabled = zone.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ZoneValidationError("Zone enabled values must be true or false.")
        points = zone.get("points")
        if not isinstance(points, list) or not 3 <= len(points) <= max_points:
            raise ZoneValidationError(f"{name} needs between 3 and {max_points} points.")
        clean_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ZoneValidationError(f"{name} contains an invalid point.")
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ZoneValidationError(f"{name} contains a non-numeric point.") from exc
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
                raise ZoneValidationError(f"{name} coordinates must be normalized between 0 and 1.")
            clean_points.append([x, y])
        if len({(round(x, 8), round(y, 8)) for x, y in clean_points}) < 3:
            raise ZoneValidationError(f"{name} needs at least 3 distinct points.")
        contour = np.asarray([[x, y] for x, y in clean_points], dtype=np.float32)
        if abs(cv2.contourArea(contour)) < 1e-6:
            raise ZoneValidationError(f"{name} must enclose an area.")
        normalized.append({"zone_id": zone_id, "name": name, "enabled": enabled, "points": clean_points})
    if len({zone["zone_id"] for zone in normalized}) != len(normalized):
        raise ZoneValidationError("Zone identifiers must be unique.")
    return normalized


def associate_event_with_zones(event, zones, width, height):
    """Annotate a localized event with zones containing its bounding-box bottom center."""
    enabled = [zone for zone in (zones or []) if zone.get("enabled") is True]
    if not enabled:
        return event
    details = event.setdefault("details", {})
    box = details.get("person_bbox_xyxy")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or width <= 0 or height <= 0:
        event["zone_association_status"] = "unavailable_no_person_location"
        event["zone_matches"] = []
        return event
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError
    except (TypeError, ValueError):
        event["zone_association_status"] = "unavailable_no_person_location"
        event["zone_matches"] = []
        return event
    point = ((x1 + x2) / 2.0, y2)
    details["zone_test_point_xy"] = [round(point[0], 2), round(point[1], 2)]
    details["zone_test_rule"] = "bottom-center of selected pose person's bounding box"
    matches = []
    for zone in enabled:
        polygon = np.asarray([[round(x * width), round(y * height)] for x, y in zone["points"]],
                             dtype=np.int32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            matches.append(zone["zone_id"])
    event["zone_association_status"] = "inside_active_zone" if matches else "outside_active_zones"
    event["zone_matches"] = matches
    return event


def draw_zones(frame, zones):
    """Draw translucent polygons on a derived frame, never on the uploaded source."""
    if not zones:
        return frame
    height, width = frame.shape[:2]
    overlay = frame.copy()
    outlines = []
    for zone in zones:
        points = np.asarray([[round(x * width), round(y * height)] for x, y in zone["points"]],
                            dtype=np.int32).reshape((-1, 1, 2))
        active = zone.get("enabled") is True
        color = (45, 65, 225) if active else (125, 135, 145)
        if active:
            cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], True, color, 2, cv2.LINE_AA)
        anchor = tuple(points.reshape((-1, 2))[0])
        outlines.append((anchor, str(zone.get("name") or "Zone"), color))
    output = cv2.addWeighted(overlay, 0.24, frame, 0.76, 0)
    for anchor, name, color in outlines:
        cv2.putText(output, name[:40], (max(4, anchor[0]), max(16, anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output
