"""Resolve only internally generated upload paths kept inside the upload directory."""
import os
import re

_INTERNAL_NAME = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{1,8}$")


def safe_video_path(video, upload_dir):
    stored = video.get("stored_filename")
    reference = video.get("file_path")
    if not isinstance(stored, str) or reference != stored or not _INTERNAL_NAME.fullmatch(stored):
        return None
    root = os.path.realpath(upload_dir)
    candidate = os.path.join(root, stored)
    if os.path.islink(candidate):
        return None
    target = os.path.realpath(candidate)
    try:
        if os.path.commonpath([root, target]) != root:
            return None
    except ValueError:
        return None
    return target if os.path.isfile(target) else None
def safe_processed_path(video, processed_dir):
    name = video.get("processed_filename")
    # Accept legacy MP4 results as well as new browser-compatible VP8/WebM files.
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{24}_annotated\.(?:mp4|webm)", name):
        return None
    root = os.path.realpath(processed_dir)
    target = os.path.realpath(os.path.join(root, name))
    try:
        if os.path.commonpath([root, target]) != root or os.path.islink(os.path.join(root, name)):
            return None
    except ValueError:
        return None
    return target if os.path.isfile(target) else None
