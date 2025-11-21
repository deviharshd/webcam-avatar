"""
Webcam Avatar
=============

Real-time 2D stick-figure avatar driven by your webcam using MediaPipe Pose.

This script does NOT use PyBullet or physics. It is a pure computer-vision
demo: a neural network detects your body landmarks from the webcam and we draw
an avatar that mirrors your movements in a separate window.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Joint2D:
    """2D joint position in avatar space."""

    x: float
    y: float


class PoseSmoother:
    """Simple exponential smoother for joint positions."""

    def __init__(self, alpha: float = 0.75) -> None:
        self.alpha = alpha
        self.state: Dict[str, np.ndarray] = {}

    def update(self, joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update smoothed positions for all joints."""
        smoothed: Dict[str, np.ndarray] = {}
        for name, pos in joints.items():
            if name in self.state:
                prev = self.state[name]
                new = self.alpha * pos + (1.0 - self.alpha) * prev
            else:
                new = pos
            self.state[name] = new
            smoothed[name] = new
        return smoothed


def get_landmark_dict(
    pose_landmarks,
    image_width: int,
    image_height: int,
) -> Dict[str, np.ndarray]:
    """
    Extract selected landmarks into a dict of pixel coordinates.

    Keys: head, neck, right_shoulder, right_elbow, right_wrist,
          left_shoulder, left_elbow, left_wrist,
          right_hip, right_knee, right_ankle,
          left_hip, left_knee, left_ankle.
    """
    mp_pose = mp.solutions.pose.PoseLandmark
    index_map = {
        "nose": mp_pose.NOSE.value,
        "right_shoulder": mp_pose.RIGHT_SHOULDER.value,
        "right_elbow": mp_pose.RIGHT_ELBOW.value,
        "right_wrist": mp_pose.RIGHT_WRIST.value,
        "left_shoulder": mp_pose.LEFT_SHOULDER.value,
        "left_elbow": mp_pose.LEFT_ELBOW.value,
        "left_wrist": mp_pose.LEFT_WRIST.value,
        "right_hip": mp_pose.RIGHT_HIP.value,
        "right_knee": mp_pose.RIGHT_KNEE.value,
        "right_ankle": mp_pose.RIGHT_ANKLE.value,
        "left_hip": mp_pose.LEFT_HIP.value,
        "left_knee": mp_pose.LEFT_KNEE.value,
        "left_ankle": mp_pose.LEFT_ANKLE.value,
    }

    pts: Dict[str, np.ndarray] = {}
    for name, idx in index_map.items():
        lm = pose_landmarks.landmark[idx]
        if lm.visibility < 0.5:
            continue
        x = lm.x * image_width
        y = lm.y * image_height
        pts[name] = np.array([x, y], dtype=np.float32)

    # Derive approximate head and neck points.
    if "nose" in pts:
        pts["head"] = pts["nose"]
    if "right_shoulder" in pts and "left_shoulder" in pts:
        pts["neck"] = 0.5 * (pts["right_shoulder"] + pts["left_shoulder"])
    return pts


def normalize_to_avatar_space(
    joints: Dict[str, np.ndarray],
    avatar_size: Tuple[int, int],
) -> Dict[str, Joint2D]:
    """
    Map pixel-space joints into a centered avatar coordinate system.

    - Centered on mid-hip (average of left/right hip) if available.
    - Scaled to fit nicely inside the avatar canvas.
    """
    width, height = avatar_size
    if not joints:
        return {}

    # Choose origin: mid-hip if available, else neck if available, else mean.
    if "right_hip" in joints and "left_hip" in joints:
        origin = 0.5 * (joints["right_hip"] + joints["left_hip"])
    elif "neck" in joints:
        origin = joints["neck"]
    else:
        stacked = np.stack(list(joints.values()), axis=0)
        origin = stacked.mean(axis=0)

    centered = {name: pos - origin for name, pos in joints.items()}

    # Compute scale based on max distance.
    stacked = np.stack(list(centered.values()), axis=0)
    max_radius = float(np.linalg.norm(stacked, axis=1).max())
    if max_radius < 1e-3:
        scale = 1.0
    else:
        # Keep some margin in the avatar canvas.
        scale = 0.4 * min(width, height) / max_radius

    avatar_joints: Dict[str, Joint2D] = {}
    center_x = width * 0.5
    center_y = height * 0.6  # place hips slightly below vertical center
    for name, pos in centered.items():
        ax = center_x + scale * pos[0]
        ay = center_y + scale * pos[1]
        avatar_joints[name] = Joint2D(x=float(ax), y=float(ay))
    return avatar_joints


def draw_avatar(
    canvas: np.ndarray,
    joints: Dict[str, Joint2D],
) -> None:
    """Draw a more human-like stick avatar on the given canvas."""
    h, w, _ = canvas.shape

    # Clear canvas with a subtle vertical gradient background so improvements
    # are clearly visible and the avatar stands out.
    top_color = np.array([245, 252, 255], dtype=np.uint8)
    bottom_color = np.array([225, 240, 250], dtype=np.uint8)
    for y in range(h):
        t = y / max(h - 1, 1)
        row_color = (1.0 - t) * top_color + t * bottom_color
        canvas[y, :] = row_color

    # Scale line thickness and joint size with image size so the avatar looks
    # bold and clearly visible. Values are deliberately large so arms/legs and
    # head feel chunky and readable.
    limb_thickness = max(16, h // 90)        # much thicker limbs
    torso_thickness = limb_thickness + 4
    joint_radius = max(12, h // 120)         # larger joints
    head_radius = max(joint_radius + 24, h // 50)  # noticeably larger head

    # Optional ground line and simple shadow under the feet.
    ground_y = None
    if "right_ankle" in joints and "left_ankle" in joints:
        ra = joints["right_ankle"]
        la = joints["left_ankle"]
        ground_y = int(max(ra.y, la.y) + joint_radius * 1.2)
    elif "right_ankle" in joints:
        ground_y = int(joints["right_ankle"].y + joint_radius * 1.2)
    elif "left_ankle" in joints:
        ground_y = int(joints["left_ankle"].y + joint_radius * 1.2)

    if ground_y is not None:
        ground_y = min(max(ground_y, int(h * 0.55)), h - 5)
        cv2.line(
            canvas,
            (int(w * 0.15), ground_y),
            (int(w * 0.85), ground_y),
            (220, 220, 220),
            max(1, limb_thickness - 1),
            lineType=cv2.LINE_AA,
        )
        # Soft shadow ellipse under the character.
        center_x = int(w * 0.5)
        shadow_width = max(20, int(w * 0.18))
        shadow_height = max(8, joint_radius * 2)
        cv2.ellipse(
            canvas,
            (center_x, ground_y + shadow_height // 2),
            (shadow_width, shadow_height),
            0,
            0,
            360,
            (230, 230, 230),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # Helper for torso polygon.
    has_shoulders = "right_shoulder" in joints and "left_shoulder" in joints
    has_hips = "right_hip" in joints and "left_hip" in joints

    if has_shoulders and has_hips:
        rs = joints["right_shoulder"]
        ls = joints["left_shoulder"]
        rh = joints["right_hip"]
        lh = joints["left_hip"]
        torso_pts = np.array(
            [
                [int(ls.x), int(ls.y)],
                [int(rs.x), int(rs.y)],
                [int(rh.x), int(rh.y)],
                [int(lh.x), int(lh.y)],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, torso_pts, color=(230, 240, 255))
        cv2.polylines(canvas, [torso_pts], isClosed=True, color=(0, 120, 220), thickness=torso_thickness)

    # Arm and leg connections.
    arm_color = (0, 150, 255)
    leg_color = (0, 120, 180)

    arm_connections = [
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    ]
    leg_connections = [
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "left_hip"),  # pelvis
    ]

    for a, b in arm_connections:
        if a in joints and b in joints:
            pa = joints[a]
            pb = joints[b]
            cv2.line(
                canvas,
                (int(pa.x), int(pa.y)),
                (int(pb.x), int(pb.y)),
                arm_color,
                limb_thickness,
                lineType=cv2.LINE_AA,
            )

    for a, b in leg_connections:
        if a in joints and b in joints:
            pa = joints[a]
            pb = joints[b]
            cv2.line(
                canvas,
                (int(pa.x), int(pa.y)),
                (int(pb.x), int(pb.y)),
                leg_color,
                limb_thickness,
                lineType=cv2.LINE_AA,
            )

    # Head with simple face.
    if "head" in joints:
        head = joints["head"]
        center = (int(head.x), int(head.y))
        cv2.circle(canvas, center, head_radius, (255, 230, 230), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, head_radius, (200, 100, 100), thickness=2, lineType=cv2.LINE_AA)

        eye_offset_x = max(2, head_radius // 3)
        eye_offset_y = max(2, head_radius // 3)
        eye_radius = max(2, head_radius // 6)
        cv2.circle(canvas, (center[0] - eye_offset_x, center[1] - eye_offset_y), eye_radius, (40, 40, 40), -1)
        cv2.circle(canvas, (center[0] + eye_offset_x, center[1] - eye_offset_y), eye_radius, (40, 40, 40), -1)
        mouth_y = int(center[1] + head_radius * 0.3)
        cv2.ellipse(
            canvas,
            (center[0], mouth_y),
            (int(head_radius * 0.4), int(head_radius * 0.25)),
            0,
            10,
            170,
            (80, 80, 80),
            1,
            lineType=cv2.LINE_AA,
        )

    # Neck line from head to torso.
    if "neck" in joints and "head" in joints:
        head = joints["head"]
        neck = joints["neck"]
        cv2.line(
            canvas,
            (int(neck.x), int(neck.y)),
            (int(head.x), int(head.y) + head_radius),
            (0, 120, 220),
            limb_thickness,
            lineType=cv2.LINE_AA,
        )

    # Joints as small circles (except head, already drawn).
    for name, pt in joints.items():
        if name == "head":
            continue
        cv2.circle(
            canvas,
            (int(pt.x), int(pt.y)),
            joint_radius,
            (255, 80, 80),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam stick-figure avatar driven by MediaPipe Pose.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the Avatar View to a video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="avatar_recording.mp4",
        help="Output video filename when --record is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.", file=sys.stderr)
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    smoother = PoseSmoother(alpha=0.75)
    writer = None

    print("[INFO] Webcam avatar running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            avatar_canvas = np.ones_like(frame) * 255  # white background

            if results.pose_landmarks:
                h, w, _ = frame.shape
                joints_px = get_landmark_dict(results.pose_landmarks, w, h)

                if joints_px:
                    smoothed_px = smoother.update(joints_px)
                    avatar_joints = normalize_to_avatar_space(
                        smoothed_px, avatar_size=(w, h)
                    )
                    draw_avatar(avatar_canvas, avatar_joints)

                # Draw skeleton overlay on webcam view for reference.
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            # Lazy-init writer after we know canvas size.
            if args.record and writer is None:
                h, w, _ = avatar_canvas.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 25.0, (w, h))
                if not writer.isOpened():
                    print("[WARN] Could not open video writer; recording disabled.", file=sys.stderr)
                    writer = None
                else:
                    print(f"[INFO] Recording Avatar View to {args.output}")

            if writer is not None:
                writer.write(avatar_canvas)

            cv2.imshow("Webcam View (press q to quit)", frame)
            cv2.imshow("Avatar View (press q to quit)", avatar_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pose.close()


if __name__ == "__main__":
    main()
