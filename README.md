# Webcam Avatar Lab

This project is a clean, self-contained **webcam avatar** demo.

It uses **MediaPipe Pose** (a pre-trained deep neural network) to detect your
body landmarks from a single webcam, and renders a simple 2D humanoid
stick-figure that mirrors your movements in real time. There is no physics
engine involved, so the avatar does not fall over or behave unpredictably.

The goal is to give you a visually clear, smooth, and reliable mirroring
experience that you can show on your GitHub as an AI + computer-vision project.

## Features

- Single-camera body tracking using **MediaPipe Pose**.
- Real-time **2D avatar** that mirrors:
  - Head, shoulders, elbows, wrists
  - Hips, knees, ankles
- Two synchronized views:
  - Webcam feed with skeleton overlay.
  - Clean avatar view drawn on a blank canvas.
- Basic temporal smoothing to reduce jitter.
- Centered avatar: movement is shown as pose changes around a stable origin,
  not as the whole figure sliding off-screen.

## Setup

From the project root (`rl-locomotion-starter`):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# or on Windows
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r webcam-avatar/requirements.txt
```

This will install:

- `opencv-python`
- `mediapipe`
- `numpy`

## Usage

Run the avatar demo from the project root:

```bash
python webcam-avatar/webcam_avatar.py
```

You should see two windows:

- **Webcam View**: your live camera with a skeleton drawn on your body.
- **Avatar View**: a stick-figure humanoid on a plain background that mirrors
  your current pose.

Controls:

- `q` - quit both windows.

Tips for best results:

- Make sure your upper body and, if possible, your legs are fully visible in
  the webcam frame.
- Stand at a consistent distance from the camera.
- Avoid very fast, abrupt movements if you want the avatar to look smooth.

### Recording the avatar

To save the avatar view to a video file while it tracks your movements:

```bash
python webcam-avatar/webcam_avatar.py --record --output avatar_session.mp4
```

This records only the **Avatar View** window to the given output file
(`avatar_session.mp4` by default if `--output` is not provided).

## How it works (high level)

1. Each frame, the script grabs an image from your webcam.
2. MediaPipe Pose runs a neural network to detect 3D body landmarks.
3. We select key joints (head, shoulders, elbows, wrists, hips, knees, ankles)
   and convert them into 2D coordinates.
4. A small smoothing filter is applied over time so joints do not jitter.
5. The avatar view is drawn by:
   - Centering on your hip position.
   - Scaling and drawing lines between joints to form a stick-figure.

There is no RL or training loop here: all "intelligence" is in the pre-trained
pose estimator plus our deterministic retargeting and drawing code.

This makes the behaviour predictable and easy to reason about, while still
being a genuine AI/computer-vision showcase.
