# Real-Time Motion Detection System

A real-time computer vision application using **OpenCV frame analysis and background modeling** to detect movement from a live camera feed.

## Project Overview

This project demonstrates a practical approach to **motion detection using computer vision**.

A camera continuously captures video frames and the system analyzes changes between the current scene and the previously observed background. Significant changes are interpreted as motion and highlighted in the live video output.

### System Concept

```text
Camera
   ↓
Video Frame Capture
   ↓
Background Analysis
   ↓
Frame Difference
   ↓
Thresholding
   ↓
Noise Filtering
   ↓
Contour Detection
   ↓
Motion Identification
   ↓
Real-Time Visualization
