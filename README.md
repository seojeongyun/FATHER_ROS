# FATHER_ROS

## Overview

**ROS 기반 실시간 특정 인물 탐지 및 얼굴 식별 시스템**입니다.

카메라 영상에서 YOLOv6를 활용해 얼굴을 탐지하고, 탐지된 얼굴을 GroupFace 기반 Face Recognition 모델로 임베딩하여 사전에 등록된 Gallery의 얼굴과 비교합니다.

ROS의 Publisher / Subscriber 구조를 기반으로 영상 입력, 얼굴 탐지, Gallery 관리, 얼굴 유사도 계산을 각각의 Node로 분리하여 실시간 Pipeline을 구성했습니다.

---

## System Pipeline

```text
Camera / Drone
      │
      ▼
  ROS Image
      │
      ▼
FPS Controller
      │
      ▼
YOLOv6 Face Detector
      │
      ├──── Bounding Box Result ────▶ Detection Viewer
      │
      ▼
  Cropped Face
      │
      ▼
GroupFace-based
Face Recognition
      │
      ▼
Face Embedding
      │
      ▼
Gallery Feature Database
      │
      ▼
Similarity Matching
      │
      ▼
 Top-K Candidates
```

전체 시스템은 크게 **Face Detection**과 **Face Recognition** 단계로 구성됩니다.

---

## 1. Face Detection

카메라에서 입력된 영상에 **YOLOv6 기반 Face Detector**를 적용하여 영상 내 얼굴을 탐지합니다.

```text
Input Frame
    │
    ▼
YOLOv6
    │
    ▼
Face Bounding Box
    │
    ├── Detection Result
    │
    └── Cropped Face
```

탐지된 얼굴 영역은 이후 Face Recognition 모델에서 사용할 수 있도록 별도의 ROS Image Topic으로 전달합니다.

주요 설정값은 다음과 같습니다.

* YOLOv6 기반 Face Detection
* GPU Inference
* FP16 지원
* Confidence / IoU Threshold 기반 filtering
* 탐지 얼굴 Crop 및 별도 ROS Topic 전달

---

## 2. Face Recognition

YOLOv6에서 탐지한 얼굴을 **GroupFace 기반 Face Recognition 모델**에 입력합니다.

각 얼굴 이미지를 고차원 Face Representation으로 변환하고, 사전에 등록된 Gallery의 얼굴 representation과 비교하여 동일 인물 여부를 판단합니다.

```text
Cropped Face
     │
     ▼
Face Recognition Network
(GroupFace + Capsule)
     │
     ▼
Face Embedding
     │
     ▼
Similarity Matching
     │
     ▼
Top-5 Candidates
```

현재 구현에서는 Gallery에 등록된 얼굴들과 비교하여 **유사도가 높은 Top-5 인물과 score**를 반환합니다.

---

## 3. Gallery Management

식별 대상자의 얼굴을 Gallery에 등록하고 이후 검색 대상으로 사용할 수 있도록 구성했습니다.

```text
Detected Face
     │
     ▼
Register Face
     │
     ▼
Gallery Directory
     │
     ▼
Feature Extraction
     │
     ▼
Gallery Feature Database
```

새로운 얼굴이 Gallery에 추가되면 Face Recognition Node에 update signal을 전달하고, Gallery Feature Database를 다시 생성합니다.

이를 통해 프로그램을 종료하지 않고도 새로운 인물을 식별 대상으로 추가할 수 있습니다.

---

## 4. ROS Pipeline

각 기능을 독립적인 ROS Node로 구성하여 영상 처리 Pipeline을 연결했습니다.

### FPS Controller

```text
/camera/color/image_raw
          │
          ▼
    fps_controller
          │
          ▼
/fps_controller/image_raw
```

입력 영상을 Face Detector로 전달하는 역할을 담당합니다.

Drone 영상을 사용할 경우 `/drone/image_raw`를 입력으로 사용할 수 있도록 구성되어 있습니다.

### Face Detector

```text
/fps_controller/image_raw
          │
          ▼
      YOLOv6
          │
      ┌───┴────────────┐
      ▼                ▼
Detection Result    Cropped Face
```

주요 출력 Topic:

```text
/face_detector/image_result
/face_detector/cropped_face_for_similarity
/face_detector/cropped_face_for_update_gallery
```

### Face Similarity

```text
/face_detector/cropped_face_for_similarity
                  │
                  ▼
          Face Recognition
                  │
                  ▼
        Gallery Comparison
                  │
                  ▼
      Top-5 Similar Faces
```

Face Recognition 결과 이미지는 다음 Topic으로 publish됩니다.

```text
/calc_face_similarity/result_image
```

---

## 5. Drone Input

`drone.py`에는 DJI Tello Drone의 카메라 영상을 ROS Image Topic으로 전달하기 위한 기능이 구현되어 있습니다.

```text
DJI Tello
    │
    ▼
Drone Camera
    │
    ▼
/drone/image_raw
    │
    ▼
FPS Controller
```

이를 통해 고정형 카메라뿐만 아니라 이동형 Drone Camera를 입력 영상으로 사용할 수 있도록 구성했습니다.

---

## Video Demo

ROS 기반으로 구성한 **Face Detection → Face Recognition → Gallery Matching** 전체 Pipeline의 실제 동작을 아래 영상에서 확인할 수 있습니다.

<p align="center">
  <a href="https://www.youtube.com/watch?v=3rGlCjDPHTc">
    <img src="https://img.youtube.com/vi/3rGlCjDPHTc/maxresdefault.jpg" width="75%">
  </a>
</p>

<p align="center">
  <b>▶ Click to watch the FATHER_ROS Demo</b>
</p>

---

## Key Features

* **YOLOv6-based Face Detection**  
  영상 내 얼굴 Bounding Box 실시간 탐지

* **GroupFace-based Face Recognition**  
  탐지된 얼굴의 고차원 representation 생성 및 인물 식별

* **Gallery-based Identification**  
  사전 등록된 얼굴과의 유사도 비교를 통한 특정 인물 검색

* **Top-K Face Matching**  
  Gallery에서 유사도가 높은 인물 후보 검색

* **Dynamic Gallery Update**  
  실행 중 새로운 얼굴을 Gallery에 등록하고 feature database 갱신

* **ROS-based Modular Pipeline**  
  영상 입력, Detection, Recognition을 독립적인 ROS Node로 구성

* **Camera / Drone Input**  
  Astra Camera와 DJI Tello Camera 입력 지원

* **Real-time System Demo**  
  Face Detection, Recognition, Gallery Matching으로 이어지는 전체 ROS Pipeline의 실제 동작 시연

---

## Repository Structure

```text
FATHER_ROS/
├── YOLOv6/                   # Face Detection
├── groupface/                # Face Recognition
│
├── face_detector.py          # YOLOv6 Face Detection Node
├── calc_face_similarity.py   # Face Recognition & Similarity
├── detected_face_view.py     # Detection Viewer & Gallery Update
├── face_similrity_view.py    # Recognition Result Viewer
│
├── fps_controller.py         # Input Frame Controller
├── drone.py                  # DJI Tello Camera Node
│
├── ring_buffer.py            # Frame Buffer
├── keyboard.py               # Keyboard Input
├── function.py               # ROS / OpenCV Utilities
│
└── README.txt
```

---

## Tech Stack

`Python` · `PyTorch` · `ROS` · `OpenCV` · `YOLOv6` · `GroupFace` · `CUDA` · `DJI Tello`

---

# Usage

아래 내용은 기존 Repository의 실행 방법을 유지한 것입니다.

## 1. Make ROS Workspace

```bash
mkdir -p ~/ros_ws/src && cd ~/ros_ws/src
catkin_init_workspace

cd ~/ros_ws
catkin_make

source ./devel/setup.bash
```

---

## 2. Make ROS Package

```bash
cd ~/ros_ws/src

catkin_create_pkg video_stream rospy std_msgs roscpp
```

Python scripts는 다음 경로에 위치하도록 구성합니다.

```text
~/ros_ws/src/video_script/src
```

---

## 3. Run FATHER

각 명령은 별도의 Terminal에서 실행합니다.

### Terminal 0-6

모든 Terminal에서 먼저 ROS workspace 환경을 설정합니다.

```bash
cd ~/ros_ws
source ./devel/setup.bash
```

### Terminal 1 — ROS Master

```bash
roscore
```

### Terminal 2 — Astra Camera

```bash
roslaunch astra_camera astra.pro.plus.launch
```

Astra Camera ROS Driver:

https://github.com/orbbec/ros_astra_camera

### Terminal 3 — FPS Controller

```bash
rosrun video_stream fps_controller.py
```

### Terminal 4 — Face Detector

```bash
rosrun video_stream face_detector.py
```

### Terminal 5 — Detection Viewer

```bash
rosrun video_stream detected_face_view.py
```

### Terminal 6 — Face Similarity

```bash
rosrun video_stream calc_face_similarity.py
```

---

## Execution Flow

```text
Terminal 2
Astra Camera
     │
     ▼
Terminal 3
FPS Controller
     │
     ▼
Terminal 4
Face Detector
     │
     ├──────────────▶ Terminal 5
     │                Detection Viewer
     │
     ▼
Terminal 6
Face Similarity
```

---

## Notes

현재 코드는 연구 및 실험 환경을 기준으로 작성되어 있어 일부 파일에 다음과 같은 **환경 종속적인 절대 경로**가 포함되어 있습니다.

* YOLOv6 Weight
* GroupFace Weight
* Gallery Directory
* Probe Directory
* Python Module Path

다른 환경에서 실행할 경우 해당 경로를 수정해야 합니다.

특히 다음 파일의 경로 설정을 확인해야 합니다.

```text
face_detector.py
calc_face_similarity.py
detected_face_view.py
```
