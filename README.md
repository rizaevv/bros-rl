AI Basketball Bot

Overview

AI Basketball Bot is a computer vision and reinforcement learning project that detects basketball gameplay elements in real-time, including players, the ball, hoops, and the scoreboard. It can automatically determine when a player scores, combining YOLOv8 object detection, ball tracking, and OCR for score recognition.

This project demonstrates advanced object detection, temporal analysis, and game-state understanding, making it ideal for AI research, gaming analytics, and autonomous sports applications.

Features

Object Detection: Detects players, ball, hoop, scoreboard, and enemies using YOLOv8.

Score Detection: Determines when a player scores by analyzing ball trajectory and scoreboard updates.

Reinforcement Learning Ready: Can integrate with RL agents for gameplay analysis or autonomous decision-making.

Supports FHD Video: Optimized for full-HD images and videos.

Installation

Clone the repository:

git clone https://github.com/yourusername/bros-ai-bot.git
cd bros-ai-bot


Create a virtual environment and activate it:

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS


Install required packages:

pip install --upgrade pip
pip install -r requirements.txt


Make sure you have PyTorch with CUDA installed if you want GPU acceleration.

Dataset

The repository supports custom datasets in YOLO format:

train/
    images/
    labels/
valid/
    images/
    labels/
test/
    images/
    labels/


Label format: one bounding box per line:
<class> <x_center> <y_center> <width> <height> (normalized 0–1)

Usage
1. Training YOLOv8
yolo detect train data=basketball.yaml model=yolov8n.pt epochs=100 imgsz=1080 device=0


device=0 for GPU, device=cpu for CPU.

Trained weights will be saved in runs/detect/train/weights/best.pt.

2. Running Inference
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("example_image.png")[0]
results.show()


Returns bounding boxes, class labels, and confidence scores.

3. Score Detection

Integrates ball tracking + OCR on scoreboard region.

Detects when a player scores automatically.

Project Structure
bros-ai-bot/
├── data/             # Sample images & labels
├── notebooks/        # Jupyter experiments
├── src/              # Scripts (training, inference, scoring)
├── models/           # Trained models
├── results/          # Example outputs (images, videos)
├── README.md
├── requirements.txt
└── LICENSE

Contributing

Contributions welcome!

Please submit bug reports or feature requests via GitHub Issues.

Code style: follow Python best practices and PEP8.

License

MIT License – see LICENSE file for details.
