# 🏀 AI Basketball Bot

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU-orange) ![License](https://img.shields.io/badge/License-MIT-green)


---

## 🌟 Overview

AI Basketball Bot is a **state-of-the-art computer vision and reinforcement learning project** designed to detect basketball gameplay elements in real-time. Using YOLOv8 for object detection, ball tracking, and OCR for score recognition, the bot can **automatically detect when a player scores** and provides a foundation for AI-driven gameplay analytics and autonomous decision-making.

This project demonstrates advanced AI skills suitable for research, gaming analytics, and real-time applications, making it a standout project for college admissions and portfolios.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🏀 Object Detection | Detects ball, players, enemies, hoops, and scoreboard using YOLOv8. |
| 📊 Score Detection | Combines ball trajectory + OCR on scoreboard to determine scored points automatically. |
| 🎮 Reinforcement Learning Ready | Can integrate with RL agents for automated gameplay strategies. |
| 📹 Full-HD Support | Optimized for FHD images and video streams. |
| ⚡ GPU Acceleration | Leverages CUDA for fast real-time inference. |

---

## 🛠 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/bros-ai-bot.git
cd bros-ai-bot
```

2️⃣ Create a virtual environment
```python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate
```
3️⃣ Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

**🚀 Training YOLOv8**
```
yolo detect train data=basketball.yaml model=yolov8n.pt epochs=150 imgsz=1080 device=0
```

**📂PROJECT STRUCTURE**
```
bros-ai-bot/
├── data/           # Sample images & labels
├── notebooks/      # Jupyter experiments and visualizations
├── src/            # Scripts: train_model.py, detect.py, score_detection.py
├── models/         # Trained models: best.pt, last.pt
├── results/        # Example outputs: images, videos, GIFs
├── README.md       # This file
├── requirements.txt
└── LICENSE
```


💡 Contributing

Contributions welcome!

Submit issues or feature requests on GitHub.

Follow PEP8 and Python best practices.
