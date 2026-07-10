#  RiskPilot AI

## AI-Based Predictive Public Safety Intelligence Platform

RiskPilot AI is a real-time crowd monitoring and predictive public safety system that combines **YOLOv8**, **ByteTrack**, **Machine Learning**, and **Flask** to detect people, analyze crowd behavior, and predict potential risk levels.

The system supports both **Live Camera Monitoring** and **Dataset Video Analysis** through an interactive web dashboard.

---

## Features

- Real-time person detection using YOLOv8
- Multi-object tracking using ByteTrack
- Crowd density estimation
- Moving vs stationary people analysis
- Average crowd movement speed calculation
- AI-based risk prediction using Random Forest
- Live Flask dashboard
- Event logging
- Risk trend graph
- Dataset video mode
- Live webcam mode

---

## Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

---

## Project Structure

```
RiskPilot/
│
├── src/
├── templates/
├── static/
├── models/
├── requirements.txt
└── README.md
```

---

## Dashboard Features

- Live Camera Feed
- Dataset Video Playback
- Crowd Density Monitoring
- Risk Prediction
- Event Logging
- Risk Trend Visualization
- AI Recommendations

---

## Risk Levels

| Risk Level | Description |
|------------|-------------|
| SAFE | Normal crowd activity |
| MODERATE | Increasing crowd density |
| HIGH | Dense crowd with high movement |
| CRITICAL | Potential crowd emergency |

---

## Installation

Clone the repository

```bash
git clone https://github.com/amulya-br/RiskPilot.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python src/flask_app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Future Improvements

- Deep Learning-based Risk Prediction
- Heatmap Visualization
- Face Blur for Privacy
- Cloud Deployment
- Email/SMS Alerts
- Multi-Camera Support

---

## Author

**Amulya B R**

Electrical and Electronics Engineering

AI | Computer Vision | Machine Learning

GitHub:
https://github.com/amulya-br
