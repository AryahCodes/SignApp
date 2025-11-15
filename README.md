# 🤟 SignApp - Real-Time ASL Alphabet Recognition

![Demo Screenshot](path/to/screenshot.png)
![socketio-flowchart](https://github.com/user-attachments/assets/e96965f2-350e-4d2b-b0c2-09914535341e)<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1200">
  <defs>
    <style>
      .box { fill: #4A90E2; stroke: #2E5C8A; stroke-width: 2; }
      .process { fill: #50C878; stroke: #2E8B57; stroke-width: 2; }
      .data { fill: #FFB347; stroke: #E67E22; stroke-width: 2; }
      .decision { fill: #9B59B6; stroke: #6C3483; stroke-width: 2; }
      .text { fill: white; font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }
      .label { fill: #333; font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
      .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
      .dotted { stroke-dasharray: 5,5; }
      .title { fill: #2C3E50; font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; text-anchor: middle; }
      .section-title { fill: #34495E; font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#333" />
    </marker>
  </defs>

  <!-- Title -->
  <text x="400" y="30" class="title">Socket.IO Flow in Sign Language App</text>

  <!-- FRONTEND SECTION -->
  <text x="200" y="70" class="section-title">FRONTEND (React)</text>
  
  <!-- User opens app -->
  <rect x="100" y="90" width="200" height="60" rx="10" class="box"/>
  <text x="200" y="115" class="text">User Opens App</text>
  <text x="200" y="135" class="text">(Webcam Activated)</text>

  <!-- Socket.IO Connect -->
  <rect x="100" y="180" width="200" height="60" rx="10" class="process"/>
  <text x="200" y="205" class="text">Socket.IO Connect</text>
  <text x="200" y="225" class="text">io.connect(backend_url)</text>

  <!-- MediaPipe Processing -->
  <rect x="100" y="320" width="200" height="80" rx="10" class="data"/>
  <text x="200" y="345" class="text">MediaPipe Extracts</text>
  <text x="200" y="365" class="text">Hand Landmarks</text>
  <text x="200" y="385" class="text">(21 points, 3D coords)</text>

  <!-- Emit to Backend -->
  <rect x="100" y="430" width="200" height="60" rx="10" class="process"/>
  <text x="200" y="455" class="text">socket.emit('predict',</text>
  <text x="200" y="475" class="text">{landmarks: [...]})</text>

  <!-- Receive Prediction -->
  <rect x="100" y="650" width="200" height="60" rx="10" class="process"/>
  <text x="200" y="675" class="text">socket.on('prediction',</text>
  <text x="200" y="695" class="text">callback)</text>

  <!-- Update UI -->
  <rect x="100" y="740" width="200" height="60" rx="10" class="box"/>
  <text x="200" y="765" class="text">Update UI</text>
  <text x="200" y="785" class="text">(Display Letter)</text>

  <!-- BACKEND SECTION -->
  <text x="600" y="70" class="section-title">BACKEND (Flask)</text>

  <!-- Socket.IO Server -->
  <rect x="500" y="180" width="200" height="60" rx="10" class="process"/>
  <text x="600" y="205" class="text">Socket.IO Server</text>
  <text x="600" y="225" class="text">Accepts Connection</text>

  <!-- Receive Data -->
  <rect x="500" y="430" width="200" height="60" rx="10" class="process"/>
  <text x="600" y="455" class="text">@socketio.on('predict')</text>
  <text x="600" y="475" class="text">Receives landmarks</text>

  <!-- Feature Engineering -->
  <rect x="500" y="520" width="200" height="70" rx="10" class="data"/>
  <text x="600" y="540" class="text">Feature Engineering</text>
  <text x="600" y="560" class="text">(72 features: distances,</text>
  <text x="600" y="580" class="text">angles, z-score norm)</text>

  <!-- ML Model -->
  <rect x="500" y="620" width="200" height="60" rx="10" class="decision"/>
  <text x="600" y="645" class="text">TensorFlow Model</text>
  <text x="600" y="665" class="text">Predicts Letter</text>

  <!-- Emit Result -->
  <rect x="500" y="710" width="200" height="60" rx="10" class="process"/>
  <text x="600" y="735" class="text">emit('prediction',</text>
  <text x="600" y="755" class="text">{letter: 'A', conf: 0.98})</text>

  <!-- Arrows - Frontend Flow -->
  <path d="M 200 150 L 200 180" class="arrow"/>
  <path d="M 200 240 L 200 320" class="arrow"/>
  <path d="M 200 400 L 200 430" class="arrow"/>
  <path d="M 200 490 L 200 520" class="arrow"/>
  <path d="M 200 710 L 200 740" class="arrow"/>
  <path d="M 200 800 L 200 830" class="arrow"/>

  <!-- Arrows - Backend Flow -->
  <path d="M 600 240 L 600 270" class="arrow"/>
  <path d="M 600 490 L 600 520" class="arrow"/>
  <path d="M 600 590 L 600 620" class="arrow"/>
  <path d="M 600 680 L 600 710" class="arrow"/>

  <!-- Cross arrows -->
  <!-- Connect event -->
  <path d="M 300 210 L 500 210" class="arrow"/>
  <text x="400" y="200" class="label">Connection established</text>

  <!-- Data send -->
  <path d="M 300 460 L 500 460" class="arrow"/>
  <text x="400" y="450" class="label">Send landmark data</text>

  <!-- Response -->
  <path d="M 500 740 L 300 680" class="arrow"/>
  <text x="400" y="720" class="label">Return prediction</text>

  <!-- Loop arrow -->
  <path d="M 200 830 L 50 830 L 50 340 L 100 340" class="arrow dotted"/>
  <text x="30" y="580" class="label">Continuous</text>
  <text x="30" y="600" class="label">Loop</text>
  <text x="30" y="620" class="label">(30-60 FPS)</text>

  <!-- Real-time indicator -->
  <rect x="320" y="270" width="160" height="180" rx="5" fill="none" stroke="#E74C3C" stroke-width="3" stroke-dasharray="10,5"/>
  <text x="400" y="290" style="fill: #E74C3C; font-size: 14px; font-weight: bold; text-anchor: middle;">REAL-TIME CHANNEL</text>
  <text x="400" y="310" style="fill: #E74C3C; font-size: 11px; text-anchor: middle;">WebSocket Connection</text>
  <text x="400" y="328" style="fill: #E74C3C; font-size: 11px; text-anchor: middle;">Stays Open</text>

  <!-- Performance metrics -->
  <rect x="250" y="900" width="300" height="120" rx="10" fill="#ECF0F1" stroke="#95A5A6" stroke-width="2"/>
  <text x="400" y="930" style="fill: #2C3E50; font-size: 16px; font-weight: bold; text-anchor: middle;">Performance Achieved</text>
  <text x="400" y="955" style="fill: #34495E; font-size: 13px; text-anchor: middle;">⚡ Latency: &lt;100ms</text>
  <text x="400" y="975" style="fill: #34495E; font-size: 13px; text-anchor: middle;">🎯 Accuracy: 96.86%</text>
  <text x="400" y="995" style="fill: #34495E; font-size: 13px; text-anchor: middle;">🔄 Processing: 30-60 frames/sec</text>

  <!-- Legend -->
  <rect x="50" y="1050" width="700" height="120" rx="10" fill="#F8F9FA" stroke="#BDC3C7" stroke-width="2"/>
  <text x="400" y="1075" style="fill: #2C3E50; font-size: 14px; font-weight: bold; text-anchor: middle;">Component Legend</text>
  
  <rect x="80" y="1090" width="40" height="20" rx="3" class="box"/>
  <text x="130" y="1105" style="fill: #333; font-size: 12px;">User Action</text>
  
  <rect x="230" y="1090" width="40" height="20" rx="3" class="process"/>
  <text x="280" y="1105" style="fill: #333; font-size: 12px;">Socket Event</text>
  
  <rect x="380" y="1090" width="40" height="20" rx="3" class="data"/>
  <text x="430" y="1105" style="fill: #333; font-size: 12px;">Data Processing</text>
  
  <rect x="550" y="1090" width="40" height="20" rx="3" class="decision"/>
  <text x="600" y="1105" style="fill: #333; font-size: 12px;">ML Model</text>

  <text x="80" y="1135" style="fill: #7F8C8D; font-size: 11px; font-style: italic;">30-60x per second loop</text>
  
  <path d="M 270 1125 L 330 1125" class="arrow"/>
  <text x="335" y="1130" style="fill: #7F8C8D; font-size: 11px;">Data Flow</text>
  
  <path d="M 450 1125 L 510 1125" class="arrow dotted"/>
  <text x="515" y="1130" style="fill: #7F8C8D; font-size: 11px;">Loop Back</text>
</svg>

https://signapp-frontend.vercel.app/

An interactive American Sign Language (ASL) learning application using real-time hand tracking and deep learning for letter recognition.

## 🎯 Features

- **Real-time Hand Tracking** with MediaPipe
- **ASL Alphabet Recognition** (A-Y, 24 letters)
- **96.86% Test Accuracy** using professional deep learning model
- **Interactive Learning Modes**
  - Hand Tracking: Real-time letter recognition
  - Training Mode: Collect custom training data
- **Professional ML Pipeline**
  - Z-score normalization for camera-distance invariance
  - Temporal smoothing for stable predictions
  - Frame buffering for robust recognition
  - SigNN-inspired neural network architecture

## 🚀 Tech Stack

### Frontend
- React + TypeScript
- Socket.IO for real-time communication
- MediaPipe Hands (browser-based hand tracking)
- Tailwind CSS

### Backend
- Python + Flask
- TensorFlow/Keras (Deep Learning)
- MediaPipe (Hand landmark detection)
- Socket.IO
- Eventlet (Async support)

### ML Architecture
- **Model:** Deep Neural Network (SigNN-based)
  - 900 → 400 → 200 → 24 neurons
  - Batch normalization + Dropout
  - ReLU and Tanh activations
- **Features:** 78 engineered features
  - Z-score normalized coordinates
  - Finger angles and extension ratios
  - Inter-finger spacing
  - Hand direction and palm size
- **Accuracy:** 96.86% on test set (9,572 samples)

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.86% |
| Real-time FPS | 10 FPS |
| Confidence (avg) | 65-75% |
| Letters Supported | 24 (A-Y) |
| Training Samples | 9,572 |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 🎮 Usage

### Start Backend
```bash
cd backend
source venv/bin/activate
python server.py
```

### Start Frontend
```bash
cd frontend
npm start
```

Visit `http://localhost:3000`

## 🧠 How It Works

### 1. Hand Tracking
MediaPipe detects 21 hand landmarks in real-time from webcam feed

### 2. Feature Extraction
- Extract 78 features from landmarks
- Apply z-score normalization
- Calculate finger angles and extensions

### 3. Prediction Pipeline
```
Raw Frame → MediaPipe → Landmarks → Feature Extraction → 
Z-Score Norm → Frame Buffer (10 frames) → Model Prediction → 
Temporal Smoothing (7 frames) → Confidence Threshold (50%) → Display
```

### 4. Model Architecture
```python
Input (78 features)
    ↓
Dense(900) + BatchNorm + Dropout(0.15)
    ↓
Dense(400) + BatchNorm + Dropout(0.25)
    ↓
Dense(200) + Dropout(0.4)
    ↓
Dense(24) + Softmax
```

## 📚 Training Your Own Model

### Using Kaggle Data (Recommended)
```bash
cd backend
python train_professional_kaggle.py
```

### Using Custom Data
1. Go to Training Mode in the app
2. Collect 30-50 samples per letter
3. Click "Train Model"

## 🎨 Project Structure
```
SignApp/
├── backend/
│   ├── server.py                      # Flask server + Socket.IO
│   ├── hand_processor.py              # MediaPipe hand tracking
│   ├── feature_extractor.py           # Feature engineering
│   ├── professional_letter_classifier.py  # TensorFlow model
│   ├── train_professional_kaggle.py   # Training script
│   └── models/
│       ├── professional_model.h5      # Trained model
│       └── professional_labels.json   # Label mappings
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── HandTracking.jsx       # Main recognition UI
│   │   │   └── TrainingMode.jsx       # Data collection UI
│   │   └── App.jsx
│   └── package.json
└── README.md
```

## 🔬 Technical Deep Dive

### Z-Score Normalization
Removes camera distance variance by normalizing landmark coordinates:
```python
x_normalized = (x - mean(x)) / std(x)
```

### Temporal Smoothing
Uses sliding window (7 frames) to require consistent predictions:
- Letter must appear in 40%+ of window
- Confidence must average > 50%

### Frame Buffering
Averages landmarks over 10 frames before prediction for stability

## 🐛 Known Issues

- Letters D, M, N, G have lower accuracy (~60-70%)
  - These letters have very similar hand shapes
  - Industry-wide challenge in ASL recognition
- Dynamic letters J and Z not supported (require temporal LSTM)

## 🚀 Future Enhancements

- [ ] Word mode (string multiple letters)
- [ ] Phrase recognition
- [ ] LSTM for dynamic gestures (J, Z)
- [ ] Multi-hand support
- [ ] Mobile app deployment
- [ ] User progress tracking

## 📖 References

- [SigNN Research Paper](https://github.com/AriAlavi/SigNN)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [FreeCodeCamp ASL Tutorial](https://www.freecodecamp.org/news/create-a-real-time-gesture-to-text-translator/)

## 👨‍💻 Author

[Aryahvishwa Babu](https://github.com/AryahCodes)

## 📄 License

MIT License

## 🙏 Acknowledgments

- Kaggle ASL Alphabet Dataset
- SigNN Research Team
- MediaPipe Team
- FreeCodeCamp Community
