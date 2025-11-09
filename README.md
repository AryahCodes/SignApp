# Sign Language Learning App

An interactive ASL (American Sign Language) learning application using hand tracking and machine learning.

## 🎯 Features
- Real-time hand tracking with MediaPipe
- Letter recognition (A-Z)
- Word recognition
- Interactive learning modes

## 🛠️ Tech Stack
- **Frontend**: React, Socket.IO, Webcam
- **Backend**: Python, Flask, MediaPipe, TensorFlow
- **ML**: MediaPipe Hands, Custom gesture recognition models

## 📦 Setup Instructions

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 📊 Project Status
🚧 **Phase 1**: Environment Setup - In Progress

## 📚 Resources
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [React Documentation](https://react.dev/)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)

## 👤 Author
AryahCodes

## 📝 License
This project is open source and available under the MIT License.
```

3. Save (Cmd+S)

---

### **STEP 2: Create .gitignore**

1. Right-click in the empty space in the EXPLORER panel (left side)
2. Click "New File"
3. Name it `.gitignore`
4. Paste this content:
```
# Python
backend/venv/
backend/__pycache__/
backend/*.pyc
backend/*.pyo
backend/*.pyd
backend/.Python
backend/env/
backend/*.egg-info/
backend/dist/
backend/build/

# Node
frontend/node_modules/
frontend/build/
frontend/.env.local
frontend/.env.development.local
frontend/.env.test.local
frontend/.env.production.local
frontend/npm-debug.log*
frontend/yarn-debug.log*
frontend/yarn-error.log*

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
*.env

# Models and data
backend/models/*.h5
backend/models/*.tflite
backend/data/raw/