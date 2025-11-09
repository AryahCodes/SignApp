from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

print("=" * 50)
print("🚀 Sign Language App - Backend Server")
print("=" * 50)

@app.route('/test')
def test():
    return {"message": "Backend is working!", "status": "success"}

@app.route('/')
def home():
    return {"message": "Sign Language App API", "version": "1.0"}

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected!')
    emit('response', {'message': 'Connected to backend server!'})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected!')

@socketio.on('test_message')
def handle_test_message(data):
    print(f'📩 Received message: {data}')
    emit('response', {'message': f'Echo: {data}'})

if __name__ == '__main__':
    print("\n🌐 Server starting on http://localhost:5001")
    print("📡 Socket.IO enabled for real-time communication")
    print("Press CTRL+C to stop\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)    