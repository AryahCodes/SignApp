from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet
from hand_processor import HandProcessor

# Patch eventlet
eventlet.monkey_patch()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Socket.IO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=10 * 1024 * 1024  # 10MB for images
)

# Initialize hand processor
hand_processor = HandProcessor()

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

@socketio.on('process_frame')
def handle_process_frame(data):
    """Process webcam frame for hand detection"""
    try:
        frame_data = data.get('frame')
        
        # Process the frame
        result = hand_processor.process_frame(frame_data)
        
        # Send results back to client
        emit('hand_landmarks', result)
        
        # Log detection
        if result['hands_detected'] > 0:
            print(f'👋 Detected {result["hands_detected"]} hand(s)')
            
    except Exception as e:
        print(f'❌ Error in process_frame: {str(e)}')
        emit('hand_landmarks', {
            'success': False,
            'error': str(e),
            'hands_detected': 0,
            'hands': []
        })

@socketio.on_error_default
def default_error_handler(e):
    print(f'⚠️  Socket.IO Error: {str(e)}')

if __name__ == '__main__':
    print("\n🌐 Server starting on http://localhost:5001")
    print("📡 Socket.IO enabled for real-time communication")
    print("🤖 MediaPipe hand tracking active")
    print("Press CTRL+C to stop\n")
    
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5001,
            debug=True,
            use_reloader=True
        )
    finally:
        hand_processor.cleanup()