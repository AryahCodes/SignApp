from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet
from hand_processor import HandProcessor
from letter_classifier import LetterClassifier
from data_collector import DataCollector

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

# Initialize components
hand_processor = HandProcessor()
letter_classifier = LetterClassifier()
data_collector = DataCollector()

# Try to load existing model
letter_classifier.load_model()

print("=" * 50)
print("🚀 Sign Language App - Backend Server")
print("=" * 50)

@app.route('/test')
def test():
    return {"message": "Backend is working!", "status": "success"}

@app.route('/')
def home():
    return {"message": "Sign Language App API", "version": "1.0"}

@app.route('/model/status')
def model_status():
    """Get model training status"""
    sample_counts = data_collector.get_sample_counts()
    return {
        'is_trained': letter_classifier.is_trained,
        'labels': sorted(letter_classifier.labels) if letter_classifier.is_trained else [],
        'sample_counts': sample_counts
    }

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected!')
    emit('response', {'message': 'Connected to backend server!'})
    
    # Send model status
    sample_counts = data_collector.get_sample_counts()
    emit('model_status', {
        'is_trained': letter_classifier.is_trained,
        'labels': sorted(letter_classifier.labels) if letter_classifier.is_trained else [],
        'sample_counts': sample_counts
    })

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
        
        # If hand detected and classifier is trained, predict letter
        if result['success'] and result['hands_detected'] > 0 and letter_classifier.is_trained:
            # Use first hand for prediction
            first_hand = result['hands'][0]
            landmarks = first_hand['landmarks']
            
            # Predict letter
            prediction = letter_classifier.predict(landmarks)
            
            if prediction['success']:
                result['letter_prediction'] = prediction
        
        # Send results back to client
        emit('hand_landmarks', result)
        
        # Log detection
        if result['hands_detected'] > 0:
            if letter_classifier.is_trained and 'letter_prediction' in result:
                pred = result['letter_prediction']
                print(f'👋 Detected {result["hands_detected"]} hand(s) - Predicted: {pred["letter"]} ({pred["confidence"]:.2%})')
            else:
                print(f'👋 Detected {result["hands_detected"]} hand(s)')
            
    except Exception as e:
        print(f'❌ Error in process_frame: {str(e)}')
        emit('hand_landmarks', {
            'success': False,
            'error': str(e),
            'hands_detected': 0,
            'hands': []
        })

@socketio.on('save_training_sample')
def handle_save_sample(data):
    """Save a training sample"""
    try:
        landmarks = data.get('landmarks')
        label = data.get('label', '').upper()
        
        if not landmarks or not label:
            emit('sample_saved', {
                'success': False,
                'error': 'Missing landmarks or label'
            })
            return
        
        # Save the sample
        filepath = data_collector.save_sample(landmarks, label)
        
        # Get updated counts
        sample_counts = data_collector.get_sample_counts()
        
        print(f'💾 Saved training sample: {label} (total: {sample_counts.get(label, 0)})')
        
        emit('sample_saved', {
            'success': True,
            'label': label,
            'filepath': filepath,
            'sample_counts': sample_counts
        })
        
    except Exception as e:
        print(f'❌ Error saving sample: {str(e)}')
        emit('sample_saved', {
            'success': False,
            'error': str(e)
        })

@socketio.on('train_model')
def handle_train_model(data):
    """Train the letter classifier"""
    try:
        print('🎓 Training model...')
        
        # Load all samples
        landmarks_list, labels_list = data_collector.load_all_samples()
        
        if len(landmarks_list) == 0:
            emit('training_complete', {
                'success': False,
                'error': 'No training samples found'
            })
            return
        
        # Train the classifier
        success = letter_classifier.train(landmarks_list, labels_list)
        
        if success:
            # Save the model
            letter_classifier.save_model()
            
            emit('training_complete', {
                'success': True,
                'message': f'Model trained on {len(landmarks_list)} samples',
                'labels': sorted(letter_classifier.labels),
                'sample_count': len(landmarks_list)
            })
            
            print(f'✅ Model trained successfully!')
        else:
            emit('training_complete', {
                'success': False,
                'error': 'Training failed'
            })
            
    except Exception as e:
        print(f'❌ Error training model: {str(e)}')
        emit('training_complete', {
            'success': False,
            'error': str(e)
        })

@socketio.on_error_default
def default_error_handler(e):
    print(f'⚠️  Socket.IO Error: {str(e)}')

if __name__ == '__main__':
    print("\n🌐 Server starting on http://localhost:5001")
    print("📡 Socket.IO enabled for real-time communication")
    print("🤖 MediaPipe hand tracking active")
    print("🎓 Letter recognition ready")
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