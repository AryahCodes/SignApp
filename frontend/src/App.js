import React, { useState } from 'react';
import HandTracking from './HandTracking';
import WebcamCapture from './WebcamCapture';
import SocketTest from './SocketTest';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('tracking'); // 'tracking', 'test', 'socket'

  return (
    <div className="App">
      {/* Navigation */}
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        padding: '15px',
        borderRadius: '10px',
        margin: '20px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        display: 'flex',
        gap: '10px',
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={() => setCurrentView('tracking')}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            fontWeight: 'bold',
            cursor: 'pointer',
            backgroundColor: currentView === 'tracking' ? '#007bff' : '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '8px'
          }}
        >
          🤟 Hand Tracking
        </button>
        
        <button
          onClick={() => setCurrentView('test')}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            fontWeight: 'bold',
            cursor: 'pointer',
            backgroundColor: currentView === 'test' ? '#007bff' : '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '8px'
          }}
        >
          📸 Webcam Test
        </button>
        
        <button
          onClick={() => setCurrentView('socket')}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            fontWeight: 'bold',
            cursor: 'pointer',
            backgroundColor: currentView === 'socket' ? '#007bff' : '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '8px'
          }}
        >
          🔌 Socket Test
        </button>
      </div>

      {/* Content */}
      <div style={{ 
        backgroundColor: 'white',
        borderRadius: '15px',
        padding: '20px',
        margin: '20px',
        boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
      }}>
        {currentView === 'tracking' && <HandTracking />}
        {currentView === 'test' && <WebcamCapture />}
        {currentView === 'socket' && <SocketTest />}
      </div>
    </div>
  );
}

export default App;