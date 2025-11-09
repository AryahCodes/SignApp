import React from 'react';
import WebcamCapture from './WebcamCapture';
import SocketTest from './SocketTest';
import './App.css';

function App() {
  return (
    <div className="App">
      <div style={{ 
        backgroundColor: 'white',
        borderRadius: '15px',
        padding: '20px',
        margin: '20px',
        boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
      }}>
        <WebcamCapture />
      </div>
      
      <SocketTest />
    </div>
  );
}

export default App;