import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';

function WebcamCapture() {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>Sign Language App - Webcam Test</h1>
      
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        style={{
          width: '640px',
          height: '480px',
          border: '2px solid #333',
          borderRadius: '10px'
        }}
      />
      
      <br />
      
      <button 
        onClick={capture}
        style={{
          padding: '10px 20px',
          fontSize: '16px',
          marginTop: '10px',
          cursor: 'pointer'
        }}
      >
        Capture Photo
      </button>

      {imgSrc && (
        <div style={{ marginTop: '20px' }}>
          <h3>Captured Image:</h3>
          <img 
            src={imgSrc} 
            alt="captured" 
            style={{
              width: '320px',
              border: '2px solid #333',
              borderRadius: '10px'
            }}
          />
        </div>
      )}
    </div>
  );
}

export default WebcamCapture;