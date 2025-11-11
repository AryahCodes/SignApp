const BACKEND_URL = process.env.NODE_ENV === 'production' 
  ? 'https://signapp-backend.onrender.com'  // Your Render URL
  : 'http://localhost:5001';

export default BACKEND_URL;