import io from 'socket.io-client';

// More aggressive connection settings
const socket = io('http://localhost:5001', {
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 500,
  reconnectionAttempts: 10,
  timeout: 20000,
  forceNew: true
});

socket.on('connect', () => {
  console.log('✅ Socket connected! ID:', socket.id);
});

socket.on('disconnect', (reason) => {
  console.log('❌ Socket disconnected! Reason:', reason);
});

socket.on('connect_error', (error) => {
  console.error('❌ Connection error:', error.message);
});

socket.on('reconnect_attempt', () => {
  console.log('🔄 Attempting to reconnect...');
});

socket.on('reconnect', () => {
  console.log('✅ Reconnected!');
});

export default socket;