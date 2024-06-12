import logo from './logo.svg';
//import ChatBot from "react-chatbotify";
import Chat from './chatbot';
import './App.css';

function App() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: '100%',
    }}>
      <Chat />
    </div>
  );
}

export default App;
