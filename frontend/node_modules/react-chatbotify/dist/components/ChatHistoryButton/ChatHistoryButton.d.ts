import "./ChatHistoryButton.css";
/**
 * Supports viewing of old messages.
 *
 * @param chatHistory string representation of old chat messages
 * @param showChatHistory entry point for showing of chat history
 */
declare const ChatHistoryButton: ({ chatHistory, showChatHistory }: {
    chatHistory: string;
    showChatHistory: (chatHistory: string) => void;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatHistoryButton;
