import "./ChatBotButton.css";
/**
 * Toggles opening and closing of the chat window when theme.embedded is false.
 *
 * @param unreadCount number of unread messages from the bot
 */
declare const ChatBotButton: ({ unreadCount }: {
    unreadCount: number;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatBotButton;
