import { Flow } from "../types/Flow";
import "./ChatBotContainer.css";
/**
 * Integrates and contains the various components that makeup the chatbot.
 *
 * @param flow conversation flow for the bot
 */
declare const ChatBotContainer: ({ flow }: {
    flow: Flow;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatBotContainer;
