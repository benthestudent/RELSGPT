import { RefObject, Dispatch, SetStateAction } from "react";
import "./ChatBotBody.css";
/**
 * Contains chat messages between the user and bot.
 *
 * @param chatBodyRef reference to the chat body
 * @param isBotTyping boolean indicating if bot is typing
 * @param isLoadingChatHistory boolean indicating is chat history is being loaded
 * @param chatScrollHeight number representing chat window scroll height
 * @param setChatScrollHeight setter for tracking chat window scroll height
 * @param setIsLoadingChatHistory setter for tracking whether is loading history
 * @param isScrolling boolean representing whether user is scrolling chat
 * @param setIsScrolling setter for tracking if user is scrolling
 * @param unreadCount number representing unread messages count
 * @param setUnreadCount setter for unread messages count
 */
declare const ChatBotBody: ({ chatBodyRef, isBotTyping, isLoadingChatHistory, chatScrollHeight, setChatScrollHeight, setIsLoadingChatHistory, isScrolling: isScrolling, setIsScrolling, unreadCount, setUnreadCount, }: {
    chatBodyRef: RefObject<HTMLDivElement>;
    isBotTyping: boolean;
    isLoadingChatHistory: boolean;
    chatScrollHeight: number;
    setChatScrollHeight: Dispatch<SetStateAction<number>>;
    setIsLoadingChatHistory: Dispatch<SetStateAction<boolean>>;
    isScrolling: boolean;
    setIsScrolling: Dispatch<SetStateAction<boolean>>;
    unreadCount: number;
    setUnreadCount: Dispatch<SetStateAction<number>>;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatBotBody;
