import { RefObject, Dispatch, SetStateAction } from "react";
import "./ChatMessagePrompt.css";
/**
 * Provides scroll to bottom option for users when there are unread messages.
 *
 * @param chatBodyRef reference to the chat body
 * @param isScrolling boolean representing whether user is scrolling chat
 * @param setIsScrolling setter for tracking if user is scrolling
 * @param unreadCount number representing unread messages count
 */
declare const ChatMessagePrompt: ({ chatBodyRef, isScrolling, setIsScrolling, unreadCount, }: {
    chatBodyRef: RefObject<HTMLDivElement>;
    isScrolling: boolean;
    setIsScrolling: Dispatch<SetStateAction<boolean>>;
    unreadCount: number;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatMessagePrompt;
