import { Dispatch, SetStateAction } from "react";
import { Message } from "../types/Message";
/**
 * Creates the useMessages() hook to manage messages.
 */
type MessagesContextType = {
    messages: Message[];
    setMessages: Dispatch<SetStateAction<Message[]>>;
};
declare const MessagesContext: import("react").Context<MessagesContextType>;
declare const useMessages: () => MessagesContextType;
export { MessagesContext, useMessages };
