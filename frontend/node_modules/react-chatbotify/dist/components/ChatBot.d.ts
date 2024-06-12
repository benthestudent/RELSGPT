import { Options } from "../types/Options";
import { Flow } from "../types/Flow";
/**
 * Initializes providers for chatbot.
 *
 * @param flow conversation flow for the bot
 * @param options options to setup the bot
 */
declare const ChatBot: ({ flow, options }: {
    flow?: Flow | undefined;
    options?: Options | undefined;
}) => import("react/jsx-runtime").JSX.Element;
export default ChatBot;
