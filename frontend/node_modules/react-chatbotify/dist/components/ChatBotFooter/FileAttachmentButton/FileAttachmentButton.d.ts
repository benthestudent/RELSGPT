import { RefObject } from "react";
import { Flow } from "../../../types/Flow";
import "./FileAttachmentButton.css";
/**
 * Supports uploading of files from user.
 *
 * @param inputRef reference to the textarea
 * @param flow conversation flow for the bot
 * @param injectMessage utility function for injecting a message into the messages array
 * @param streamMessage utility function for streaming a message into the messages array
 * @param openChat utility function to open/close chat window
 * @param getCurrPath retrieves current path for the user
 * @param getPrevPath retrieves previous path for the user
 * @param handleActionInput handles action input from user
 */
declare const FileAttachmentButton: ({ inputRef, flow, injectMessage, streamMessage, openChat, getCurrPath, getPrevPath, handleActionInput }: {
    inputRef: RefObject<HTMLTextAreaElement | HTMLInputElement>;
    flow: Flow;
    injectMessage: (content: string | JSX.Element, sender?: string) => Promise<void>;
    streamMessage: (content: string | JSX.Element, sender?: string) => Promise<void>;
    openChat: (isOpen: boolean) => void;
    getCurrPath: () => keyof Flow | null;
    getPrevPath: () => keyof Flow | null;
    handleActionInput: (path: keyof Flow, userInput: string, sendUserInput?: boolean) => Promise<void>;
}) => import("react/jsx-runtime").JSX.Element;
export default FileAttachmentButton;
