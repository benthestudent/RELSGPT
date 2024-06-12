import { MouseEvent } from "react";
import "./SendButton.css";
/**
 * Sends current user input to the chat bot.
 *
 * @param handleSubmit handles submission of user input
 */
declare const SendButton: ({ handleSubmit }: {
    handleSubmit: (event: MouseEvent) => void;
}) => import("react/jsx-runtime").JSX.Element;
export default SendButton;
