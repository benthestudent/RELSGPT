import { RefObject } from "react";
import "./EmojiPicker.css";
/**
 * Supports selecting of emojis.
 *
 * @param inputRef reference to the textarea
 * @param textAreaDisabled boolean indicating if textarea is disabled
 */
declare const EmojiPicker: ({ inputRef, textAreaDisabled }: {
    inputRef: RefObject<HTMLTextAreaElement | HTMLInputElement>;
    textAreaDisabled: boolean;
}) => import("react/jsx-runtime").JSX.Element;
export default EmojiPicker;
