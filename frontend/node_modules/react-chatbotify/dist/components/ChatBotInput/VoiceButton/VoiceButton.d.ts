import { RefObject, Dispatch, SetStateAction } from "react";
import "./VoiceButton.css";
/**
 * Toggles voice to text input to the chat bot.
 *
 * @param inputRef reference to the textarea
 * @param textAreaDisabled boolean indicating if textarea is disabled
 * @param voiceToggledOn boolean indicating if voice is toggled on
 * @param handleToggleVoice handles toggling of voice
 * @param triggerSendVoiceInput triggers sending of voice input into chat window
 * @param setInputLength sets the input length to reflect character count & limit
 */
declare const VoiceButton: ({ inputRef, textAreaDisabled, voiceToggledOn, handleToggleVoice, triggerSendVoiceInput, setInputLength }: {
    inputRef: RefObject<HTMLTextAreaElement | HTMLInputElement>;
    textAreaDisabled: boolean;
    voiceToggledOn: boolean;
    handleToggleVoice: () => void;
    triggerSendVoiceInput: () => void;
    setInputLength: Dispatch<SetStateAction<number>>;
}) => import("react/jsx-runtime").JSX.Element;
export default VoiceButton;
