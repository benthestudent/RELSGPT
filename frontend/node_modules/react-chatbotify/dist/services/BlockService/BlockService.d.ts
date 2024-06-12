import { Dispatch, SetStateAction } from "react";
import { Flow } from "../../types/Flow";
import { Params } from "../../types/Params";
/**
 * Handles the preprocessing within a block.
 *
 * @param flow conversation flow for the bot
 * @param path path associated with the current block
 * @param params contains userInput, prevPath and injectMessage that can be used/passed into attributes
 * @param setTextAreaDisabled sets the state of the textarea for user input
 * @param setTextAreaSensitiveMode sets the sensitive mode of the textarea for user input
 * @param setPaths updates the paths taken by the user
 * @param setTimeoutId sets the timeout id for the transition attribute if it is interruptable
 * @param handleActionInput handles action input from user
 */
export declare const preProcessBlock: (flow: Flow, path: keyof Flow, params: Params, setTextAreaDisabled: (inputDisabled: boolean) => void, setTextAreaSensitiveMode: (inputDisabled: boolean) => void, setPaths: Dispatch<SetStateAction<string[]>>, setTimeoutId: (timeoutId: ReturnType<typeof setTimeout>) => void, handleActionInput: (path: keyof Flow, userInput: string, sendUserInput: boolean) => Promise<void>) => Promise<void>;
/**
 * Handles the postprocessing within a block.
 *
 * @param flow conversation flow for the bot
 * @param path path associated with the current block
 * @param params contains utilities that can be used/passed into attributes
 * @param setPaths updates the paths taken by the user
 */
export declare const postProcessBlock: (flow: Flow, path: keyof Flow, params: Params, setPaths: Dispatch<SetStateAction<string[]>>) => Promise<boolean>;
