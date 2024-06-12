import { Dispatch, SetStateAction } from "react";
import { Flow } from "../../types/Flow";
import { Params } from "../../types/Params";
/**
 * Handles processing of transition in current block.
 *
 * @param messages messages containing current conversation with the bot
 * @param flow conversation flow for the bot
 * @param path path associated with the current block
 * @param params contains userInput, prevPath and injectMessage that can be used/passed into attributes
 * @param setPaths updates the paths taken by the user
 * @param setTimeoutId sets the timeout id for the transition attribute if it is interruptable
 */
export declare const processTransition: (flow: Flow, path: keyof Flow, params: Params, setPaths: Dispatch<SetStateAction<string[]>>, setTimeoutId: (timeoutId: ReturnType<typeof setTimeout>) => void) => Promise<void>;
