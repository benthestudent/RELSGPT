import { Dispatch, SetStateAction } from "react";
import { Block } from "../../types/Block";
import { Params } from "../../types/Params";
/**
 * Handles processing of path in current block.
 *
 * @param block current block being processed
 * @param params contains userInput, prevPath and injectMessage that can be used/passed into attributes
 * @param setPaths updates the paths taken by the user
 */
export declare const processPath: (block: Block, params: Params, setPaths: Dispatch<SetStateAction<string[]>>) => Promise<boolean>;
