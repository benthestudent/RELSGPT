import { Block } from "../../types/Block";
import { Params } from "../../types/Params";
/**
 * Handles processing of function in current block.
 *
 * @param block current block being processed
 * @param params contains userInput, prevPath and injectMessage that can be used/passed into attributes
 */
export declare const processFunction: (block: Block, params: Params) => Promise<void>;
