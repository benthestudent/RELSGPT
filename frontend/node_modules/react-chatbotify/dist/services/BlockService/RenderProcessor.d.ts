import { Block } from "../../types/Block";
import { Params } from "../../types/Params";
/**
 * Handles processing of render in current block.
 *
 * @param block current block being processed
 * @param params contains userInput, prevPath and injectMessage that can be used/passed into attributes
 */
export declare const processRender: (block: Block, params: Params) => Promise<void>;
