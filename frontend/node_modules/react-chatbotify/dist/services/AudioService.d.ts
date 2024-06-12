import { Message } from "../types/Message";
import { Options } from "../types/Options";
/**
 * Handles logic for whether a bot message should be read out.
 *
 * @param botOptions options provide to the bot
 * @param voiceToggledOn boolean indicating if voice is toggled on
 * @param message message to read out
 */
export declare const processAudio: (botOptions: Options, voiceToggledOn: boolean, message: Message) => void;
