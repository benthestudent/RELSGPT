import { Dispatch, SetStateAction } from "react";
import { Options } from "../types/Options";
/**
 * Creates the useBotOptions() hook to manage bot options.
 */
type BotOptionsContextType = {
    botOptions: Options;
    setBotOptions: Dispatch<SetStateAction<Options>>;
};
declare const BotOptionsContext: import("react").Context<BotOptionsContextType>;
declare const useBotOptions: () => BotOptionsContextType;
export { BotOptionsContext, useBotOptions };
