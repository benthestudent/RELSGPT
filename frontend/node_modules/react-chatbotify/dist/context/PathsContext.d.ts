import { Dispatch, SetStateAction } from "react";
import { Flow } from "../types/Flow";
/**
 * Creates the usePaths() context hook to manage user paths.
 */
type PathsContextType = {
    paths: (keyof Flow)[];
    setPaths: Dispatch<SetStateAction<string[]>>;
};
declare const PathsContext: import("react").Context<PathsContextType>;
declare const usePaths: () => PathsContextType;
export { PathsContext, usePaths };
