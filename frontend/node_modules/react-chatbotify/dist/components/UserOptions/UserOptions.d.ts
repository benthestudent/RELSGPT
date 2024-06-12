import "./UserOptions.css";
import { Flow } from "../../types/Flow";
/**
 * Supports showing of options for user to select.
 *
 * @param options array representing options to show
 * @param path path associated with the current block
 * @param handleActionInput handles input (selected option) from user
 */
declare const UserOptions: ({ options, path, handleActionInput }: {
    options: string[];
    path: keyof Flow;
    handleActionInput: (path: keyof Flow, userInput: string, sendUserInput: boolean) => Promise<void>;
}) => import("react/jsx-runtime").JSX.Element;
export default UserOptions;
