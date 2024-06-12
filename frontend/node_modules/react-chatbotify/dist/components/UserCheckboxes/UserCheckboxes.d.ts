import "./UserCheckboxes.css";
import { Flow } from "../../types/Flow";
/**
 * Supports showing of checkboxes for user to mark.
 *
 * @param checkboxes object representing checkboxes to show and min/max number of selections
 * @param checkedItems set representing items marked
 * @param path path associated with the current block
 * @param handleActionInput handles input (marked checkboxes) from user
 */
declare const UserCheckboxes: ({ checkboxes, checkedItems, path, handleActionInput, }: {
    checkboxes: {
        items: Array<string>;
        max?: number;
        min?: number;
    };
    checkedItems: Set<string>;
    path: keyof Flow;
    handleActionInput: (path: keyof Flow, userInput: string, sendUserInput: boolean) => Promise<void>;
}) => import("react/jsx-runtime").JSX.Element;
export default UserCheckboxes;
