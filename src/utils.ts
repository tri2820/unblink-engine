/**
 * Flexibly extracts and parses a JSON from a raw string,
 * even if it's embedded in markdown code blocks or other text.
 *
 * It works by finding the first occurrence of '[' and the last occurrence
 * of ']' and attempting to parse everything in between as a JSON.
 *
 * @param text The raw string that may contain a JSON.
 * @returns The parsed array if found and valid, otherwise an empty array.
 *          The elements of the array are of type `any`.
 */
export function parseJsonFromString(text: string): any {
    // Use a regex to find content between the first '[' and the last ']'.
    // The 's' flag (dotAll) allows '.' to match newline characters,
    // which is equivalent to Python's re.DOTALL.
    const match = text.match(/\[.*\]/s);

    if (match) {
        // The matched string is at index 0 of the result array.
        const jsonStr = match[0];
        try {
            const result = JSON.parse(jsonStr);
            return result;
        } catch (error) {
            // The extracted string was not valid JSON.
            // You can optionally log the error for debugging purposes.
            // console.error("Failed to parse JSON from string:", error);
            return [];
        }
    } else {
        // No JSON array-like structure was found in the string.
        return [];
    }
}