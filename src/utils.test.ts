import { describe, test, expect } from "bun:test";
import { parseJsonFromString } from "./utils";

describe("parseJsonFromString", () => {
    // --- No changes needed for success cases ---
    describe("Successful Parsing - Objects", () => { /* ... */ });
    describe("Successful Parsing - Arrays", () => { /* ... */ });
    describe("Edge Cases", () => { /* ... */ });

    // --- UPDATED Failure Cases ---
    describe("Failure Cases and Invalid Input", () => {
        test("should return an error for a string with no JSON structure", () => {
            const input = "This is just a regular sentence.";
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            // UPDATE: The new generic error message is more accurate.
            expect(result.error).toBe("No valid JSON object or array found in the string.");
        });

        test("should return an error for an empty string", () => {
            const input = "";
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            expect(result.error).toBe("Input must be a non-empty string.");
        });

        test("should return an error for a null input", () => {
            // @ts-ignore
            const result = parseJsonFromString(null);
            expect(result).toHaveProperty("error");
            expect(result.error).toBe("Input must be a non-empty string.");
        });

        test("should return an error for an incomplete object (missing closing brace)", () => {
            const input = 'Some text {"key": "value" and more';
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            // UPDATE: The function fails because it never finds a balanced structure to parse.
            expect(result.error).toBe("No valid JSON object or array found in the string.");
        });

        test("should return an error for an incomplete array (missing closing bracket)", () => {
            const input = 'Data: [1, 2, 3';
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            // UPDATE: Same as above, no balanced structure is ever found.
            expect(result.error).toBe("No valid JSON object or array found in the string.");
        });

        test("should return a parsing error for malformed JSON (trailing comma)", () => {
            const input = 'This is invalid: [1, 2, 3, ]';
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            // UPDATE: The new function provides a more detailed error.
            expect(result.error).toStartWith("Malformed JSON found. Details:");
        });

        test("should return a parsing error for malformed JSON (single quotes)", () => {
            const input = "{'key': 'value'}"; // JSON spec requires double quotes
            const result = parseJsonFromString(input);
            expect(result).toHaveProperty("error");
            // UPDATE: Same as above, a detailed parsing error is now returned.
            expect(result.error).toStartWith("Malformed JSON found. Details:");
        });
    });
});