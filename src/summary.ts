import { sendJob, type SummaryBuilder } from "..";
import type { EngineToServer } from "../engine";
import { logger } from "../logger";
import { parseJsonFromString } from "./utils";


export async function summarize(media_id: string, sb: SummaryBuilder): Promise<EngineToServer | undefined> {
    try {
        // Summarize every 5 minutes of video
        // if (sb.latest_to_time - sb.earliest_from_time < 5 * 60 * 1000) return;

        logger.info({
            event: 'media_units_summary_check',
            media_id,
            media_units_length: sb.media_units.length,
        })

        // test: summarize every 4 media units
        if (sb.media_units.length < 4) return;

        // 
        const _sb = structuredClone(sb);

        // Reset summary builder
        sb.media_units = [];

        const paragraphs = _sb.media_units.map((mu, i) => `${i}. ${new Date(mu.at_time).toISOString()}\n${mu.description}`).join('\n\n');
        const context = `Here are some descriptions of the video stream through the camera POV:\n\n${paragraphs}\n\nBased on the above descriptions, group closely related events into 'moments' and provide a JSON summary. Each moment should represent a distinct event or activity and have a start and end time. Each moment must also have its own list of descriptive labels.`;
        const trimmed_context = context.slice(0, 50000); // Trim to first 50,000 characters to avoid token limits


        // And here is the AI's response based on the new `trimmed_context`
        // MODIFIED: 'timestamp' is replaced with 'from_time' and 'to_time'. Moments are grouped.
        const example_ai_response = {
            "background": "A person is cooking in a kitchen. An unattended pan catches fire, which is then extinguished.",
            "moments": [
                {
                    "from_time": "2024-03-18T18:03:45.000Z",
                    "to_time": "2024-03-18T18:03:45.000Z",
                    "description": "A person leaves a pan unattended on an active stove, creating a potential fire risk.",
                    "importance_score": 0.5,
                    "labels": ["Unattended Cooking", "Potential Hazard"]
                },
                {
                    "from_time": "2024-03-18T18:05:20.000Z",
                    "to_time": "2024-03-18T18:06:10.000Z",
                    "description": "The unattended pan begins to smoke, ignites into a small fire triggering an alarm, and is then extinguished by the person.",
                    "importance_score": 1.0,
                    "labels": ["Smoke Detected", "Fire", "Alarm Triggered", "Intervention", "Emergency"]
                }
            ]
        };

        const example = [{
            role: 'user',
            content: `Here are some descriptions of a real-time kitchen camera stream:

1. 2024-03-18T18:01:15.000Z
A person is cooking at the stove. Everything appears normal.

2. 2024-03-18T18:03:45.000Z
The person leaves the kitchen, leaving a pan on the active stove.

3. 2024-03-18T18:05:20.000Z
Smoke begins to rise from the unattended pan.

4. 2024-03-18T18:05:55.000Z
A small flame ignites in the pan. The smoke detector is triggered.

5. 2024-03-18T18:06:10.000Z
The person rushes back into the kitchen with a fire extinguisher and puts out the flame.

Based on the above descriptions, group related, sequential events into 'moments' and provide a structured JSON object. The JSON must strictly follow this format:

\`\`\`json
{
  "background": "A brief string describing the overall setting and context.",
  "moments": [
    {
      "from_time": "YYYY-MM-DDTHH:mm:ss.sssZ",
      "to_time": "YYYY-MM-DDTHH:mm:ss.sssZ",
      "description": "A concise string describing a key event or a sequence of related events.",
      "importance_score": 0.0,
      "labels": ["A list of strings that categorize this specific moment."]
    }
  ]
}
\`\`\`

- The 'from_time' should be the timestamp of the first event in a moment.
- The 'to_time' should be the timestamp of the last event in a moment. If a moment consists of a single event, 'from_time' and 'to_time' should be the same.
- The 'importance_score' must be a floating-point number between 0.0 (mundane) and 1.0 (critical).
- Each moment must include a 'labels' array containing one or more relevant string tags.
- Do not include any other keys or explanatory text outside of the JSON object.`
        }, {
            role: 'assistant',
            // MODIFIED: The example response is stringified here
            content: JSON.stringify(example_ai_response, null, 4)
        }]



        const messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes video streams into a specific JSON format. You will only output valid JSON."
            },
            ...example,
            {
                "role": "user",
                "content": trimmed_context
            }
        ];
        const summarize_job = {
            messages
        };

        logger.info({
            event: 'media_units_summary_start_generate',
            media_id,
            summarize_job
        })

        const output = await new Promise<Record<string, any>>((resolve) => {
            sendJob(summarize_job, 'llm', {
                async cont(output) {
                    resolve(output);
                }
            });
        })

        const summary_str = output.response;
        const summary_parsed = parseJsonFromString(summary_str);
        logger.info({
            event: 'media_units_summary_generated',
            media_id,
            summary: summary_parsed.data
        })

        if (!summary_parsed.data) {
            throw new Error(`Failed to parse summary response as JSON: ${summary_parsed.error}`);
        }

        // MODIFIED: Updated validation logic to check for 'from_time', 'to_time' and 'labels' array within each moment.
        if (typeof summary_parsed.data.background !== 'string' ||
            !Array.isArray(summary_parsed.data.moments) ||
            !summary_parsed.data.moments.every((moment: any) =>
                typeof moment.from_time === 'string' &&
                typeof moment.to_time === 'string' &&
                typeof moment.description === 'string' &&
                typeof moment.importance_score === 'number' &&
                Array.isArray(moment.labels) &&
                moment.labels.every((label: any) => typeof label === 'string')
            )
        ) {
            throw new Error("Summary JSON does not conform to the expected structure (with from_time, to_time, and per-moment labels).");
        }


        return {
            type: "media_summary",
            media_id,
            summary: summary_parsed.data,
        }
    } catch (error) {
        logger.error({
            event: 'media_units_summary_error',
            media_id,
            error
        }, 'Error generating media summary');
        return undefined;
    }
}