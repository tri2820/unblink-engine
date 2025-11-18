import { sendJob, type SummaryBuilder } from "..";
import type { EngineToServer } from "../engine";
import { logger } from "../logger";
import { parseJsonFromString } from "./utils";


export async function summarize(media_id: string, sb: SummaryBuilder): Promise<EngineToServer | undefined> {
    try {
        // Summarize every 1 min of video
        const first = sb.media_units.at(0);
        const last = sb.media_units.at(-1);


        if (!first || !last) {
            // no-op
        } else {
            const time_span = last.at_time - first.at_time;
            const time_span_sec = Math.floor(time_span / 1000);
            const SUMMARY_INTERVAL_SEC = 60;
            if (time_span_sec < SUMMARY_INTERVAL_SEC) {
                logger.info({
                    event: 'media_units_summary_skip_too_short',
                    media_id,
                    length: sb.media_units.length,
                    first_unit: {
                        at_time: first.at_time,
                        id: first.id,
                    },
                    last_unit: {
                        at_time: last.at_time,
                        id: last.id,
                    },
                    time_span,
                    time_span_sec,
                    required_time_span_sec: SUMMARY_INTERVAL_SEC,
                })
                return;
            }
        }



        //         // test: summarize every 4 media units
        //         // if (sb.media_units.length < 4) return;


        logger.info({
            event: 'media_units_summary_start',
            media_id,
        })

        // Clone the summary builder to avoid mutating while sending job
        const _sb = structuredClone(sb);

        // Reset summary builder
        sb.media_units = [];

        const paragraphs = _sb.media_units.map((mu, i) => `${i}. ${new Date(mu.at_time).toISOString()}\n${mu.description}`).join('\n\n');
        const context = `
                ${_sb.rolling_summary ? `Previously, this were the context summaries of the video stream: ${JSON.stringify(_sb.rolling_summary, null, 4)}\n\n. Do NOT output this context again. Avoid being repetitive. Avoid background noise or previous inferences.` : ""}

                Here are the scenes of the video stream through the camera POV:\n\n${paragraphs}\n\nBased on the above descriptions, group closely related events into 'moments' and provide a JSON summary. Each moment should represent a distinct event or activity and have a start and end time. Each moment must also have its own list of descriptive labels.`;
        const trimmed_context = context.slice(0, 50000); // Trim to first 50,000 characters to avoid token limits


        // And here is the AI's response based on the new `trimmed_context`
        // MODIFIED: 'timestamp' is replaced with 'from_time' and 'to_time'. Moments are grouped.
        const example_ai_response = {
            "background": "A person is cooking in a kitchen. An unattended pan catches fire, which is then extinguished.",
            "moments": [
                {
                    "from_time": "2024-03-18T18:03:45.000Z",
                    "to_time": "2024-03-18T18:03:45.000Z",
                    what_old: "A kitchen, with a person cooking at the stove.",
                    "what_new": "A person leaves a pan unattended on an active stove, creating a potential fire risk.",
                    "importance_score": 0.5,
                    "labels": ["Unattended Cooking", "Potential Hazard"]
                },
                {
                    "from_time": "2024-03-18T18:05:20.000Z",
                    "to_time": "2024-03-18T18:06:10.000Z",
                    what_old: "The person leaves the kitchen, leaving a pan on the active stove.",
                    "what_new": "The unattended pan begins to smoke, ignites into a small fire triggering an alarm, and is then extinguished by the person.",
                    "importance_score": 1.0,
                    "labels": ["Smoke Detected", "Fire", "Alarm Triggered", "Intervention", "Emergency"]
                }
            ]
        };

        const example = [{
            role: 'user',
            content: `Here are the scenes from a real-time kitchen camera stream:

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
              "what_old": "A concise string describing OLD key events",
              "what_new": "A concise string describing NEW key events or just \"nothing new\"",
              "importance_score": 0.0,
              "labels": ["A list of strings that categorize this specific moment."]
            }
          ]
        }
        \`\`\`

        - The 'from_time' should be the timestamp of the first event in a moment.
        - The 'to_time' should be the timestamp of the last event in a moment. If a moment consists of a single event, 'from_time' and 'to_time' should be the same.
        - The 'importance_score' must be a floating-point number between 0.0 (mundane) and 1.0 (critical). For normal, non interesting moments, use scores closer to 0.0 (e.g. 0.1). For significant events (e.g., emergencies), use scores closer to 1.0 (e.g. 0.8).
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


        // FIX: Handle the case where the LLM returns an array
        let summary_data = summary_parsed.data;
        if (Array.isArray(summary_data) && summary_data.length > 0) {
            summary_data = summary_data[0];
        }

        // MODIFIED: Updated validation logic to check for 'from_time', 'to_time' and 'labels' array within each moment.
        if (typeof summary_data.background !== 'string' ||
            !Array.isArray(summary_data.moments) ||
            !summary_data.moments.every((moment: any) =>
                typeof moment.from_time === 'string' &&
                typeof moment.to_time === 'string' &&
                typeof moment.what_new === 'string' &&
                typeof moment.what_old === 'string' &&
                typeof moment.importance_score === 'number' &&
                Array.isArray(moment.labels) &&
                moment.labels.every((label: any) => typeof label === 'string')
            )
        ) {
            throw new Error("Summary JSON does not conform to the expected structure (with from_time, to_time, and per-moment labels).");
        }

        sb.rolling_summary = summary_data;

        return {
            type: "media_summary",
            media_id,
            summary: summary_data,
        }
    } catch (error) {
        logger.error({
            event: 'media_units_summary_error',
            media_id,
            error_msg: error instanceof Error ? error.message : String(error),
        }, 'Error generating media summary');
        return undefined;
    }
}