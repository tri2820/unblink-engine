import type { ServerWebSocket } from "bun";
import { decode, encode } from "cbor-x";
import fs from 'fs';
import path from 'path';
import { sendJob, type Client } from "..";
import { ensureDirExists, FRAMES_DIR } from "../appdir";
import type { EngineToServer, ServerToEngine } from "../engine";
import { logger } from "../logger";
import type { EngineReceivedMessage, RegistrationMessage, WorkerToEngine } from "../shared";
import { summarize } from "./summary";

const FRAME_SIZE_LIMIT = 2 * 1024 * 1024; // 2 MB

async function handle_register(props: {
    decoded: RegistrationMessage,
    job_map: Map<string, { cont: (result: Record<string, any>) => void }>,
    client: Client,
    ws: ServerWebSocket<unknown>,
}) {
    const { decoded, client, ws, job_map } = props;
    if (decoded.type === "i_am_worker") {
        if (!decoded.worker_config || !decoded.worker_config.worker_type) {
            console.error("Invalid worker_config from worker.", decoded);
            return;
        }

        if (decoded.worker_secret !== process.env.WORKER_SECRET) {
            console.error("Invalid WORKER_SECRET from worker.");
            ws.close(1008, "Invalid WORKER_SECRET");
            return;
        }

        logger.info({
            event: 'worker_registration',
            c_id: client.id,
            worker_config: decoded.worker_config
        }, 'Registered worker', client.id, decoded.worker_config);
        client.worker_config = {
            max_batch_size: 32,
            max_latency_ms: 30000,
            worker_type: decoded.worker_config.worker_type,
            gathered: [],
        };

        client.worker_config = { ...client.worker_config, ...decoded.worker_config };
        return;
    }

    // === Worker registration and job distribution ===
    if (decoded.type === "i_am_server") {
        logger.info({
            event: 'server_registration',
            c_id: client.id
        }, 'Registered server client', client.id);
        client.server_config = {
            tenant_id: client.id, // For now, use client ID as tenant ID
            state: {},
        };

        // TODO: validate token if provided

        return;
    }
}

async function handle__workerMessage(props: {
    decoded: WorkerToEngine,
    job_map: Map<string, { cont: (result: Record<string, any>) => void }>,
    client: Client,
    ws: ServerWebSocket<unknown>,
}) {
    const { decoded, client, ws, job_map } = props;

    if (decoded.type === "worker_output") {
        if (!client.worker_config) {
            console.error("Received worker_output from non-worker client.", decoded);
            ws.close(1008, "Not a worker");
            return;
        }

        const outputs = decoded.output as any[];
        // Sanity check
        if (!outputs || !Array.isArray(outputs)) return;

        for (const output of outputs) {
            const job = job_map.get(output.id);
            if (!job) {
                logger.error("No job found for worker output ID:", output.id);
                continue;
            }
            job.cont(output);
        }
        return;
    }
}
async function handle__serverMessage(props: {
    decoded: ServerToEngine,
    job_map: Map<string, { cont: (result: Record<string, any>) => void }>,
    client: Client,
    ws: ServerWebSocket<unknown>,
}) {
    const { decoded, client, ws } = props;

    if (!client.server_config) {
        console.error("Received server message from non-server client.", decoded);
        ws.close(1008, "Not a server");
        return;
    }

    const tenant_id = client.server_config.tenant_id;

    if (!tenant_id) {
        console.error("Received frame_binary from unauthenticated client.", decoded);
        ws.close(1008, "Unauthenticated");
        return;
    }

    if (decoded.type === "frame_binary") {
        if (client.server_config.state[decoded.stream_id] === undefined) {
            client.server_config.state[decoded.stream_id] = {
                summary_builder: {
                    media_units: [],
                }
            }
        }

        if (decoded.frame.byteLength > FRAME_SIZE_LIMIT) {
            console.error("Frame size exceeds limit of", FRAME_SIZE_LIMIT, "bytes.");
            ws.close(1009, "Frame size exceeds limit");
            return;
        }

        // Create job for processing frame
        const FRAMES_DIR_TENANT = FRAMES_DIR(tenant_id);
        await ensureDirExists(FRAMES_DIR_TENANT);
        // This ensures unique filenames even if frames arrive quickly, for proper cleanup
        const nonce = crypto.randomUUID();
        const file_path = path.join(FRAMES_DIR_TENANT, `${decoded.frame_id}-${nonce}.jpg`);
        // logger.info({
        //     event: 'frame_received',
        //     c_id: client.id,
        //     size: decoded.frame.byteLength,
        //     file_path
        // });
        fs.writeFileSync(file_path, decoded.frame);

        let __cleanupCountdown = Object.entries(decoded.workers).filter(x => x[1]).length;

        // Cleanup frame file after all workers have responded
        const countdown = () => {
            __cleanupCountdown--;
            if (__cleanupCountdown <= 0) {
                fs.unlink(file_path, (err) => {
                    if (err) {
                        console.error("Failed to delete frame file:", file_path, err);
                    }
                });
            }
        };

        // Requested VLM worker
        if (decoded.workers.vlm) {

            // const example = [
            //     {
            //         role: 'user', content: [
            //             { type: 'text', text: 'Describe in detailed. Do NOT say anything about being blurry. Try to describe what looks like inside. Previous description: A garage with a red car. The car seems to be broken. The door is openning' }
            //         ]
            //     },
            //     {
            //         role: 'assistant', content: [
            //             { type: 'text', text: 'A person walked in with a wrench in his hand.' }
            //         ]
            //     },
            // ]

            // const last_media_unit = client.server_config.state[decoded.stream_id]?.last_media_unit;
            // const image_description_job = {
            //     messages: [
            //         {
            //             role: 'system',
            //             content: [
            //                 { type: 'text', text: `Describe what you see in detailed. Do NOT say anything about being blurry.` },
            //             ]
            //         },
            //         // ...example,
            //         {
            //             "role": "user",
            //             "content": [
            //                 ...(last_media_unit ? [
            //                     {
            //                         type: "text", text: `Previous description so we can skip describing these again: ${JSON.stringify(last_media_unit.description, null, 4)}. Do not be repetitive. Do not output noise or background again. \n\nFocusing on changes or new elements.`
            //                     }
            //                 ] : []),
            //                 { type: "text", text: "Describe what you see in detailed:" },
            //                 { "type": "image", "image": file_path },
            //             ]
            //         }
            //     ]
            // };

            const image_caption_job = {
                "image_path": file_path
            }

            const at_time = new Date().getTime();
            sendJob(image_caption_job, 'caption', {
                async cont(output) {
                    let description = output.response;
                    // Try to remove common prefixes
                    // E.g., "This image depicts ...", "This image captures ...", "In this image, ", "The image shows ...", "The image captures ..."
                    const prefixes = [
                        "This is an image of a ",
                        "The image is ",
                        "This image depicts ",
                        "This image captures ",
                        "In this image, ",
                        "The image shows ",
                        "The image captures ",
                        "This photo depicts ",
                        "This photo captures ",
                        "In this photo, ",
                        "The photo shows ",
                        "The photo captures ",
                    ];
                    for (const prefix of prefixes) {
                        if (description.startsWith(prefix)) {
                            description = description.slice(prefix.length);
                            // Properly capitalize first letter
                            description = description.charAt(0).toUpperCase() + description.slice(1);
                            break;
                        }
                    }

                    const msg: EngineToServer = {
                        type: "frame_description",
                        frame_id: decoded.frame_id,
                        stream_id: decoded.stream_id,
                        description,
                    }
                    logger.info({
                        event: 'frame_description',
                        c_id: client.id,
                        msg
                    });
                    const encoded = encode(msg);
                    client.ws.send(encoded);
                    countdown();

                    // Update last media unit if it's older than the current one
                    const last_media_unit_at_time = client.server_config!.state[decoded.stream_id]!.last_media_unit?.at_time || 0;
                    if (at_time >= last_media_unit_at_time) {
                        client.server_config!.state[decoded.stream_id]!.last_media_unit = {
                            id: decoded.frame_id,
                            description,
                            at_time,
                        };
                    }

                    // Add to summary builder
                    const sb = client.server_config!.state[decoded.stream_id]!.summary_builder;
                    sb.media_units.push({
                        id: decoded.frame_id,
                        description,
                        at_time,
                    });

                    const summary_msg = await summarize(decoded.stream_id, sb);
                    if (summary_msg) {
                        const encoded = encode(summary_msg);
                        client.ws.send(encoded);
                    }
                }
            });
        }

        // Requested embedding worker
        if (decoded.workers.embedding) {
            const image_embedding_job = {
                "filepath": file_path
            }

            sendJob(image_embedding_job, 'embedding', {
                async cont(output) {
                    const msg: EngineToServer = {
                        type: "frame_embedding",
                        frame_id: decoded.frame_id,
                        stream_id: decoded.stream_id,
                        embedding: output.embedding,
                    }
                    // logger.info({
                    //     event: 'frame_embedding',
                    //     c_id: client.id,
                    //     msg: '[embedding data]',
                    // });
                    const encoded = encode(msg);
                    client.ws.send(encoded);
                    countdown();
                }
            });
        }

        if (decoded.workers.object_detection) {
            const object_detection_job = {
                "filepath": file_path
            }

            sendJob(object_detection_job, 'object_detection', {
                async cont(output) {
                    const msg: EngineToServer = {
                        type: "frame_object_detection",
                        frame_id: decoded.frame_id,
                        stream_id: decoded.stream_id,
                        objects: output.detections,
                    }
                    // logger.info({
                    //     event: 'frame_object_detection',
                    //     c_id: client.id,
                    //     msg: {
                    //         ...msg,
                    //         objects: `[${output.detections.length} objects]`
                    //     }
                    // });
                    const encoded = encode(msg);
                    // logger.info({
                    //     frame_id: decoded.frame_id,
                    //     detections_length: output.detections.length,
                    //     job_id: (object_detection_job as any).id
                    // }, "Sending object detection result:");
                    client.ws.send(encoded);

                    countdown();
                }
            });
        }
        return;
    }
}


export function createWsMessageHandler(clients$: () => Map<ServerWebSocket<unknown>, Client>, job_map$: () => Map<string, { cont: (result: Record<string, any>) => void }>) {
    return async function handleWsMessage(ws: ServerWebSocket<unknown>, message: string | Buffer<ArrayBuffer>) {
        const clients = clients$();
        const job_map = job_map$();

        let decoded: EngineReceivedMessage
        try {
            decoded = decode(message as any);
        } catch (error) {
            console.error("Failed to decode message:", error, message);
            return;
        }
        const client = clients.get(ws);
        if (!client) return;

        if (decoded.type === "i_am_worker" || decoded.type === "i_am_server") {
            await handle_register({ decoded, job_map, client, ws });
            return;
        }

        if (client.worker_config) {
            await handle__workerMessage({ decoded: decoded as any, job_map, client, ws });
            return;
        }

        if (client.server_config) {
            await handle__serverMessage({ decoded: decoded as any, job_map, client, ws });
            return;
        }
        // === Worker outputs ===
        console.error("Received message from unauthenticated and non-worker client.", decoded);
        ws.close(1008, "Unauthenticated");

    }
}