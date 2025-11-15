import type { ServerWebSocket } from "bun";
import { decode, encode } from "cbor-x";
import fs from 'fs';
import path from 'path';
import { sendJob, type Client } from "..";
import { ensureDirExists, FRAMES_DIR } from "../appdir";
import type { EngineReceivedMessage, EngineToServer } from "../engine";
import { logger } from "../logger";

const FRAME_SIZE_LIMIT = 2 * 1024 * 1024; // 2 MB

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

        // === Worker registration and job distribution ===
        if (decoded.type === "i_am_worker") {
            if (!decoded.worker_config || !decoded.worker_config.worker_type) {
                console.error("Invalid worker_config from worker.", decoded);
                return;
            }

            if (decoded.secret !== process.env.WORKER_SECRET) {
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

        if (decoded.type === "i_am_server") {
            logger.info({
                event: 'server_registration',
                c_id: client.id
            }, 'Registered server client', client.id);
            client.server_config = {
                tenant_id: client.id, // For now, use client ID as tenant ID
            };

            // TODO: validate token if provided

            return;
        }

        if (decoded.type === "frame_binary") {
            const tenant_id = client.server_config?.tenant_id;
            if (!tenant_id) {
                console.error("Received frame_binary from unauthenticated client.", decoded);
                ws.close(1008, "Unauthenticated");
                return;
            }

            if (decoded.frame.byteLength > FRAME_SIZE_LIMIT) {
                console.error("Frame size exceeds limit of", FRAME_SIZE_LIMIT, "bytes.");
                ws.close(1009, "Frame size exceeds limit");
                return;
            }

            // Create job for processing frame
            const FRAMES_DIR_TENANT = FRAMES_DIR(tenant_id);
            await ensureDirExists(FRAMES_DIR_TENANT);
            const file_path = path.join(FRAMES_DIR_TENANT, `${decoded.frame_id}.jpg`);
            logger.info({
                event: 'frame_received',
                c_id: client.id,
                size: decoded.frame.byteLength,
                file_path
            });
            fs.writeFileSync(file_path, decoded.frame);

            // Requested VLM worker
            if (decoded.workers.vlm) {
                const image_description_job = {
                    messages: [
                        {
                            role: 'system',
                            content: [
                                { type: 'text', text: `Describe this image in detailed. Do NOT say anything about the image being blurry. Try to describe what looks like inside.` },
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                { "type": "image", "image": file_path },
                            ]
                        }
                    ]
                };

                sendJob(image_description_job, 'vlm', {
                    async cont(output) {
                        const msg: EngineToServer = {
                            type: "frame_description",
                            frame_id: decoded.frame_id,
                            stream_id: decoded.stream_id,
                            description: output.description,
                        }
                        logger.info({
                            event: 'frame_description',
                            c_id: client.id,
                            msg
                        });
                        const encoded = encode(msg);
                        client.ws.send(encoded);
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
                        logger.info({
                            event: 'frame_embedding',
                            c_id: client.id,
                            msg: '[embedding data]',
                        });
                        const encoded = encode(msg);
                        client.ws.send(encoded);
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
                        logger.info({
                            event: 'frame_object_detection',
                            c_id: client.id,
                            msg: {
                                ...msg,
                                objects: `[${output.detections.length} objects]`
                            }
                        });
                        const encoded = encode(msg);
                        logger.info({
                            frame_id: decoded.frame_id,
                            detections_length: output.detections.length,
                            job_id: (object_detection_job as any).id
                        }, "Sending object detection result:");
                        client.ws.send(encoded);

                        // TODO: Implement proper cleanup flow
                        fs.unlinkSync(file_path);
                    }
                });
            }
            return;
        }

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

        console.error("Received message from unauthenticated and non-worker client.", decoded);
        ws.close(1008, "Unauthenticated");
    }
}