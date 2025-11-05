
import type { ServerWebSocket } from 'bun';
import { decode, encode } from 'cbor-x';
import fs from 'fs';
import type { EngineReceivedMessage, EngineToServer, ServerToEngine } from './shared';
import { ensureDirExists, FRAMES_DIR } from './appdir';
import path from 'path';
import { logger } from './logger';

const FRAME_SIZE_LIMIT = 2 * 1024 * 1024; // 2 MB
export type Client = {
    id: string;
    ws: ServerWebSocket<unknown>;
    server_config?: {
        tenant_id: string;
    }
    worker_config?: {
        worker_type: string;
        max_batch_size: number;
        max_latency_ms: number;
        gathered: any[];
        send_timeout?: NodeJS.Timeout;
    },
}


export type JobMap = Map<string, { cont: (result: Record<string, any>) => void }>;
const job_map = new Map() as JobMap;
const clients = new Map<ServerWebSocket<unknown>, Client>();


export function sendJob(job: Record<string, any>, worker_type: string, opts?: {
    cont: (result: Record<string, any>) => void;
}) {
    job.id = crypto.randomUUID();
    if (opts?.cont) {
        job_map.set(job.id, {
            cont: opts.cont
        });
    }

    // TODO: distribute work for all workers of that worker_type, not just the first one
    const worker = clients.values().find(c => c.worker_config?.worker_type === worker_type);
    if (!worker) return;

    worker.worker_config!.gathered.push(job);
    if (worker.worker_config!.gathered.length > 1000) {
        console.warn("Worker gathered queue too large, dropping jobs.");
        worker.worker_config!.gathered = worker.worker_config!.gathered.slice(-1000);
    }

    if (worker.worker_config!.send_timeout) clearTimeout(worker.worker_config!.send_timeout);

    if (worker.worker_config!.gathered.length >= (worker.worker_config!.max_batch_size)) {
        workerFlush(worker);
        return;
    }

    worker.worker_config!.send_timeout = setTimeout(async () => {
        workerFlush(worker);
    }, worker.worker_config!.max_latency_ms);
}

export function workerFlush(c: Client) {
    // // This function might be called from timeout, so check everything
    if (!c.ws || c.ws.readyState !== WebSocket.OPEN) return;
    if (!c.worker_config) return;
    const inputs = structuredClone(c.worker_config.gathered);
    c.worker_config.gathered = []
    const msg = encode({
        inputs
    })
    logger.info({
        event: 'worker_flush',
        c_id: c.id,
        length: inputs.length,
    }, "Sending job batch to worker");
    c.ws.send(msg);
}


const port = 5000;
Bun.serve({
    port,
    async fetch(req, server) {
        const url = new URL(req.url);

        // Dedicated endpoint for WebSocket upgrades
        if (url.pathname === "/ws") {
            logger.info('Upgrading to WebSocket');
            const upgraded = server.upgrade(req);
            if (upgraded) {
                // Bun automatically handles the response for successful upgrades
                return;
            }
            return new Response("WebSocket upgrade failed", { status: 400 });
        }

        return new Response("Not Found", { status: 404 });
    },
    routes: {
        "/api/autocomplete": {
            async POST(req) {
                try {
                    const { text } = await req.json() as { text: string };
                    // Perform autocomplete logic here
                    return new Response(JSON.stringify({ suggestions: [] }), {
                        status: 200,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                } catch (error) {
                    logger.error({ event: 'autocomplete_error', error }, 'Error processing autocomplete request');
                    return new Response(JSON.stringify({ error: 'Invalid request' }), {
                        status: 400,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                }
            }
        },
        "/api/search": {
            async POST(req) {
                try {
                    const { query } = await req.json() as { query: string };
                    logger.info({ event: 'search_request', query }, 'Received search request');
                    return new Response(JSON.stringify({ results: [] }), {
                        status: 200,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                } catch (error) {
                    logger.error({ event: 'search_error', error }, 'Error processing search request');
                    return new Response(JSON.stringify({ error: 'Invalid request' }), {
                        status: 400,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                }
            }
        },
        "/version": {
            GET() {
                return new Response(JSON.stringify({ version: "1.0.0" }), {
                    status: 200,
                    headers: {
                        "Content-Type": "application/json",
                    }
                });
            }
        }
    },

    websocket: {
        open(ws) {
            // This is different from tenant_id
            const id = crypto.randomUUID();
            clients.set(ws, { id, ws, });
        },
        async message(ws, message) {
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
                        console.error("No job found for worker output ID:", output.id);
                        continue;
                    }
                    job.cont(output);
                }
                return;
            }

            console.error("Received message from unauthenticated and non-worker client.", decoded);
            ws.close(1008, "Unauthenticated");
        },
        close(ws) {
            clients.delete(ws);
        }
    }, // handlers
});

logger.info(`Server running on http://localhost:${port}`);
