
import type { ServerWebSocket } from 'bun';
import { encode } from 'cbor-x';
import { logger } from './logger';
import { createWsMessageHandler } from './src/handle_ws_message';
import { parseJsonFromString } from './src/utils';

export type SummaryBuilder = {
    media_units: {
        id: string;
        description: string,
        at_time: number,
    }[],
}
export type Client = {
    id: string;
    ws: ServerWebSocket<unknown>;
    server_config?: {
        tenant_id: string;
        state: {
            [media_id: string]: {
                summary_builder: SummaryBuilder
            }
        }
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
const ws_handler = createWsMessageHandler(() => clients, () => job_map);


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
        worker_type: c.worker_config.worker_type,
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

                    const messages = [
                        {
                            "role": "system",
                            "content":
                                `You are an expert security and threat assessment assistant. Your task is to refine a user's keyword into 5 specific, actionable search queries with an investigative mindset. Respond ONLY with a single, valid JSON array of 5 strings. Do not include explanations or markdown.`

                        },

                        {
                            "role": "user",
                            "content": "Refine the following keyword: \"person\""
                        },
                        {
                            "role": "assistant",
                            "content": `["person carrying suspicious package", "person loitering in restricted area", "person acting erratically", "person looking into vehicles", "person wearing a disguise"]`
                        },

                        {
                            "role": "user",
                            "content": "Refine the following keyword: \"car\""
                        },
                        {
                            "role": "assistant",
                            "content": `["car parked in no-parking zone", "car circling the block", "car with obscured license plate", "unattended vehicle near entrance", "driver slumped over steering wheel"]`
                        },

                        {
                            "role": "user",
                            "content": `Refine the following keyword: "${text}"`
                        }
                    ]

                    const output = await new Promise<Record<string, any>>((resolve) => {
                        sendJob({
                            messages,
                        }, 'llm_fast', {
                            cont(output) {
                                logger.info({ event: 'autocomplete', text }, 'Processed autocomplete request');
                                resolve(output);
                            }
                        });
                    })

                    console.log('autocomplete output', output);
                    const parsed = parseJsonFromString(output.response);
                    const array_of_strings = parsed.data;
                    if (!array_of_strings || !Array.isArray(array_of_strings) || array_of_strings.length === 0 || !array_of_strings.every((item) => typeof item === 'string')) {
                        throw new Error("Failed to parse autocomplete response as array of strings");
                    }

                    // Return the array of strings as autocomplete items
                    return new Response(JSON.stringify({
                        items: array_of_strings.map(str => ({ text: str }))
                    }), {
                        status: 200,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                } catch (error) {
                    logger.error({ event: 'autocomplete_error', error }, 'Error processing autocomplete request');
                    return new Response(JSON.stringify({
                        // Empty items on error
                        items: [],
                        error: 'Failed to process autocomplete request'
                    }), {
                        status: 400,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                }
            }
        },
        "/api/worker/fast_embedding": {
            async POST(req) {
                const { job } = await req.json() as { job: { text: string, prompt_name: string } };
                if (job.prompt_name !== 'query') {
                    return new Response(JSON.stringify({ error: 'Invalid prompt_name' }), {
                        status: 400,
                        headers: {
                            "Content-Type": "application/json",
                        }
                    });
                }

                const result = await new Promise<{ embedding: number[] }>((resolve) => {
                    sendJob(job, 'fast_embedding', {
                        cont(output) {
                            resolve({ embedding: output.embedding });
                        }
                    });
                });

                logger.info({ event: 'fast_embedding', job }, 'Processed fast embedding request');

                return new Response(JSON.stringify(result), {
                    status: 200,
                    headers: {
                        "Content-Type": "application/json",
                    }
                });
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
            try {
                await ws_handler(ws, message);
            } catch (error) {
                logger.error({ event: 'ws_message_error', error_msg: (error as Error).message }, 'Error handling WebSocket message');
            }
        },
        close(ws) {
            clients.delete(ws);
        }
    }, // handlers
});

logger.info(`Server running on http://localhost:${port}`);
