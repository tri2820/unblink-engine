
import type { ServerWebSocket } from 'bun';
import { decode, encode } from 'cbor-x';
import fs from 'fs';
import type { EngineReceivedMessage, ServerToEngine } from './shared';

export type Client = {
    id: string;
    ws: ServerWebSocket<unknown>;
    is_server?: boolean;
    worker_config?: {
        worker_type: string;
        max_batch_size: number;
        max_latency_ms: number;
        gathered: any[];
        send_timeout?: NodeJS.Timeout;
    },
}


// async function loopStream(stream_id: string, url: string) {
//     const messages = forwardStream(url);

//     try {
//         for await (const msg of messages) {
//             const now = Date.now();
//             if (!stream_state[stream_id]) stream_state[stream_id] = {
//                 agents: {
//                     'default': {
//                         name: 'Default Agent',
//                         role_description: `Describe this drone footage in detailed. Also describe the telemetry data shown in the HUD.`,
//                         updates: []
//                     }
//                 }
//             };
//             // Limit to 30 fps per stream
//             if (now - (stream_state[stream_id]?.last_sent ?? 0) < 1000 / 30) {
//                 continue;
//             }

//             stream_state[stream_id].last_sent = now;
//             // Forward all messages to backend for indexing
//             // This id is used to identify frames
//             const id = crypto.randomUUID();

//             if (msg.type === "frame") {
//                 for (const client of clients.values()) {
//                     if (client.ws.readyState !== WebSocket.OPEN) continue;
//                     if (client.worker_config) continue; // Don't send frames to workers
//                     client.ws.send(createMessage({ type: "frame", stream_id, id }, msg.buffer));
//                 }


//                 // Save every 5 seconds
//                 if (now - (stream_state[stream_id]?.last_save_file ?? 0) > 5000) {
//                     // Also send to workers for processing

//                     // Pick random agent
//                     const agent_id = Object.keys(stream_state[stream_id].agents)[
//                         Math.floor(Math.random() * Object.keys(stream_state[stream_id].agents).length)
//                     ];
//                     const agent = stream_state[stream_id].agents[agent_id];
//                     if (!agent) return;

//                     const file_path = `/tmp/frame-${id}.jpg`;
//                     fs.writeFileSync(file_path, msg.buffer as any);
//                     stream_state[stream_id].last_save_file = now;

//                     console.log("Saved frame to", file_path, "for stream", stream_id, "agent", agent_id);
//                     // Add to database
//                     addMediaUnit({
//                         media_id: stream_id,
//                         at_time: new Date(),
//                         description: null,
//                         embedding: null,
//                         id,
//                     });

//                     const image_description_job = {
//                         messages: [
//                             {
//                                 role: 'system',
//                                 content: [
//                                     { type: 'text', text: `${agent.role_description}. Do NOT say anything about the image being blurry. Try to describe what looks like inside.` },
//                                 ]
//                             },
//                             {
//                                 "role": "user",
//                                 "content": [
//                                     { "type": "image", "image": file_path },
//                                 ]
//                             }
//                         ]
//                     };





//                     sendJob(image_description_job, 'vlm', {
//                         async cont(output) {
//                             console.log("Update database with description");
//                             updateMediaUnit({
//                                 id,
//                                 description: output.description,
//                             });

//                             clients.forEach(c => {
//                                 if (c.ws.readyState !== WebSocket.OPEN) return;
//                                 if (!c.is_browser) return; // Don't send to workers
//                                 console.log("Sending update to browser", output.description.substring(0, 20), '...');
//                                 c.ws.send(createMessage({ type: "update", stream_id, id, description: output.description, agent }));
//                             });

//                             stream_state[stream_id].agents[agent_id].updates = [
//                                 ...stream_state[stream_id].agents[agent_id].updates,
//                                 {
//                                     at_time: Date.now(),
//                                     description: output.description,
//                                 }
//                             ].slice(-20); // Keep last 20 updates
//                         }
//                     });



//                     const embedding_job = { filepath: file_path };
//                     sendJob(embedding_job, 'embedding', {
//                         async cont(output) {

//                             console.log("Update database with embedding");
//                             updateMediaUnit({
//                                 id,
//                                 embedding: output.embedding,
//                             });
//                         }
//                     });


//                 }


//             }

//             // Forward codecpar messages to clients
//             if (msg.type === "codecpar") {
//                 stream_state[stream_id].codecpar = msg.data;
//                 updateStreamState()
//             }

//         }
//     } catch (e) {
//         //   log("Error in stream loop: " + e);
//     }
// }

// // Start streaming all cameras
// Object.entries(streams).forEach(([id, stream]) => {
//     loopStream(id, (stream as any).uri);
// });


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
    c.ws.send(msg);
}


const port = 5000;
Bun.serve({
    port,
    async fetch(req, server) {
        const url = new URL(req.url);
        console.log('HTTP request', req.method, req.url, url.pathname);

        // Dedicated endpoint for WebSocket upgrades
        if (url.pathname === "/ws") {
            console.log('Upgrading to WebSocket');
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
            let parsed: EngineReceivedMessage
            try {
                parsed = decode(message as any);
            } catch (error) {
                console.error("Failed to decode message:", error, message);
                return;
            }
            const client = clients.get(ws);
            if (!client) return;

            // === Worker registration and job distribution ===
            if (parsed.type === "i_am_worker") {
                if (!parsed.worker_config || !parsed.worker_config.worker_type) {
                    console.error("Invalid worker_config from worker.", parsed);
                    return;
                }

                if (parsed.secret !== process.env.WORKER_SECRET) {
                    console.error("Invalid WORKER_SECRET from worker.");
                    ws.close(1008, "Invalid WORKER_SECRET");
                    return;
                }

                console.log('Registered worker', client.id, parsed.worker_config);
                client.worker_config = {
                    max_batch_size: 32,
                    max_latency_ms: 30000,
                    worker_type: parsed.worker_config.worker_type,
                    gathered: [],
                };

                client.worker_config = { ...client.worker_config, ...parsed.worker_config };
                return;
            }

            if (parsed.type === "i_am_server") {
                console.log('Registered server client', client.id);
                client.is_server = true;

                // TODO: validate token if provided


                return;
            }

            if (parsed.type === "frame_binary") {

                console.log("Received frame from server client", client.id, "size", parsed.frame.byteLength);
                // Create job for processing frame
                // const file_path = `/tmp/frame-${client.id}-${Date.now()}.jpg`;
                // fs.writeFileSync(file_path, parsed.frame);

                // const image_description_job = {
                //     messages: [
                //         {
                //             role: 'system',
                //             content: [
                //                 { type: 'text', text: `Describe this image in detailed. Do NOT say anything about the image being blurry. Try to describe what looks like inside.` },
                //             ]
                //         },
                //         {
                //             "role": "user",
                //             "content": [
                //                 { "type": "image", "image": file_path },
                //             ]
                //         }
                //     ]
                // };

                // sendJob(image_description_job, 'vlm', {
                //     async cont(output) {
                //         console.log("Received description for frame", output);
                //         // clients.forEach(c => {
                //         //     if (c.ws.readyState !== WebSocket.OPEN) return;
                //         //     if (!c.is_browser) return; // Don't send to workers
                //         //     console.log("Sending update to browser", output.description.substring(0, 20), '...');
                //         //     c.ws.send(createMessage({ type: "update", stream_id, id, description: output.description, agent }));
                //         // });
                //     }
                // });
                return;
            }

            if (parsed.type === "worker_output") {
                if (!client.worker_config) {
                    console.error("Received worker_output from non-worker client.", parsed);
                    ws.close(1008, "Not a worker");
                    return;
                }

                const outputs = parsed.output as any[];
                // Sanity check
                if (!outputs || !Array.isArray(outputs)) return;
                for (const output of outputs) {
                    const job = job_map.get(output.id);
                    job?.cont(output);
                }
                return;
            }

            console.error("Received message from unauthenticated and non-worker client.", parsed);
            ws.close(1008, "Unauthenticated");
        },
        close(ws) {
            clients.delete(ws);
        }
    }, // handlers
});

console.log(`Server running on http://localhost:${port}`);
