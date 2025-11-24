import { file, type ServerWebSocket } from "bun";
import { decode, encode } from "cbor-x";
import fs from 'fs';
import path from 'path';
import { ENGINE_VERSION, sendJob, type Client } from "..";
import { ensureDirExists, FRAMES_DIR, TEMP_DIR } from "../appdir";
import type { EngineToServer, WorkerRequest, WorkerResponse } from "../engine";
import { logger } from "../logger";
import type { EngineReceivedMessage, RegistrationMessage, WorkerToEngine } from "../shared";

const FRAME_SIZE_LIMIT = 2 * 1024 * 1024; // 2 MB

const blacklist_servers_ip = new Set<string>();

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

        if (decoded.version !== ENGINE_VERSION) {
            if (blacklist_servers_ip.has(ws.remoteAddress)) {
                ws.close(1008, "Invalid version");
                return;
            }
            blacklist_servers_ip.add(ws.remoteAddress);
            console.error("Invalid version from server.", decoded);
            ws.close(1008, "Invalid version");
            return;
        }

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
    decoded: WorkerRequest,
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

    if (decoded.type === "worker_request") {
        const resourceRefState: {
            [resource_id: string]: {
                num_ref: number,
                uri: string,
                _cleanup: () => void
            }
        } = {};

        // Cleanup resource after all workers have responded
        const unref = (resource_id: string) => {
            if (!resourceRefState[resource_id]) {
                return;
            }

            resourceRefState[resource_id].num_ref--;
            if (resourceRefState[resource_id].num_ref <= 0) {
                resourceRefState[resource_id]._cleanup();
            }
        };

        // Create job for processing frame
        const FRAMES_DIR_TENANT = FRAMES_DIR(tenant_id);
        await ensureDirExists(FRAMES_DIR_TENANT);
        // This ensures unique filenames even if frames arrive quickly, for proper cleanup
        const nonce = crypto.randomUUID();

        // Validate 
        if (decoded.resources) {
            // All jobs refer to existing resources
            for (const job of decoded.jobs) {
                for (const resource of (job.resources || [])) {
                    const resources_item = decoded.resources?.find((r) => r.id === resource.id)
                    if (!resources_item) {
                        console.error("Job refers to non-existent resource:", resource.id);
                        ws.close(1009, "Job refers to non-existent resource");
                        return;
                    }
                }
            }

            // Size check
            for (const resource of decoded.resources) {
                if (resource.type === 'image') {
                    if (resource.data.byteLength > FRAME_SIZE_LIMIT) {
                        console.error("Frame size exceeds limit of", FRAME_SIZE_LIMIT, "bytes.");
                        ws.close(1009, "Frame size exceeds limit");
                        return;
                    }
                }
            }

            // All in resources are refered to by at least one job
            for (const resource of decoded.resources) {
                if (!decoded.jobs.some((job) => job.resources?.some((r) => r.id === resource.id))) {
                    console.error("Resource is not refered to by any job:", resource.id);
                    ws.close(1009, "Resource is not refered to by any job");
                    return;
                }
            }
        }

        // Cache resources
        if (decoded.resources) {
            for (const resource of decoded.resources) {
                const file_path = path.join(FRAMES_DIR_TENANT, `${resource.id}-${nonce}.jpg`);
                fs.writeFileSync(file_path, resource.data);

                resourceRefState[resource.id] = {
                    num_ref: 0,
                    uri: file_path,
                    _cleanup: () => {
                        fs.unlink(file_path, (err) => {
                            if (err && (err as any).code !== 'ENOENT') {
                                console.error("Failed to delete resource:", file_path, err);
                            }
                        });
                    }
                }
            }
        }

        // Calculate reference counts based on job usage
        for (const job of decoded.jobs) {
            for (const resource of (job.resources || [])) {
                resourceRefState[resource.id]!.num_ref++;
            }
        }


        for (const job of decoded.jobs) {
            // Some translation needed here 
            // TODO: make all workers accept resources
            let job_data: undefined | Record<string, any> = undefined;
            if (job.worker_type === 'caption') {
                job_data = {
                    "images": job.resources?.map((resource) => resourceRefState[resource.id]!.uri)
                }
            }

            if (job.worker_type === 'embedding') {
                job_data = {
                    "filepath": job.resources?.map((resource) => resourceRefState[resource.id]!.uri)[0]
                }
            }

            if (job.worker_type === 'object_detection') {
                job_data = {
                    "filepath": job.resources?.map((resource) => resourceRefState[resource.id]!.uri)[0]
                }
            }

            if (job.worker_type === 'motion_energy') {
                job_data = {
                    current_frame: job.resources?.map((resource) => resourceRefState[resource.id]!.uri)[0],
                    media_id: job.cross_job_id
                }
            }

            if (!job_data) {
                console.error("Invalid job type:", job.worker_type);
                ws.close(1009, "Invalid job type");
                return;
            }

            job_data.cross_job_id = job.cross_job_id;

            sendJob(job_data, job.worker_type, {
                async cont(output) {
                    const msg: EngineToServer = {
                        type: "worker_response",
                        output,
                        job_id: job.job_id,
                    }

                    const encoded = encode(msg);
                    client.ws.send(encoded);
                    for (const resource of job.resources || []) {
                        unref(resource.id);
                    }
                }
            });
        }


        // if (decoded.worker_id === 'caption') {
        //     const tempDir = TEMP_DIR();

        //     const image_paths: string[] = [];
        //     const nonce = crypto.randomUUID();

        //     for (const resource of decoded.resources) {
        //         if (resource.type === 'image') {
        //             const file_path = path.join(tempDir, `${resource.type}-${nonce}.jpg`);
        //             fs.writeFileSync(file_path, resource.data);
        //             image_paths.push(file_path);
        //         }
        //     }

        //     if (image_paths.length === 0) {
        //         console.error("No valid frames found in moment_enrichment");
        //         return;
        //     }

        //     const job = {
        //         "images": image_paths,
        //         "query": decoded.query
        //     };

        //     sendJob(job, 'caption', {
        //         async cont(output) {
        //             // const response_text = output.response || "";
        //             const msg: WorkerResponse = {
        //                 type: "worker_response",
        //                 output,
        //                 worker_id: 'caption',
        //                 identifier: {
        //                     media_id: decoded.identifier.media_id,
        //                     moment_id: decoded.identifier.moment_id,
        //                 }
        //             }
        //             const encoded = encode(msg);
        //             client.ws.send(encoded);
        //             // let enrichment;

        //             // try {
        //             //     // Attempt to parse JSON from the response
        //             //     // Sometimes models wrap JSON in markdown code blocks, so strip them if present
        //             //     const cleaned = response_text.replace(/```json/g, '').replace(/```/g, '').trim();
        //             //     enrichment = JSON.parse(cleaned);
        //             // } catch (e) {
        //             //     console.error("Failed to parse JSON from worker response:", response_text);
        //             //     // Fallback if parsing fails
        //             //     enrichment = {
        //             //         title: "Activity Detected",
        //             //         short_description: response_text.length > 100 ? response_text.substring(0, 97) + "..." : response_text,
        //             //         long_description: response_text
        //             //     };
        //             // }

        //             // // Ensure required keys exist
        //             // if (!enrichment.title) enrichment.title = "Activity Detected";
        //             // if (!enrichment.short_description) enrichment.short_description = enrichment.long_description ? (enrichment.long_description.substring(0, 97) + "...") : "No description";
        //             // if (!enrichment.long_description) enrichment.long_description = enrichment.short_description;

        //             // const msg: EngineToServer = {
        //             //     type: "moment_enrichment",
        //             //     media_id: decoded.identifier.media_id,
        //             //     moment_id: decoded.identifier.moment_id,
        //             //     enrichment
        //             // };

        //             // const encoded = encode(msg);
        //             // client.ws.send(encoded);

        //             // Cleanup files
        //             for (const p of image_paths) {
        //                 fs.unlink(p, (err) => {
        //                     if (err) console.error("Failed to delete moment frame:", p, err);
        //                 });
        //             }
        //         }
        //     });
        // }
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