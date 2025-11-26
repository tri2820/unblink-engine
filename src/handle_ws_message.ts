import { file, type ServerWebSocket } from "bun";
import { decode, encode } from "cbor-x";
import fs from 'fs';
import path from 'path';
import { ENGINE_VERSION, sendJob, type Client, type JobMap } from "..";
import { ensureDirExists, FRAMES_DIR, TEMP_DIR } from "../appdir";
import type { EngineToServer, WorkerRequest, WorkerResponse, Resource } from "../engine";
import { logger } from "../logger";
import type { EngineReceivedMessage, RegistrationMessage, WorkerToEngine } from "../shared";

const FRAME_SIZE_LIMIT = 2 * 1024 * 1024; // 2 MB

const blacklist_servers_ip = new Set<string>();

async function handle_register(props: {
    decoded: RegistrationMessage,
    job_map: JobMap,
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
    job_map: JobMap,
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
            if (job.timeout) clearTimeout(job.timeout);
            job.cont(output);
            job_map.delete(output.id);
        }
        return;
    }
}
async function handle__serverMessage(props: {
    decoded: WorkerRequest,
    job_map: JobMap,
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

    try {
        if (decoded.type === "worker_request") {
            type RefState = {
                num_ref: number,
                uri: string,
                _cleanup: () => void
            }
            const resourceRefState: {
                [resource_id: string]: RefState
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

            // Helper to collect all resource IDs from job.input
            const collectResourceIdsFromInput = (obj: any): string[] => {
                if (!obj) return [];
                if (obj.__type === 'resource-ref') return [obj.id];
                if (Array.isArray(obj)) return obj.flatMap(collectResourceIdsFromInput);
                if (typeof obj === 'object') return Object.values(obj).flatMap(collectResourceIdsFromInput);
                return [];
            };
            
            // Validate 
            if (decoded.resources) {
                // All jobs refer to existing resources
                for (const job of decoded.jobs) {
                    const resourceIds = collectResourceIdsFromInput(job.input);
                    for (const resourceId of resourceIds) {
                        const resources_item = decoded.resources?.find((r) => r.id === resourceId)
                        if (!resources_item) {
                            console.error("Job refers to non-existent resource:", resourceId);
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
                    if (!decoded.jobs.some((job) => {
                        const resourceIds = collectResourceIdsFromInput(job.input);
                        return resourceIds.includes(resource.id);
                    })) {
                        console.error("Resource is not refered to by any job:", resource.id);
                        ws.close(1009, "Resource is not refered to by any job");
                        return;
                    }
                }
            }

            // Cache resources
            if (decoded.resources) {
                for (const resource of decoded.resources) {
                    if (resource.type === 'image') {
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
                    } else if (resource.type === 'document') {
                        const file_path = path.join(FRAMES_DIR_TENANT, `${resource.id}-${nonce}.txt`);
                        fs.writeFileSync(file_path, resource.content, 'utf8');
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
            }

            // Calculate reference counts based on job usage
            // Helper to collect all resource IDs from job.input
            const collectResourceIds = (obj: any): string[] => {
                if (!obj) return [];
                if (obj.__type === 'resource-ref') return [obj.id];
                if (Array.isArray(obj)) return obj.flatMap(collectResourceIds);
                if (typeof obj === 'object') return Object.values(obj).flatMap(collectResourceIds);
                return [];
            };
            
            for (const job of decoded.jobs) {
                const resourceIds = collectResourceIds(job.input);
                for (const resourceId of resourceIds) {
                    // text resources are not cached
                    if (resourceRefState[resourceId]) {
                        resourceRefState[resourceId]!.num_ref++;
                    }
                }
            }

            const getFullResource = (r: { id: string }) => {
                const resource = decoded.resources?.find((r2) => r2.id === r.id)
                return resource
            }

            const getRefState = (r: { id: string }): RefState => {
                const refState = resourceRefState[r.id]
                if (!refState) {
                    throw new Error("Resource not found");
                }
                return refState
            }

            // Helper to resolve ResourceRef objects in job.input
            const resolveResourceRefs = (obj: any): any => {
                if (obj === null || obj === undefined) return obj;
                
                if (obj.__type === 'resource-ref') {
                    const resource = decoded.resources?.find(r => r.id === obj.id);
                    if (!resource) return obj;
                    
                    // Replace ref with URI (all resources are saved to files)
                    return getRefState(resource).uri;
                }
                
                if (Array.isArray(obj)) {
                    return obj.map(resolveResourceRefs);
                }
                
                if (typeof obj === 'object') {
                    const resolved: any = {};
                    for (const key in obj) {
                        resolved[key] = resolveResourceRefs(obj[key]);
                    }
                    return resolved;
                }
                
                return obj;
            };

            for (const job of decoded.jobs) {
                // Resolve ResourceRefs in job.input to actual file paths/content
                const resolvedInput = resolveResourceRefs(job.input);
                
                // Minimal handler: just pass resolved input to worker
                const job_data = {
                    ...resolvedInput,
                };

                sendJob(job_data, job.worker_type, {
                    async cont(output) {
                        const msg: EngineToServer = {
                            type: "worker_response",
                            output,
                            job_id: job.job_id,
                        }

                        const encoded = encode(msg);
                        client.ws.send(encoded);
                        
                        // Cleanup: find and unref all resources used in this job's input
                        const collectResourceIds = (obj: any): string[] => {
                            if (!obj) return [];
                            if (obj.__type === 'resource-ref') return [obj.id];
                            if (Array.isArray(obj)) return obj.flatMap(collectResourceIds);
                            if (typeof obj === 'object') return Object.values(obj).flatMap(collectResourceIds);
                            return [];
                        };
                        
                        const resourceIds = collectResourceIds(job.input);
                        for (const resourceId of resourceIds) {
                            unref(resourceId);
                        }
                    }
                });
            }


            return;
        }
    } catch (e) {
        console.error('Wow', e);
        console.trace();
        ws.close(1008, "Error running job");
    }
}


export function createWsMessageHandler(clients$: () => Map<ServerWebSocket<unknown>, Client>, job_map$: () => JobMap) {
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