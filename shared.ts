import type { ServerToEngine } from "./engine";

export type WorkerToEngine = {
    type: "i_am_worker";
    secret: string;
    worker_config: {
        worker_type: string;
        max_batch_size?: number;
        max_latency_ms?: number;
    };
} | {
    type: 'worker_output';
    output: {
        id: string;
        description: string;
    }[]

}

export type EngineReceivedMessage = ServerToEngine | WorkerToEngine;