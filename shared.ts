export type ServerToEngine = {
    type: "frame_binary";
    frame_id: string;
    stream_id: string;
    frame: Uint8Array;
} | {
    type: "i_am_server";
    token?: string;
}
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