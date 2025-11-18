import type { ServerRegistrationMessage, ServerToEngine } from "./engine";
export type WorkerRegistrationMessage = {
    type: "i_am_worker";
    worker_secret: string;
    worker_config: {
        worker_type: string;
        max_batch_size?: number;
        max_latency_ms?: number;
    };
}
export type WorkerToEngine = | {
    type: 'worker_output';
    output: {
        id: string;
        description: string;
    }[]

}

export type RegistrationMessage = ServerRegistrationMessage | WorkerRegistrationMessage;
export type EngineReceivedMessage = ServerToEngine | WorkerToEngine | RegistrationMessage;