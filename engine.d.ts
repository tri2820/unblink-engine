export type ServerRegistrationMessage = {
    type: "i_am_server";
    token?: string;
}

export type ServerToEngine =
    | {
        type: "frame_binary";
        workers: Partial<{
            'vlm': true,
            'object_detection': true,
            'embedding': true,
        }>
        frame_id: string;
        stream_id: string;
        frame: Uint8Array;
    }

export type DetectionObject = {
    label: string;
    confidence: number;
    box: {
        x_min: number;
        y_min: number;
        x_max: number;
        y_max: number;
    }
}

export type EngineToServer = {
    type: "frame_description";
    frame_id: string;
    stream_id: string;
    description: string;
} | {
    type: "frame_embedding";
    frame_id: string;
    stream_id: string;
    embedding: number[];
} | {
    type: "frame_object_detection";
    stream_id: string;
    frame_id: string;
    objects: DetectionObject[];
} | {
    type: "media_summary",
    media_id: string;
    summary: Summary
} | FrameMotionEnergyMessage

export type FrameMotionEnergyMessage = {
    type: "frame_motion_energy";
    stream_id: string;
    frame_id: string;
    motion_energy: number;
}

export type Moment = {
    from_time: number;
    to_time: number;
    what_old: string,
    what_new: string,
    importance_score: number,
    labels: string[],
}
export type Summary = {
    background: string;
    moments: Moment[],
}