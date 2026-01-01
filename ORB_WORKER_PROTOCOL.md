# ORB Worker Protocol

## Overview

The ORB Worker Protocol defines the communication layer between the unblink-engine server and distributed Python worker processes. It enables job distribution, batch processing, resource sharing, and result delivery over WebSocket with CBOR-X serialization.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ORB Engine Server (Bun)                     │
│  - Job distribution & queuing                               │
│  - Resource lifecycle management                            │
│  - Job callback registry with timeouts                      │
└────────────┬──────────────────────────────────────────────┬─┘
             │ WebSocket + CBOR-X                           │
             │ (Binary serialization)                        │
    ┌────────▼────────┐                         ┌────────▼─────────┐
    │  Worker: LLM    │                         │ Worker: Embedding │
    │  Process Pool   │                         │ Process Pool      │
    └─────────────────┘                         └───────────────────┘
             │                                           │
    ┌────────▼────────┐                         ┌────────▼─────────┐
    │  Worker: VLM    │                         │  Worker: Segm.   │
    │  Process Pool   │                         │  Process Pool    │
    └─────────────────┘                         └───────────────────┘
```

**Technology Stack:**
- Transport: WebSocket
- Serialization: CBOR-X (binary encoded)
- Worker Environment: Python 3.8+
- Concurrency: ThreadPoolExecutor for blocking operations

---

## Message Types

### 1. Worker Registration

**Direction:** Worker → Engine
**Type:** `i_am_worker`

```typescript
{
    type: "i_am_worker",
    worker_secret: string,
    worker_config: {
        worker_type: string,        // Type identifier: "embedding", "llm", "vlm", etc.
        max_batch_size?: number,    // Max jobs per batch (default: 32)
        max_latency_ms?: number     // Max wait time before partial batch send (default: 30000)
    }
}
```

**Validation:**
- `worker_secret` must match `WORKER_SECRET` environment variable on engine
- `worker_type` must be recognized by engine or connection closes with code 1008
- `max_batch_size` > 0 (default 32)
- `max_latency_ms` > 0 (default 30000)

**Response:** Connection remains open on success, closes with code 1008 on failure.

**Python Example:**
```python
import asyncio
import json
from cbor2 import dumps, loads
import websockets

async def register_worker():
    uri = "ws://localhost:5000/ws"
    async with websockets.connect(uri) as websocket:
        msg = {
            "type": "i_am_worker",
            "worker_secret": "my-secret-key",
            "worker_config": {
                "worker_type": "embedding",
                "max_batch_size": 32,
                "max_latency_ms": 5000
            }
        }
        await websocket.send(dumps(msg))
        # Connection established, ready to receive jobs
```

---

### 2. Job Batch Distribution

**Direction:** Engine → Worker
**Format:** CBOR-encoded array of job objects

```typescript
{
    inputs: Array<{
        id: string,                     // Unique job ID
        job_id?: string,                // Alternate job ID field
        input: Record<string, any>,     // Job-specific input
        [key: string]: any              // Additional fields passed through
    }>
}
```

**Job Accumulation Rules:**
- Engine accumulates jobs in a queue per worker
- Batch is sent when:
  - `gathered.length >= max_batch_size`, OR
  - `max_latency_ms` expires since first job in queue
- Maximum queue size: 1000 jobs
- Job timeout (max wait before dropped): 5 minutes (300,000ms)

**Worker Responsibilities:**
1. Parse CBOR-encoded message
2. Extract all job objects from `inputs` array
3. Process each job (order-independent)
4. Return results with matching IDs

**TypeScript Example (Engine sends):**
```typescript
const batch = {
    inputs: [
        {
            id: "job-uuid-1",
            input: { text: "hello world", prompt_name: "query" }
        },
        {
            id: "job-uuid-2",
            input: { text: "goodbye world", prompt_name: "passage" }
        }
    ]
};
const encoded = encode(batch);
worker_ws.send(encoded);
```

**Python Example (Worker receives):**
```python
async def process_batch(websocket):
    async for message in websocket:
        try:
            batch_data = loads(message)  # CBOR decode
            inputs = batch_data.get("inputs", [])

            outputs = []
            for job in inputs:
                job_id = job.get("id") or job.get("job_id")
                job_input = job.get("input", {})

                # Process job
                result = await process_job(job_input)
                outputs.append({
                    "id": job_id,
                    **result
                })

            # Send results back
            response = {
                "type": "worker_output",
                "output": outputs
            }
            await websocket.send(dumps(response))
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
```

---

### 3. Worker Output

**Direction:** Worker → Engine
**Type:** `worker_output`

```typescript
{
    type: "worker_output",
    output: Array<{
        id: string,                     // Must match job ID from input
        [result_key: string]: any       // Worker-specific result fields
    }>
}
```

**Requirements:**
- Must include `type: "worker_output"`
- `output` array contains result objects
- Each result must have `id` field matching input job ID
- Result fields are worker-type-specific
- Order of results can differ from input order
- If error occurs, include `error` field instead of results

**Output Delivery:**
1. Engine matches result ID to registered job callback
2. Clears job timeout
3. Invokes callback with result object
4. Deletes job from tracking map

**Error Handling:**
```typescript
// Worker-side error in result
{
    type: "worker_output",
    output: [
        {
            id: "job-uuid-1",
            error: "Model failed to process input: invalid format"
        }
    ]
}
```

**Python Example:**
```python
# Successful results
response = {
    "type": "worker_output",
    "output": [
        {"id": "job-1", "embedding": [0.1, 0.2, ...], "shape": [768]},
        {"id": "job-2", "embedding": [0.3, 0.4, ...], "shape": [768]}
    ]
}
await websocket.send(dumps(response))

# With errors
response = {
    "type": "worker_output",
    "output": [
        {"id": "job-1", "embedding": [0.1, 0.2, ...], "shape": [768]},
        {"id": "job-2", "error": "Image too large"}
    ]
}
await websocket.send(dumps(response))
```

---

## Resource System

### Resource Reference Model

Resources (images, documents, media) are deduplicated and shared across jobs. Instead of sending raw data multiple times, jobs reference resources by ID.

**Resource Definition:**
```typescript
type Resource = ({
    type: 'image',
    data: Uint8Array,           // Raw image bytes
} | {
    type: 'document',
    data: string
}) & {
    id: string,                 // Unique resource ID
}
```

**Resource Reference:**
```typescript
type ResourceRef = {
    __type: 'resource-ref',
    id: string                  // References Resource.id
}
```

### Resource Lifecycle

**Resource Submission:**
- Client sends `WorkerRequest` with both `resources` array and `jobs` array
- Resources submitted once with job batch
- All job inputs reference resources via `ResourceRef`

**Resource Example:**
```typescript
// Client sends:
{
    type: "worker_request",
    resources: [
        {
            id: "img-001",
            type: "image",
            data: <Uint8Array of PNG bytes>
        }
    ],
    jobs: [
        {
            job_id: "job-1",
            worker_type: "embedding",
            input: {
                filepath: {
                    __type: "resource-ref",
                    id: "img-001"
                }
            }
        }
    ]
}
```

**Storage:**
- Resources cached to disk in tenant-specific directory:
  ```
  ~/.local/share/unblink-engine/{tenant_id}/files/frames/{resource_id}-{nonce}.{ext}
  ```
- File extension inferred from resource type or detected from content
- Stored for duration of job processing + cleanup period

**Resource Validation:**
- Maximum resource size: 2MB per resource
- All jobs must reference at least one existing resource (or none)
- All referenced resources must be present in resource array
- Invalid/missing resources close connection with code 1009

**Reference Counting & Cleanup:**
- Engine tracks which jobs reference which resources
- When job completes, engine decrements reference count
- Resource deleted automatically when all jobs complete
- Prevents orphaned files on disk

### Worker Resource Access

Workers receive job inputs with `ResourceRef` objects already resolved to file paths.

**Engine Processing (before sending to worker):**
```typescript
// Input with ResourceRef
{
    id: "job-1",
    input: {
        filepath: { __type: "resource-ref", id: "img-001" }
    }
}

// Engine resolves to actual path:
{
    id: "job-1",
    input: {
        filepath: "/home/user/.local/share/unblink-engine/tenant-1/files/frames/img-001-abc123.png"
    }
}
```

**Python Worker Example:**
```python
from PIL import Image
import numpy as np

def process_embedding_job(job_input):
    filepath = job_input.get("filepath")

    # Load image from resolved path
    image = Image.open(filepath)
    image_array = np.array(image)

    # Process image
    embedding = model.encode(image_array)

    return {"embedding": embedding.tolist()}
```

---

## Message Shapes by Worker Type

### Embedding Worker

**Input:**
```typescript
{
    id: string,
    input: {
        text?: string,
        prompt_name?: 'query' | 'passage',
        filepath?: string   // Resolved resource path
    }
}
```

**Output:**
```typescript
{
    id: string,
    embedding?: number[],           // Float vector, ~768 dims
    shape?: number[],               // [dimensions]
    error?: string                  // If processing failed
}
```

**Batching:** All embeddings in batch can be processed in parallel.

---

### LLM Worker

**Input:**
```typescript
{
    id: string,
    input: {
        messages: Array<{
            role: string,           // "user", "assistant", "system"
            content: string
        }>
    }
}
```

**Output:**
```typescript
{
    id: string,
    response?: string,              // Generated text
    error?: string                  // If inference failed
}
```

**Batching:** Can process multiple inference requests in parallel.

---

### Vision Language Model (VLM) Worker

**Input:**
```typescript
{
    id: string,
    input: {
        messages: Array<{
            role: string,
            content: Array<
                | { type: 'text', text: string }
                | { type: 'image', image: string }  // Resolved resource path
            >
        }>
    }
}
```

**Output:**
```typescript
{
    id: string,
    response?: string,              // Model's text response
    error?: string                  // If processing failed
}
```

**Batching:** Limited by model's context window; batch size typically small.

---

### Image Captioning Worker

**Input:**
```typescript
{
    id: string,
    input: {
        images: string[],           // Array of resolved resource paths
        query?: string              // Optional query/prompt
    }
}
```

**Output:**
```typescript
{
    id: string,
    response?: string,              // Caption text
    error?: string                  // If caption generation failed
}
```

---

### Object Detection Worker

**Input:**
```typescript
{
    id: string,
    input: {
        filepath: string            // Resolved resource path
    }
}
```

**Output:**
```typescript
{
    id: string,
    detections?: Array<{
        label: string,              // Object class name
        score: number,              // Confidence [0, 1]
        box: {
            x_min: number,
            y_min: number,
            x_max: number,
            y_max: number
        }
    }>,
    error?: string
}
```

---

### Image Segmentation Worker

**Input:**
```typescript
{
    id: string,
    input: {
        filepath: string,           // Resolved resource path
        prompts?: string[]          // Optional region prompts
    }
}
```

**Output:**
```typescript
{
    id: string,
    masks?: Array<{
        size: [number, number],     // [height, width]
        counts: string              // RLE-encoded mask
    }>,
    labels?: string[],              // Class labels
    boxes?: number[][],             // [[x_min, y_min, x_max, y_max], ...]
    error?: string
}
```

---

### Video Segmentation Worker (SAM3)

**Input:**
```typescript
{
    id: string,
    input: {
        cross_job_id: string,           // Session identifier for video tracking
        current_frame: string,          // Resolved resource path to frame image
        prompts?: string[],             // Optional segmentation prompts
        reset_session?: boolean         // Clear session state if true
    }
}
```

**Output (Success):**
```typescript
{
    id: string,
    cross_job_id: string,               // Echo back session ID
    frame_count: number,                // Frame index in sequence
    objects: number[],                  // Object IDs tracked across frames
    scores: number[],                   // Confidence per object
    labels: string[],                   // Class labels
    boxes: number[][],                  // Bounding boxes per object
    masks: Array<{
        size: [number, number],
        counts: string                  // RLE-encoded
    }>
}
```

**Output (Error):**
```typescript
{
    id: string,
    cross_job_id: string,
    error: string
}
```

**Session Management:**
- `cross_job_id` maps to stateful segmentation session
- Engine maintains per-session state in worker
- Multiple jobs with same `cross_job_id` share temporal context
- `reset_session: true` clears accumulated history
- Frame count increments automatically per job

---

### Motion Energy Detection Worker

**Input:**
```typescript
{
    id: string,
    input: {
        media_id: string,               // Identifier for media stream
        current_frame: string           // Resolved resource path to current frame
    }
}
```

**Output:**
```typescript
{
    id: string,
    motion_energy?: number,             // Magnitude of motion detected [0, 1]
    error?: string
}
```

**State Management:**
- Worker caches last frame per `media_id`
- Compares current frame to previous frame
- Returns motion energy metric
- Uses LRU cache with max 100 concurrent streams
- Auto-evicts oldest streams when limit exceeded

---

## Job Lifecycle

### Full Request/Response Cycle

```
1. Client sends WorkerRequest (with resources + jobs)
   ↓
2. Engine validates:
   - All resources fit size limits
   - All job references exist
   - All resources referenced by ≥1 job
   ↓
3. Engine stores resources to disk
   ↓
4. Engine queues jobs to appropriate workers
   ↓
5. Engine waits for batching conditions:
   - batch_size ≥ max_batch_size, OR
   - max_latency_ms expired
   ↓
6. Engine sends batched jobs to worker (CBOR)
   ↓
7. Worker processes batch
   ↓
8. Worker sends results (CBOR)
   ↓
9. Engine matches results to job callbacks
   ↓
10. Engine invokes client callback with result
    ↓
11. Engine decrements resource reference counts
    ↓
12. Engine deletes resources when counts = 0
    ↓
13. Job timeout cleared, job removed from map
```

### Timeout Behavior

**Job Timeout:** 5 minutes (300,000 ms)
- If job not completed within 5 minutes, it's abandoned
- Callback never invoked
- Job entry deleted from map
- Resource reference count still decremented

**Batch Send Timeout:** `max_latency_ms`
- If first job in queue waiting longer than this, partial batch sent
- Even if batch size < `max_batch_size`
- Timer resets after each batch send

---

## Connection Management

### Connection Lifecycle

**Worker Connection:**
```
1. Worker initiates WebSocket connection to engine
2. Worker sends "i_am_worker" registration message
3. Engine validates worker_secret
4. Connection established - worker ready to receive jobs
5. Worker processes incoming job batches until connection closes
6. On disconnect: all queued jobs reassigned to other workers
```

**Connection Close Codes:**
- `1000`: Normal closure
- `1008`: Policy violation (invalid secret, unknown type)
- `1009`: Message too big (oversized resource)

### Multi-Worker Setup

Multiple instances of same worker type can run in parallel:
```bash
# Launch 3 embedding workers in parallel
MAX_BATCH_SIZE=32 WORKER_TYPE=embedding WORKER_SECRET=secret worker_embedding.py &
MAX_BATCH_SIZE=32 WORKER_TYPE=embedding WORKER_SECRET=secret worker_embedding.py &
MAX_BATCH_SIZE=32 WORKER_TYPE=embedding WORKER_SECRET=secret worker_embedding.py &
```

Engine distributes jobs across all connected workers of same type using round-robin or first-available.

---

## Configuration

### Environment Variables (Worker)

```bash
# Required
WORKER_TYPE=<string>              # Worker type identifier
WORKER_SECRET=<string>            # Must match engine's WORKER_SECRET env var

# Optional
MAX_BATCH_SIZE=<int>              # Default: 32
MAX_LATENCY_MS=<int>              # Default: 30000

# Model-specific (optional)
MODEL_ID=<huggingface_id>
HF_TOKEN=<token>
CUDA_VISIBLE_DEVICES=<int>
```

### Environment Variables (Engine)

```bash
# WebSocket server
SERVER_HOST=localhost             # Default: localhost
SERVER_PORT=5000                  # Default: 5000

# Authentication
WORKER_SECRET=<string>            # Workers must match this

# Resource storage
XDG_DATA_HOME=~/.local/share      # Resource cache location

# Logging
LOG_LEVEL=info                    # info, debug, warn, error
```

---

## Error Handling & Resilience

### Worker-Side Error Handling

```python
async def process_batch(websocket):
    try:
        async for message in websocket:
            batch_data = loads(message)
            outputs = []

            for job in batch_data.get("inputs", []):
                job_id = job.get("id") or job.get("job_id")
                try:
                    result = await process_job(job)
                    outputs.append({"id": job_id, **result})
                except Exception as e:
                    # Include error in result, continue with other jobs
                    logger.error(f"Job {job_id} failed: {e}")
                    outputs.append({
                        "id": job_id,
                        "error": str(e)
                    })

            # Send all results back, including errors
            response = {
                "type": "worker_output",
                "output": outputs
            }
            await websocket.send(dumps(response))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Worker disconnected")
    except Exception as e:
        logger.error(f"Worker crashed: {e}")
        raise
```

### Engine-Side Error Handling

- **Missing resources:** Connection closed with 1009, all queued jobs for that resource failed
- **Invalid job:** Skipped, callback never called
- **Worker crash:** Jobs abandoned, reassigned to other workers after timeout
- **Timeout:** Callback never invoked, job removed after 5 minutes

### Recovery Strategies

1. **Multiple workers:** Distribute across 2+ instances for fault tolerance
2. **Health monitoring:** Check worker connection status
3. **Retry logic:** Wrap job submission with exponential backoff
4. **Graceful degradation:** Return error to client instead of hanging

---

## Best Practices

### Worker Implementation

1. **Register immediately** on connection with proper config
2. **Process all jobs** in batch, even if some fail
3. **Include job ID** in every result output
4. **Use error field** instead of throwing to report issues
5. **Load models once** at startup, reuse across batches
6. **Use thread pools** for CPU-bound work to prevent blocking async loop
7. **Clean up resources** (file handles, GPU memory) on disconnect
8. **Log errors** with job ID for debugging

### Engine Integration (Client)

1. **Send resources with jobs** in single request
2. **Use resource references** to avoid duplication
3. **Register job callbacks** with timeout safety
4. **Clean up results** after processing (for streaming responses)
5. **Monitor job timeouts** (5 minutes max)
6. **Batch multiple jobs** to same worker for efficiency
7. **Validate inputs** before submission (file sizes, required fields)

### Performance Tuning

**Batch Size:**
- Larger batches = higher throughput, higher latency
- Smaller batches = lower latency, lower throughput
- Recommendation: 16-64 jobs per batch depending on model size

**Latency Threshold:**
- Lower values = consistent latency, possible partial batches
- Higher values = full batches, variable latency
- Recommendation: 1000-10000 ms

**Worker Count:**
- More workers = higher parallelism, more resource usage
- Balance based on available GPU/CPU
- Recommendation: 1-4 workers per GPU

---

## Example: Complete Worker Implementation

```python
# worker_custom.py
import asyncio
import os
import logging
from cbor2 import dumps, loads
import websockets

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKER_TYPE = os.getenv("WORKER_TYPE", "custom")
WORKER_SECRET = os.getenv("WORKER_SECRET")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 32))
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", 30000))
SERVER_URL = os.getenv("SERVER_URL", "ws://localhost:5000/ws")

# Load model once
def load_model():
    logger.info("Loading model...")
    # model = ...expensive initialization...
    return model

model = load_model()

async def process_job(job_input):
    """Process a single job from batch"""
    try:
        # Custom processing logic
        text = job_input.get("text", "")
        result = model.process(text)
        return {"output": result}
    except Exception as e:
        logger.error(f"Job processing failed: {e}")
        return {"error": str(e)}

async def main():
    while True:
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                # Register with engine
                registration = {
                    "type": "i_am_worker",
                    "worker_secret": WORKER_SECRET,
                    "worker_config": {
                        "worker_type": WORKER_TYPE,
                        "max_batch_size": MAX_BATCH_SIZE,
                        "max_latency_ms": MAX_LATENCY_MS
                    }
                }
                await websocket.send(dumps(registration))
                logger.info(f"Registered as {WORKER_TYPE}")

                # Process batches
                async for message in websocket:
                    try:
                        batch_data = loads(message)
                        outputs = []

                        for job in batch_data.get("inputs", []):
                            job_id = job.get("id") or job.get("job_id")
                            result = await process_job(job.get("input", {}))
                            outputs.append({"id": job_id, **result})

                        response = {
                            "type": "worker_output",
                            "output": outputs
                        }
                        await websocket.send(dumps(response))

                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")

        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            await asyncio.sleep(5)  # Reconnect after delay
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Testing

### Unit Test Example

```python
# test_worker.py
import asyncio
from cbor2 import dumps, loads

async def test_worker_registration(worker_ws):
    """Test worker can register"""
    msg = {
        "type": "i_am_worker",
        "worker_secret": "test-secret",
        "worker_config": {
            "worker_type": "test",
            "max_batch_size": 10
        }
    }
    await worker_ws.send(dumps(msg))
    # Connection should remain open
    assert not worker_ws.closed

async def test_batch_processing(worker_ws):
    """Test worker processes batch"""
    batch = {
        "inputs": [
            {"id": "job-1", "input": {"text": "hello"}},
            {"id": "job-2", "input": {"text": "world"}}
        ]
    }

    # Send batch
    await worker_ws.send(dumps(batch))

    # Receive results
    response_data = await worker_ws.recv()
    results = loads(response_data)

    assert results["type"] == "worker_output"
    assert len(results["output"]) == 2
    assert {r["id"] for r in results["output"]} == {"job-1", "job-2"}
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Job** | A single unit of work submitted by client |
| **Batch** | Collection of jobs sent to worker in one message |
| **Worker** | Python process that processes jobs of specific type |
| **Resource** | Image, document, or media file referenced by jobs |
| **Resource Reference** | `ResourceRef` object with `id` pointing to actual resource |
| **Job Callback** | Function invoked on engine when job completes |
| **CBOR** | Concise Binary Object Representation (efficient serialization) |
| **cross_job_id** | Session identifier for stateful workers (e.g., SAM3) |
| **max_batch_size** | Maximum jobs in batch before sending |
| **max_latency_ms** | Maximum wait time before partial batch send |

---

## Version History

- **v1.0** (2025-01-24): Initial protocol specification
  - Message types: registration, batch, output
  - Resource system with deduplication
  - 8 worker types documented
  - Timeout and error handling
  - Configuration and best practices

---

## See Also

- `unblink-engine/index.ts` - Engine WebSocket server
- `unblink-engine/src/handle_ws_message.ts` - Message handling logic
- `unblink-engine/engine.d.ts` - TypeScript type definitions
- `unblink-engine/py/ws_client_handler.py` - Shared worker base class
