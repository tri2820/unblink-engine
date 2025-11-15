This repository contains the AI inference engine for the [Unblink](https://github.com/tri2820/unblink) camera monitoring application. It handles all the heavy lifting of running deep learning models for object detection and vision-language understanding.

This engine is designed to be run on a separate, GPU-accelerated machine for optimal performance, though it can also run on a CPU.

## Tech Stack

-   **Backend Wrapper:** [Bun](https://bun.sh) / TypeScript for handling WebSocket connections and managing the Python processes.
-   **AI Core:** Python for all model inference tasks.
-   **GPU Acceleration:** Optimized for NVIDIA GPUs via CUDA.

## Getting Started

### Prerequisites

-   **Hardware:** An **NVIDIA GPU with CUDA 12.1+** installed is highly recommended for real-time performance. CPU-only is possible but will be very slow.
-   **Software:**
    -   [Git](https://git-scm.com/)
    -   Python 3.12+
    -   [uv](https://github.com/astral-sh/uv) (recommended Python package manager) or `pip`.
    -   [Bun](https://bun.sh) runtime.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tri2820/unblink-engine
    cd unblink-engine
    ```

2.  **Set up the Python Environment:**

    Use `uv` to create a virtual environment and install dependencies.

    ```bash
    # Create a virtual environment
    uv venv

    # Activate the environment
    source .venv/bin/activate

    # Install Python dependencies
    uv sync
    ```

3.  **Install Node.js Dependencies:**
    ```bash
    bun install
    ```

### Running the Engine

The easiest way to start the engine is by using the provided script. It handles launching both the Python services and the Bun WebSocket server.

```bash
sh run.sh
```


## Connecting with the Unblink Client

This engine is designed to be used with the main [Unblink application](https://github.com/tri2820/unblink). To make your Unblink instance connect to your self-hosted engine, set the `ENGINE_URL` environment variable when running Unblink.

For example, if your engine is running at `192.168.1.100:8000`, you would start the Unblink client like this:

```bash
ENGINE_URL=ws://192.168.1.100:8000 bun dev
```

## Contributing

Contributions are welcome! If you have suggestions for performance improvements, new models, or bug fixes, please feel free to submit an issue or a pull request.