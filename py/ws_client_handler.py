import asyncio
import websockets
import concurrent.futures
import time
import json
import random
import os
from cbor2 import dumps, loads

def parse_env():
    """
    Parses worker configuration from environment variables.
    Returns a dictionary containing only the settings found in the environment.
    """
    env_config = {}
    max_latency_ms_str = os.environ.get("MAX_LATENCY_MS")
    worker_type_str = os.environ.get("WORKER_TYPE")
    if not worker_type_str:
        print("Error: WORKER_TYPE environment variable is not set.")
        exit(1)
    env_config["worker_type"] = worker_type_str

    if not max_latency_ms_str:
        print("Error: MAX_LATENCY_MS environment variable is not set.")
        exit(1)

    if max_latency_ms_str:
        try:
            env_config["max_latency_ms"] = int(max_latency_ms_str)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse MAX_LATENCY_MS from environment variable. Value: '{max_latency_ms_str}'")
    
    return env_config

async def client_handler(heavy_ai_workload):
    """
    Connects to the server with a robust, exponential backoff retry mechanism.
    """
    uri = "ws://localhost:5000/ws"
    print(f"Connecting to server at {uri}...")
    if uri is None:
        print("Error: BACKEND_WS_URL environment variable is not set.")
        return
    
    # --- Retry Logic Variables ---
    initial_delay = 1.0
    max_delay = 60.0
    reconnect_delay = initial_delay
    
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    # If the connection is successful, print a confirmation
                    # and RESET the reconnect delay to its initial value.
                    print(f"[Main] Connection successful to {uri}.")
                    reconnect_delay = initial_delay
                    
                    # 2. Parse environment variables to get any overrides.
                    worker_config = parse_env()
                    msg = {"type": "i_am_worker", "worker_config": worker_config }
                    encoded = dumps(msg)
                    await websocket.send(encoded)

                    async for message in websocket:
                        print(f"[Main] Received task from server: {message}")
                        task_data = loads(message)
                        loop = asyncio.get_running_loop()
                        
                        print("[Main] Offloading AI task to executor thread...")
                        result_json = await loop.run_in_executor(
                            pool, heavy_ai_workload, task_data
                        )
                        
                        print(f"[Main] Sending result to server: {result_json}")
                        encoded = dumps(result_json)
                        await websocket.send(encoded)

            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
                print(f"[Main] Connection failed: {e}")
                print(f"Attempting to reconnect in {reconnect_delay:.2f} seconds...")
                
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay) + random.uniform(0, 1)

            except Exception as e:
                print(f"[Main] An unexpected error occurred: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)