import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from typing import Dict, List, Optional

import httpx
from vllm import AsyncEngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils import FlexibleArgumentParser


def get_open_port() -> int:
    """
    Finds a free port on the local machine.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class RemoteOpenAIServer:
    """
    Manages a vLLM OpenAI-compatible server running in a separate process.
    Uses httpx for health checks.
    (Code adapted from vLLM test utilities).
    """

    def __init__(
        self,
        model: str,
        vllm_serve_args: List[str],
        *,
        env_dict: Optional[Dict[str, str]] = None,
        seed: Optional[int] = 0,
        auto_port: bool = True,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        self.model = model
        if auto_port:
            if any(arg.startswith("--port") or arg == "-p" for arg in vllm_serve_args):
                raise ValueError("Port specified manually when auto_port=True.")
            self.port = get_open_port()
            vllm_serve_args = vllm_serve_args + ["--port", str(self.port)]
        else:
            # Simplified port parsing if not auto_port
            found_port = None
            for i, arg in enumerate(vllm_serve_args):
                if arg == "--port" and i + 1 < len(vllm_serve_args):
                    found_port = vllm_serve_args[i + 1]
                    break
                if arg.startswith("--port="):
                    found_port = arg.split("=", 1)[1]
                    break
            if found_port is None:
                raise ValueError("Port must be specified in vllm_serve_args if auto_port=False")
            try:
                self.port = int(found_port)
            except ValueError:
                raise ValueError(f"Invalid port value specified: {found_port}")

        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError(f"Seed specified manually when seed={seed} provided.")
            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        # Parse arguments using vLLM's own parser
        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        all_cli_args = ["--model", model] + vllm_serve_args
        args = parser.parse_args(all_cli_args)
        self.host = str(args.host or "localhost")
        # self.port is set above

        self.show_hidden_metrics = (
            getattr(args, "show_hidden_metrics_for_version", None) is not None
        )

        # download the model before starting the server to avoid timeout
        is_local = os.path.isdir(model)
        if not is_local:
            assert os.getenv("HF_TOKEN", None) is not None, (
                "loading model config requires HF_TOKEN to be set"
            )
            engine_args = AsyncEngineArgs.from_cli_args(args)
            model_config = engine_args.create_model_config()
            load_config = engine_args.create_load_config()

            model_loader = get_model_loader(load_config)
            model_loader.download_model(model_config)

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict is not None:
            env.update(env_dict)

        command = ["vllm", "serve", model] + vllm_serve_args
        print(f"Starting server with command: {' '.join(command)}")

        self.proc = subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stderr)

        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"), timeout=max_wait_seconds)
        print("Server seems ready.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "proc") and self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing.", file=sys.stderr)
                self.proc.kill()
                try:
                    self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    print("Server process kill command timed out.", file=sys.stderr)
            print("Server process stopped.")

    def _wait_for_server(self, *, url: str, timeout: float):
        """Waits for the server to respond using httpx."""
        start_time = time.time()
        last_error = None
        while True:
            # Check if the process exited
            if hasattr(self, "proc") and self.proc:
                result = self.proc.poll()
                if result is not None:
                    time.sleep(0.5)  # Allow potential final messages
                    raise RuntimeError(f"Server process exited unexpectedly with code {result}.")

            try:
                # Use httpx for the GET request
                with httpx.Client(timeout=5.0) as client:  # Use a client context
                    response = client.get(url)  # Short timeout for individual check

                if response.status_code == 200:
                    print(f"Server health check OK at {url}.")
                    break  # Server ready

                else:
                    # Server responded but not with 200 OK
                    last_error = f"Server responded with status {response.status_code}"
                    print(f"WARN: Health check received status {response.status_code}")

            # Handle httpx-specific exceptions
            except httpx.TimeoutException:
                last_error = "Timeout during health check"
                # Expected if server is slow to start, continue waiting
            except httpx.NetworkError as e:
                last_error = f"NetworkError: {e}"
                # Expected initially (e.g., ConnectionRefusedError), continue waiting
            except Exception as e:
                last_error = e
                print(
                    f"WARNING: Unexpected error during health check: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                # Continue waiting, might be temporary

            # Timeout check
            if time.time() - start_time > timeout:
                error_msg = f"Server failed to start and respond at {url} within {timeout} seconds."
                if last_error:
                    error_msg += f" Last error: {last_error}"
                if hasattr(self, "proc") and self.proc:
                    # Attempt cleanup
                    self.proc.terminate()
                    try:
                        self.proc.wait(5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                raise RuntimeError(error_msg)

            time.sleep(1.0)  # Wait before next check

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        path = "/".join(str(p).strip("/") for p in parts if p)
        return f"{self.url_root}/{path}"


_server_instance = None
_server_lock = threading.Lock()


def get_openai_server(model_name, vllm_serve_args, seed, auto_port=False, max_wait_seconds=300):
    global _server_instance
    with _server_lock:
        if _server_instance is None:
            _server_instance = RemoteOpenAIServer(
                model=model_name,
                vllm_serve_args=vllm_serve_args,
                seed=seed,
                auto_port=auto_port,
                max_wait_seconds=max_wait_seconds,
            )
        return _server_instance


def stop_openai_server():
    global _server_instance
    with _server_lock:
        if _server_instance is not None:
            try:
                _server_instance.__exit__(None, None, None)
            except Exception:
                pass
            _server_instance = None
