import os
import time
from pathlib import Path
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# 配置 logging
logger = logging.getLogger(__name__)

def check_response(response):
    if not response.ok:
        print(f"Error detected: {response.status_code}")
        print("Response Body:", response.text)
    response.raise_for_status()

@dataclass
class ModelInfo:
    """Model information returned by the server"""
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "organization-owner"
    model_path: Optional[str] = None
    model_type: Optional[str] = None


@dataclass
class LaunchResult:
    """Result of launching a model"""
    model_id: str
    model_path: str
    model_type: Optional[str]
    status: str


@dataclass
class TerminateResult:
    """Result of terminating a model"""
    model_id: str
    status: str


class InferenceClient:
    """Client for interacting with the model server"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the client.

        Args:
            base_url: The base URL of the model server.
            wait_for_ready: If True, blocks initialization until the server returns 200 OK on health check.
            timeout: Maximum seconds to wait if wait_for_ready is True.
        """
        self.base_url = base_url.rstrip('/')
        self._headers: Dict[str, str] = {}
        self.session = requests.Session()
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()
        
    def wait_for_server_ready(self, timeout: int = 60, health_endpoint: str = "/health") -> bool:
        """
        Waits for the server to become ready by polling the health endpoint.
        
        Args:
            timeout: Maximum time to wait in seconds.
            health_endpoint: The endpoint to check (default: /health). 
                             If your server doesn't have /health, try /v1/models.
        
        Returns:
            True if server is ready, False if timed out.
        """
        start_time = time.time()
        url = f"{self.base_url}{health_endpoint}"
        logger.info(f"Waiting for server at {url} to be ready...")

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(url, timeout=2) # Short timeout for the request itself
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return True
                else:
                    logger.debug(f"Server responded with status {response.status_code}, waiting...")
            except requests.exceptions.ConnectionError:
                logger.debug("Connection refused (server likely initializing), retrying...")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check warning: {e}")

            time.sleep(1)
        
        logger.error("Timed out waiting for server to be ready.")
        raise TimeoutError("Timed out waiting for server to be ready.")

    def check_health(self) -> bool:
        """
        One-off check to see if server is healthy.
        """
        url = f"{self.base_url}/health" # 假设是 /health，如果不是请修改
        try:
            response = self.session.get(url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def launch_model(
        self,
        model_path: str,
        model_type: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LaunchResult:
        """
        Launch a new model on the server

        Args:
            model_path: Path to the model directory
            model_type: Type of the model (optional)
            model: Model name/identifier (optional, defaults to model_path)

        Returns:
            LaunchResult with model information

        Raises:
            requests.RequestException: If the request fails
        """
        payload = {
            "model_path": model_path,
            "model_type": model_type,
            "model": model,
            "kwargs": kwargs
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        url = f"{self.base_url}/v1/models"

        response = self.session.post(url, json=payload, headers=self._headers)
        check_response(response)
        data = response.json()

        return LaunchResult(
            model_id=data["model_id"],
            model_path=data["model_path"],
            model_type=data.get("model_type"),
            status=data["status"]
        )

    def terminate_model(self, model_id: str) -> TerminateResult:
        """
        Terminate a running model

        Args:
            model_id: ID of the model to terminate

        Returns:
            TerminateResult with termination status

        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/v1/models/{model_id}"

        response = self.session.delete(url, headers=self._headers)
        check_response(response)
        data = response.json()

        return TerminateResult(
            model_id=data["model_id"],
            status=data["status"]
        )

    def list_models(self) -> List[ModelInfo]:
        """
        List all loaded models

        Returns:
            List of ModelInfo objects

        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/v1/models"

        response = self.session.get(url, headers=self._headers)
        check_response(response)
        data = response.json()
        models = []
        for model_data in data.get("data", []):
            models.append(ModelInfo(
                id=model_data["id"],
                object=model_data.get("object", "model"),
                created=model_data.get("created", 1677610602),
                owned_by=model_data.get("owned_by", "organization-owner"),
                model_path=model_data.get("model_path"),
                model_type=model_data.get("model_type")
            ))

        return models

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request

        Args:
            model: Model ID to use
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters

        Returns:
            Response from the model in OpenAI format

        Raises:
            requests.RequestException: If the request fails
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }

        url = f"{self.base_url}/v1/chat/completions"

        response = self.session.post(url, json=payload, headers=self._headers)
        check_response(response)
        return response.json()

    def close(self):
        """Close the client session"""
        self.session.close()


# Example usage
if __name__ == "__main__":
    from openai import OpenAI
    from jamel.utils.general_utils import make_user
    # Example: Launch a model, list models, then terminate it
    client_url = "http://localhost:3210"
    inference_client = InferenceClient(client_url)
    openai_client = OpenAI(
        api_key="",
        base_url=client_url
    )
    try:
        # List all models
        models = inference_client.list_models()
        print(f"Available models: {[m.id for m in models]}")

        # Launch a model
        result = inference_client.launch_model(
            model_path=str(Path(os.getenv("MODELSCOPE_CACHE")) / "models/Qwen/Qwen2___5-VL-3B-Instruct/"),
            model_type="qwen_2_vl",
            model=None
        )
        print(f"Launched model: {result.model_id}, status: {result.status}")
        # List all models
        models = inference_client.list_models()
        print(f"Available models: {[m.id for m in models]}")
        for m in models:
            print(f"try to request model `{m.id}`")
            response = openai_client.chat.completions.create(
                model=m.id,
                messages=[make_user("hello").serialize()],
            )
            print(f"response of model `{m.id}`: {response}")

        # Terminate the model
        terminate_result = inference_client.terminate_model(result.model_id)
        print(f"Terminated model: {terminate_result.model_id}, status: {terminate_result.status}")
        try:
            print(f"try to request model `{result.model_id}`")
            response = openai_client.chat.completions.create(
                model=result.model_id,
                messages=[make_user("hello").serialize()],
            )
            print(f"response of model `{result.model_id}`: {response}")
        except Exception as e:
            print(f"error as expected! error: {e}")

    finally:
        inference_client.close()