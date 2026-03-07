import base64
import os
from typing import Any, Dict, List
import httpx
import requests
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from cat.services.factory.embedder import MultimodalEmbeddings
from cat.utils import retrieve_image


class CustomOpenAIEmbeddings(Embeddings):
    """Use OpenAI-compatible API as embedder (like llama-cpp-python)."""
    def __init__(self, url, model):
        self.url = os.path.join(url, "v1/embeddings")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API expects JSON payload, not form data
        ret = httpx.post(self.url, json={"model": self.model, "input": texts}, timeout=300.0)
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        # OpenAI API expects JSON payload, not form data
        ret = httpx.post(self.url, json={"model": self.model, "input": text}, timeout=300.0)
        ret.raise_for_status()
        return ret.json()["data"][0]["embedding"]


class CustomOllamaEmbeddings(Embeddings):
    """Use Ollama to serve embedding models."""
    def __init__(self, base_url, model):
        self.url = os.path.join(base_url, "api/embeddings")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't support batch processing, so we need to process one by one
        embeddings = []
        for text in texts:
            ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
            ret.raise_for_status()
            embeddings.append(ret.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
        ret.raise_for_status()
        return ret.json()["embedding"]


class CustomJinaEmbedder(Embeddings):
    """Use Jina AI to serve embedding models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching"):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task

    def _embed(self, texts: List[str]) -> List[List[float]]:
        ret = httpx.post(
            self.url,
            data={"model": self.model, "input": texts, "task": self.task},
            timeout=300.0,
            headers=self.headers,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class Qwen3LocalEmbeddings(Embeddings):
    """
    Local Qwen3 embeddings using HuggingFace Sentence Transformers.
    Best for: Full control, no external dependencies, offline usage
    """
    def __init__(self, model_name: str, device: str = "cuda", model: Any = None):
        self.model_name = model_name
        self.device = device
        self.model = model
        self._load_model()

    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        self._load_model()
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        self._load_model()
        embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


class Qwen3OllamaEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Ollama.
    Best for: Easy local deployment, minimal setup
    """
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)


class Qwen3DeepInfraEmbeddings(Embeddings):
    """
    Qwen3 embeddings via DeepInfra API (OpenAI-compatible).
    Best for: Production deployment, no GPU required, pay-as-you-go
    """
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from DeepInfra"""
        if not self.api_key:
            raise ValueError("DeepInfra API key is required")

        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "input": texts,
                    "model": self.model_name,
                    "encoding_format": "float"
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_embeddings]
        except requests.RequestException as e:
            raise RuntimeError(f"DeepInfra embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        # DeepInfra supports batch processing
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class Qwen3TEIEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Text Embeddings Inference (self-hosted).
    Best for: High-throughput production, full control, optimized inference
    """
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from TEI server"""
        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"TEI embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class CustomJinaMultimodalEmbedder(MultimodalEmbeddings):
    """Use Jina AI to serve embedding multimodal models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching"):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task

    def _embed(
        self,
        texts: List[str] | None = None,
        images: List[str | bytes] | None = None
    ) -> List[List[float]]:
        def parse_image(image: str | bytes) -> str:
            if isinstance(image, bytes):
                return base64.b64encode(image).decode("utf-8")
            image = retrieve_image(image)
            # remove "data:image/...;base64," prefix if present
            if image is not None and image.startswith("data:image"):
                image = image.split(",", 1)[1]
            return image

        payload = (
            [{"text": t} for t in texts] if texts else []
        ) + (
            [{"image": parse_image(i)} for i in images if i] if images else []
        )

        if not payload:
            return []

        ret = httpx.post(
            self.url,
            json={"model": self.model, "input": payload, "task": self.task},
            headers=self.headers,
            timeout=300.0,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(texts=[text])[0]

    def embed_image(self, image: str | bytes) -> List[float]:
        return self._embed(images=[image])[0]

    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        return self._embed(images=images)


class JinaCLIPEmbeddings(MultimodalEmbeddings):
    """
    Jina CLIP v2 multimodal embeddings.
    Handles both text and images in same vector space.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def _get_embeddings(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Get embeddings from Jina API.

        Args:
            inputs: List of {"text": str} or {"image": bytes/url}
        """
        if not self.api_key:
            raise ValueError("Jina API key required")

        try:
            # Prepare input for Jina API
            prepared_inputs = []
            for inp in inputs:
                if "text" in inp:
                    prepared_inputs.append({"text": inp["text"]})
                elif "image" in inp:
                    # Handle image bytes or URL
                    tmp = inp["image"]
                    if isinstance(inp["image"], bytes):
                        img_b64 = base64.b64encode(inp["image"]).decode()
                        tmp = f"data:image/png;base64,{img_b64}"

                    prepared_inputs.append({"image": tmp})

            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model_name,
                    "input": prepared_inputs
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            return [item["embedding"] for item in data["data"]]

        except requests.RequestException as e:
            raise RuntimeError(f"Jina embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents"""
        inputs = [{"text": text} for text in texts]
        return self._get_embeddings(inputs)

    def embed_query(self, text: str) -> List[float]:
        """Embed single text query"""
        return self._get_embeddings([{"text": text}])[0]

    def embed_image(self, image: bytes) -> List[float]:
        """Embed single image"""
        return self._get_embeddings([{"image": image}])[0]

    def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """Embed multiple images"""
        inputs = [{"image": img} for img in images]
        return self._get_embeddings(inputs)
