"""Nomic embedding service using Triton Inference Server."""
import aiohttp
import logging

logger = logging.getLogger("scripts")


class NomicTritonEmbeddingService:
    """Embedding service for self-hosted Nomic model on Triton Inference Server.
    
    Compatible with the OpenAIEmbeddings interface used by prepdocs and approaches.
    """

    def __init__(self, endpoint, dimensions=768, cf_client_id="", cf_client_secret=""):
        self.endpoint = endpoint
        self.open_ai_dimensions = dimensions
        self.open_ai_model_name = "nomic-embed"
        self.cf_headers = {}
        if cf_client_id:
            self.cf_headers["CF-Access-Client-Id"] = cf_client_id
        if cf_client_secret:
            self.cf_headers["CF-Access-Client-Secret"] = cf_client_secret

    async def create_embeddings(self, texts):
        """Compute embeddings for document texts (uses search_document: prefix)."""
        prefixed = [f"search_document: {t}" for t in texts]
        return await self._call_triton(prefixed, len(texts))

    async def create_query_embedding(self, text):
        """Compute embedding for a search query (uses search_query: prefix)."""
        result = await self._call_triton([f"search_query: {text}"], 1)
        return result[0]

    async def _call_triton(self, texts, count):
        headers = {"Content-Type": "application/json", **self.cf_headers}
        body = {
            "inputs": [{"name": "text", "datatype": "BYTES", "shape": [count, 1], "data": texts}],
            "outputs": [{"name": "embeddings"}],
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(self.endpoint, json=body) as resp:
                if resp.status != 200:
                    raise Exception(f"Nomic embedding failed: {resp.status} {await resp.text()}")
                result = await resp.json()
                flat = result["outputs"][0]["data"]
                dim = self.open_ai_dimensions
                return [flat[i * dim : (i + 1) * dim] for i in range(count)]
