"""Capgemini-backed LLM, chat helper and Titan embeddings."""
from __future__ import annotations

import time
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langsmith import traceable
from openai import OpenAI
from tqdm.auto import tqdm

from .config import Settings
from .observability import trace_inputs, trace_output


class CapgeminiTitanEmbeddings(Embeddings):
    """LangChain Embeddings adapter for Titan served by Capgemini's /embeddings API."""

    def __init__(self, api_client: OpenAI, model: str, batch_size: int = 16, show_progress: bool = False):
        self.api_client = api_client
        self.model = model
        self.batch_size = batch_size
        self.show_progress = show_progress

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        starts = range(0, len(texts), self.batch_size)
        if self.show_progress:
            starts = tqdm(starts, desc="Titan embeddings")
        for start in starts:
            batch = texts[start:start + self.batch_size]
            response = self.api_client.embeddings.create(model=self.model, input=batch)
            batch_vectors = [item.embedding for item in response.data]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(f"Titan returned {len(batch_vectors)} vectors for {len(batch)} texts.")
            vectors.extend(batch_vectors)
            time.sleep(0.05)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.api_client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


@traceable(name="capgemini_chat_call", run_type="llm", process_inputs=trace_inputs, process_outputs=trace_output)
def _chat_call(client: OpenAI, model: str, messages: list[dict], response_format: Any = None) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The LLM returned an empty response.")
    return content.strip()


class Services:
    """Bundles the OpenAI-compatible client, LangChain chat model and embeddings."""

    def __init__(self, settings: Settings, show_embedding_progress: bool = False):
        if not settings.ge_api_key:
            raise ValueError("GE_API_KEY is missing. Set it in .env or enter it in the app sidebar.")
        self.settings = settings
        self.client = OpenAI(api_key=settings.ge_api_key, base_url=settings.llm_base_url,
                             timeout=180.0, max_retries=1)
        self.chat_model = ChatOpenAI(
            model=settings.llm_model, api_key=settings.ge_api_key, base_url=settings.llm_base_url,
            temperature=settings.llm_temperature, timeout=180, max_retries=1, stream_usage=False,
        )
        self.embeddings = CapgeminiTitanEmbeddings(self.client, settings.embedding_model,
                                                   show_progress=show_embedding_progress)

    def llm_chat(self, messages: list[dict], response_format: Any = None) -> str:
        return _chat_call(self.client, self.settings.llm_model, messages, response_format)

    # ---- connectivity checks (used by scripts/check_setup.py and the UI) ----
    def test_chat(self) -> str:
        return self.llm_chat([{"role": "user", "content": "Reply with exactly: CAPGEMINI_CHAT_OK"}])

    def test_embeddings(self) -> int:
        resp = self.client.embeddings.create(model=self.settings.embedding_model,
                                             input=["CAPGEMINI_TITAN_EMBEDDING_TEST"])
        return len(resp.data[0].embedding)

    def test_tool_calling(self) -> bool:
        from langchain_core.tools import tool

        @tool
        def tool_calling_smoke_test(value: str) -> str:
            """Return the supplied value. Used only to verify provider tool-calling compatibility."""
            return value

        msg = self.chat_model.bind_tools([tool_calling_smoke_test]).invoke(
            "Call tool_calling_smoke_test with value exactly 'CAPGEMINI_TOOL_CALL_OK'.")
        return bool(getattr(msg, "tool_calls", None))
