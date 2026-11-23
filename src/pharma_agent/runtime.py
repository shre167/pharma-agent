"""Runtime: wires settings, services and the knowledge base together."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from .agent.report import generate_report
from .agent.session import AgentSession
from .clients import Services
from .config import Settings, load_settings
from .observability import configure_langsmith
from .retrieval.knowledge_base import KnowledgeBase


@dataclass
class Runtime:
    settings: Settings
    services: Services
    kb: KnowledgeBase

    def new_session(self) -> AgentSession:
        return AgentSession(
            kb=self.kb,
            report_writer=partial(generate_report, self.services.llm_chat),
            wave1_dir=self.settings.wave1_dir,
        )


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or load_settings()
    services = Services(settings)
    configure_langsmith(settings)
    kb = KnowledgeBase.load(settings, services.embeddings)
    return Runtime(settings=settings, services=services, kb=kb)
