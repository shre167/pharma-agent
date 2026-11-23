"""End-to-end: gate -> guard -> real LangChain agent loop -> report -> verification -> review.

The LLM is scripted, so this exercises our wiring (runner, tools, session, pipeline, storage)
without any API calls.
"""
import json
from types import SimpleNamespace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from pharma_agent.agent import prompts
from pharma_agent.runtime import Runtime
from pharma_agent.wave2.pipeline import run_generation
from pharma_agent.wave2.storage import load_report, submit_review


class ScriptedChat(BaseChatModel):
    script: list
    idx: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[self.idx](messages)
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


def agent_script():
    def search(_):
        return AIMessage(content="", tool_calls=[{"id": "t1", "name": "find_documents", "args": {
            "search_query": "acetaminophen tablets bom process", "top_k": 4,
            "alternate_names": ["paracetamol"]}}])

    def build(messages):
        tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        ids = [r["evidence_id"] for r in json.loads(tool_msg.content)["results"]]
        return AIMessage(content="", tool_calls=[{"id": "t2", "name": "build_report", "args": {
            "report_request": "design report", "evidence_ids": ids, "known_gaps": ["strength not stated"]}}])

    def final(_):
        return AIMessage(content="Searched acetaminophen; design report created.\n\nIf you want, I can do more.")

    return [search, build, final]


def make_runtime(settings, small_kb, scope_payload):
    def llm_chat(messages):
        system = messages[0]["content"]
        if system == prompts.SCOPE_PROMPT:
            return json.dumps(scope_payload)
        if system == prompts.INPUT_GUARD_PROMPT:
            return json.dumps({"action": "allow", "reason": "ordinary request"})
        if system == prompts.REPORT_SYSTEM_PROMPT:
            return "## 2. BOM\nAcetaminophen 500 mg [E1] [SOURCE: handbook.pdf, PDF page 1]"
        if system == prompts.VERIFY_PROMPT:
            return json.dumps({"verdict": "pass", "reason": "supported", "issues": []})
        raise AssertionError("unexpected LLM call")

    services = SimpleNamespace(llm_chat=llm_chat, chat_model=ScriptedChat(script=agent_script()))
    return Runtime(settings=settings, services=services, kb=small_kb)


def test_full_run_then_human_finalisation(settings, small_kb):
    rt = make_runtime(settings, small_kb, {
        "medicine_mentions": ["acetaminophen"], "matched_handbook_ingredients": ["acetaminophen"],
        "identification_method": "explicit_name", "confidence": "high"})
    events = []
    result = run_generation(rt, "Prepare a report for acetaminophen tablets", on_event=lambda s, p: events.append(s))

    assert result["status"] == "REPORT_GENERATED_PENDING_REVIEW", result.get("reason")
    assert [c["tool"] for c in result["tool_calls"]] == ["find_documents", "build_report"]
    assert events[0] == "scope" and "guard" in events and events.count("tool_call") == 2 and "verify" in events
    assert result["output_verdict"] == "PASS"

    manifest, text = load_report(settings, result["report_id"])
    assert manifest["status"] == "REPORT_GENERATED_PENDING_REVIEW" and "Acetaminophen 500 mg" in text
    assert manifest["evidence"][0]["tag"] == "E1"
    assert manifest["review_decision"]["decision"] == "pending"      # never auto-approved

    done = submit_review(settings, result["report_id"], "approve", "Dr A", "ok")
    assert done["status"] == "FINALISED" and (settings.approved_dir / f"{result['report_id']}.md").exists()


def test_out_of_scope_request_never_reaches_the_agent(settings, small_kb):
    rt = make_runtime(settings, small_kb, {
        "is_medicine_request": False, "reason": "confectionery", "medicine_mentions": ["cadbury gems"],
        "matched_handbook_ingredients": [], "identification_method": "explicit_name", "confidence": "high"})
    result = run_generation(rt, "manufacturing process of cadbury gems")
    assert result["status"] == "REFUSED" and result["report"] is None and result["tool_calls"] == []
    assert rt.services.chat_model.idx == 0, "agent LLM must not be invoked for refused requests"
