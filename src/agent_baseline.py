from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Retains context strictly within the same session/thread
    - No persistent User.md storage
    - No compact memory mechanism
    - Forgets all long-term facts when queried in a new thread
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline and self.config.model.api_key:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process incoming user turn and return response with token accounting."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                # Live LangChain invocation if available
                prompt_tokens = sum(
                    estimate_tokens(m["content"]) for m in self._get_session(thread_id).messages
                ) + estimate_tokens(message)
                response = self.langchain_agent.invoke(
                    {"input": message},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_text = response.get("output", str(response))
                agent_tokens = estimate_tokens(output_text)

                session = self._get_session(thread_id)
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output_text})
                session.token_usage += agent_tokens
                session.prompt_tokens_processed += prompt_tokens

                return {
                    "reply": output_text,
                    "tokens": agent_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                # Fallback to offline mode on any live execution issue
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent response tokens for a thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt tokens processed for a thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline agent does not have a compact memory layer."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline mode for reproducible benchmarking and testing."""
        session = self._get_session(thread_id)

        # Baseline must carry the full uncompressed thread history in prompt context
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens

        # Because baseline only has within-thread memory, it cannot recall anything
        # if the query is asked in a new/fresh thread.
        lower = message.lower()
        if len(session.messages) == 0:
            if any(q in lower for q in ["tên", "nghề", "ở đâu", "uống", "món", "corgi", "style", "ai"]):
                reply_text = "Tôi là trợ lý ảo. Tôi không có thông tin cá nhân của bạn trong phiên trò chuyện mới này."
            else:
                reply_text = "Chào bạn! Tôi đã ghi nhận thông tin."
        else:
            # Within same thread, basic acknowledgement
            reply_text = f"Tôi đã ghi nhận: '{message[:50]}...'. Lịch sử phiên hiện có {len(session.messages)} lượt."

        agent_tokens = estimate_tokens(reply_text)
        session.token_usage += agent_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build LangChain runnable if packages are available."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        model = build_chat_model(self.config.model)
        checkpointer = MemorySaver()
        return create_react_agent(model, tools=[], checkpointer=checkpointer)
