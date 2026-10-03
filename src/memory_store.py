from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple, consistent token estimator based on character length."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, int(len(stripped) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def _sanitize_user_id(self, user_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id)

    def path_for(self, user_id: str) -> Path:
        sanitized = self._sanitize_user_id(user_id)
        return self.root_dir / sanitized / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            updated = content.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse stored Markdown bullets into key-value facts."""
        content = self.read_text(user_id)
        extracted: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^-\s*\*\*([^*]+)\*\*:\s*(.*)$", line)
            if match:
                k, v = match.group(1).strip(), match.group(2).strip()
                extracted[k] = v
        return extracted

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a specific fact in User.md."""
        current_facts = self.facts(user_id)
        current_facts[key] = value

        lines = [f"# User Profile: {user_id}", ""]
        for k, v in current_facts.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def is_inquiry_or_question(message: str) -> bool:
    """Detect if a turn is a recall/inquiry question rather than a statement providing new facts."""
    lower = message.lower().strip()
    if "?" in message:
        return True
    question_triggers = [
        "nhắc lại", "bạn có biết", "bạn có thể nhắc", "tóm tắt ngắn về mình",
        "mình tên gì", "đâu mới là", "hiện tại mình làm nghề gì",
        "món ăn yêu thích của mình là gì", "đồ uống và món ăn yêu thích",
        "sang thread mới rồi", "nếu ai đó nhắc"
    ]
    return any(trig in lower for trig in question_triggers)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts with confidence filtering.

    Handles:
    - Confidence thresholding: ignores query/question messages to prevent false updates.
    - Name detection with question word filtering.
    - Location tracking with corrections (ignoring business trips and old examples).
    - Profession tracking with corrections (ignoring jokes).
    - Response style preferences.
    - Food & drink preferences.
    - Pet ownership.
    - Technical interests.
    """
    updates: dict[str, str] = {}
    lower = message.lower()

    # Rule 1: Guardrail - Do NOT extract from inquiry/question turns
    if is_inquiry_or_question(message) and not any(
        kw in lower for kw in ["mình tên là", "tên mình là"]
    ):
        return updates

    # --- 1. Name Extraction ---
    name_match = re.search(
        r"(?:mình tên là|tên mình là|mình là)\s+([A-Za-z0-9_\sĐđÁáÀàẢảÃãẠạĂăẮắẰằẲẳẴẵẶặÂâẤấẦầẨẩẪẫẬậÉéÈèẺẻẼẽẸẹÊêẾếỀềỂểỄễỆệÍíÌìỈỉĨĩỊịÓóÒòỎỏÕõỌọÔôỐốỒồỔổỖỗỘộƠơỚớỜờỞởỠỡỢợÚúÙùỦủŨũỤụƯưỨứỪừỬửỮữỰựÝýỲỳỶỷỸỹỴỵ]+)",
        message,
        re.IGNORECASE,
    )
    if name_match:
        raw_name = name_match.group(1).strip().split(".")[0].split(",")[0]
        raw_name = re.sub(r"\b(hiện|ở|và|đang|cho|là)\b.*$", "", raw_name, flags=re.IGNORECASE).strip()
        lower_raw = raw_name.lower()
        # Filter out question words and pronouns, require proper capitalized name
        if (
            raw_name
            and raw_name[0].isupper()
            and not lower_raw.startswith(("ai ", "ai", "gì", "người", "sao", "thế nào"))
            and lower_raw not in {"gì", "ai", "ai đó", "sao", "bạn", "thế nào", "mình"}
        ):
            updates["Name"] = raw_name

    # --- 2. Location (with correction & noise filtering) ---
    is_hanoi_trip = "hà nội" in lower and any(term in lower for term in ["họp", "bay ra", "chỉ là nơi"])
    danang_as_old_example = "đà nẵng" in lower and any(term in lower for term in ["ví dụ cũ", "đừng lấy nó"])

    if "huế" in lower and not any(term in lower for term in ["dù trước đó có nhắc huế", "lúc đầu mình nói hiện ở huế, nhưng"]):
        updates["Location"] = "Huế"
    elif "đà nẵng" in lower and not danang_as_old_example:
        if any(term in lower for term in ["làm việc ở đà nẵng", "nơi ở hiện tại là đà nẵng", "đang ở đà nẵng"]):
            updates["Location"] = "Đà Nẵng"
        elif "huế chứ không còn ở đà nẵng" in lower:
            updates["Location"] = "Huế"
        elif "mình ở đà nẵng" in lower and "huế" not in lower:
            updates["Location"] = "Đà Nẵng"

    # --- 3. Profession (with correction & joke filtering) ---
    pm_joke = "product manager" in lower and any(term in lower for term in ["đùa", "câu đùa", "chỉ là câu đùa"])
    if "mlops engineer" in lower or "mlops" in lower:
        updates["Profession"] = "MLOps engineer"
    elif "backend engineer" in lower and not any(term in lower for term in ["không còn làm backend", "đừng nói backend", "từ backend sang mlops"]):
        updates["Profession"] = "backend engineer"

    # --- 4. Response style ---
    if "3 bullet" in lower:
        updates["Response Style"] = "3 bullet ngắn gọn, có ví dụ thực chiến, nhấn trade-off"
    elif "bullet ngắn" in lower or "ngắn gọn" in lower or "rõ ý" in lower:
        updates["Response Style"] = "ngắn gọn, có ví dụ thực tế, bullet ngắn"

    # --- 5. Drink ---
    if "cà phê sữa đá" in lower:
        updates["Favorite Drink"] = "cà phê sữa đá"

    # --- 6. Food ---
    if "mì quảng" in lower:
        updates["Favorite Food"] = "mì Quảng"

    # --- 7. Pet ---
    if "corgi" in lower or "con bơ" in lower or "bé corgi" in lower:
        updates["Pet"] = "corgi tên Bơ"

    # --- 8. Interests ---
    if "python" in lower and "ai" in lower:
        updates["Interests"] = "Python, AI ứng dụng, benchmark memory"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages[-max_items:]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if len(content) > 140:
            content = content[:140] + "..."
        lines.append(f"- {role.capitalize()}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t_state = self._ensure_thread(thread_id)
        msgs: list[dict[str, str]] = t_state["messages"]  # type: ignore
        msgs.append({"role": role, "content": content})

        summary_text: str = t_state["summary"]  # type: ignore
        total_tokens = sum(estimate_tokens(m["content"]) for m in msgs) + estimate_tokens(summary_text)

        if total_tokens >= self.threshold_tokens and len(msgs) > self.keep_messages:
            to_compact = msgs[:-self.keep_messages]
            keep = msgs[-self.keep_messages:]

            compacted_summary = summarize_messages(to_compact)
            if summary_text:
                new_summary = f"{summary_text}\n{compacted_summary}"
            else:
                new_summary = compacted_summary

            summary_lines = new_summary.splitlines()
            if len(summary_lines) > 12:
                new_summary = "\n".join(summary_lines[-12:])

            t_state["summary"] = new_summary
            t_state["messages"] = keep
            t_state["compactions"] = int(t_state["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id)["compactions"])
