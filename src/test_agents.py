from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit and integration testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)

    user_id = "test_user"
    init_content = "# User Profile: test_user\n\n- **Location**: Đà Nẵng\n"
    created_path = store.write_text(user_id, init_content)
    assert created_path.exists()
    assert "Đà Nẵng" in store.read_text(user_id)

    # Edit fact in markdown
    edited = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert edited is True
    updated_content = store.read_text(user_id)
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content
    assert store.file_size(user_id) > 0

    # Upsert structured fact
    store.upsert_fact(user_id, "Profession", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("Profession") == "MLOps engineer"
    assert facts.get("Location") == "Huế"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction once token threshold is exceeded."""
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "long_test_thread"

    # Append turns that exceed threshold (each turn ~ 30 tokens)
    long_msg = "Đây là một lượt tin nhắn rất dài nhằm mục đích kiểm tra kích hoạt compact memory cho agent khi vượt ngưỡng token."
    manager.append(thread_id, "user", long_msg)
    manager.append(thread_id, "assistant", "Phản hồi thứ nhất từ trợ lý cũng có độ dài đáng kể để tăng token.")
    manager.append(thread_id, "user", "Tin nhắn thứ ba tiếp tục đẩy tổng số token của cuộc hội thoại lên cao hơn nữa.")
    manager.append(thread_id, "assistant", "Phản hồi thứ tư sẽ kích hoạt quá trình tóm tắt và nén lịch sử.")

    # Compaction should have fired
    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2  # type: ignore
    assert len(str(ctx["summary"])) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced agent remembers facts across fresh sessions/threads while baseline forgets."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_recall_test"
    thread_1 = "session_one"

    # Initial assertion
    intro_message = "Chào bạn, mình tên là DũngCT, hiện ở Huế và đang làm MLOps engineer."
    baseline.reply(user_id, thread_1, intro_message)
    advanced.reply(user_id, thread_1, intro_message)

    # In session 2 (brand-new thread)
    thread_2 = "session_two"
    recall_question = "Mình tên gì và hiện đang ở đâu?"

    baseline_ans = baseline.reply(user_id, thread_2, recall_question)["reply"]
    advanced_ans = advanced.reply(user_id, thread_2, recall_question)["reply"]

    # Baseline forgets across new thread
    assert "DũngCT" not in baseline_ans
    assert "Huế" not in baseline_ans

    # Advanced recalls via persistent User.md
    assert "DũngCT" in advanced_ans
    assert "Huế" in advanced_ans


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt token load of baseline vs advanced on a long conversation thread."""
    config = make_config(tmp_path)
    # Give a reasonable threshold so compaction happens repeatedly
    config.compact_threshold_tokens = 60
    config.compact_keep_messages = 2

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "test_efficiency"
    thread_id = "thread_stress"

    long_turns = [
        f"Lượt hội thoại số {i}: Đoạn văn dài chứa rất nhiều chi tiết kỹ thuật về hệ thống AI, kiến trúc memory, RAG và evaluation để làm tăng độ dài ngữ cảnh cần xử lý."
        for i in range(10)
    ]

    for turn in long_turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced agent must process significantly fewer prompt tokens thanks to compacting
    assert advanced_prompt_tokens < baseline_prompt_tokens, (
        f"Expected advanced prompt tokens ({advanced_prompt_tokens}) to be less than baseline ({baseline_prompt_tokens})"
    )
