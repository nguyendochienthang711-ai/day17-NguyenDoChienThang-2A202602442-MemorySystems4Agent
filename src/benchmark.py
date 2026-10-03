from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return proportion of expected facts appearing in the answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Evaluate response quality based on recall and responsiveness."""
    rec = recall_points(answer, expected)
    if rec == 1.0:
        return 1.0
    elif rec > 0.0:
        return round(0.5 + 0.5 * rec, 2)
    return 0.0


def run_agent_benchmark(agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions across new threads."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    user_ids: set[str] = set()
    threads: list[str] = []

    for conv in conversations:
        user_id = conv["user_id"]
        user_ids.add(user_id)
        conv_id = conv["id"]
        main_thread = f"{agent_name}_main_{conv_id}"
        threads.append(main_thread)

        # 1. Feed regular dialogue turns in sequence
        for turn in conv.get("turns", []):
            res = agent.reply(user_id, main_thread, turn)
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]

        # 2. Ask recall questions in a brand-new thread to verify cross-session memory
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"{agent_name}_recall_{conv_id}_{idx}"
            threads.append(recall_thread)
            q_text = q_item["question"]
            expected = q_item["expected_contains"]

            res = agent.reply(user_id, recall_thread, q_text)
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]

            r_score = recall_points(res["reply"], expected)
            q_score = heuristic_quality(res["reply"], expected)
            recall_scores.append(r_score)
            quality_scores.append(q_score)

    # 3. Compute memory growth and compactions
    total_compactions = sum(agent.compaction_count(t) for t in threads)
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in user_ids)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            r.memory_growth_bytes,
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run standard and long-context stress benchmarks and display comparison tables."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    # Dataset 1: Standard benchmark
    std_data_path = config.data_dir / "conversations.json"
    std_convs = load_conversations(std_data_path)

    # Reset/clean state directory for clean benchmark run
    import shutil
    if config.state_dir.exists():
        shutil.rmtree(config.state_dir, ignore_errors=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    print("Running Standard Benchmark...")
    baseline_agent_std = BaselineAgent(config, force_offline=True)
    advanced_agent_std = AdvancedAgent(config, force_offline=True)

    std_baseline_row = run_agent_benchmark("Baseline Agent", baseline_agent_std, std_convs, config)
    std_advanced_row = run_agent_benchmark("Advanced Agent", advanced_agent_std, std_convs, config)

    print("\n### Standard Benchmark (data/conversations.json)\n")
    print(format_rows([std_baseline_row, std_advanced_row]))

    # Dataset 2: Long-context stress benchmark
    stress_data_path = config.data_dir / "advanced_long_context.json"
    stress_convs = load_conversations(stress_data_path)

    print("\nRunning Long-Context Stress Benchmark...")
    baseline_agent_stress = BaselineAgent(config, force_offline=True)
    advanced_agent_stress = AdvancedAgent(config, force_offline=True)

    stress_baseline_row = run_agent_benchmark("Baseline Agent", baseline_agent_stress, stress_convs, config)
    stress_advanced_row = run_agent_benchmark("Advanced Agent", advanced_agent_stress, stress_convs, config)

    print("\n### Long-Context Stress Benchmark (data/advanced_long_context.json)\n")
    print(format_rows([stress_baseline_row, stress_advanced_row]))


if __name__ == "__main__":
    main()
