from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from benchmark.gaia import gpu_memory
from benchmark.gaia.dataset import GAIADataset
from benchmark.gaia.evaluator import GAIAEvaluator
from core.config import AgentConfig
from core.network import Network
from core.sampling_seed import DEFAULT_SEED
from core.sampling_seed import ENV_VAR as SEED_ENV_VAR
from core.sampling_seed import describe as describe_seed
from score.versa_prm_scorer import (
    DEFAULT_VERSA_PRM_BASE_MODEL_ID,
    DEFAULT_VERSA_PRM_MODEL_ID,
    ensure_versa_prm_model_cache,
)
from tools.tool_manager import ToolManager


DEFAULT_AGENT_SPECS = [
    ("nemotron", "nemotron-3-nano:4b"),
    ("qwen", "qwen3:4b"),
    ("gemma", "gemma3:4b"),
]


def boolean_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false), got {value!r}"
    )


def build_agents(model_specs: str | None = None, *, temperature: float = 0.5) -> list[AgentConfig]:
    specs = DEFAULT_AGENT_SPECS
    if model_specs:
        specs = []
        for index, raw in enumerate(model_specs.split(","), 1):
            model_name = raw.strip()
            if model_name:
                specs.append((f"agent_{index}", model_name))

    return [
        AgentConfig(agent_id=agent_id, model_name=model_name, temperature=temperature)
        for agent_id, model_name in specs
    ]


def build_attachment(sample: dict[str, Any]) -> dict[str, Any]:
    file_path_text = str(sample.get("file_name") or sample.get("file_path") or "").strip()
    if not file_path_text:
        return {}

    file_path = Path(file_path_text)
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "extension": file_path.suffix.lower(),
    }


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_dict(asdict(value))
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, set):
        return [dataclass_to_dict(item) for item in sorted(value, key=str)]
    if isinstance(value, dict):
        converted = {str(key): dataclass_to_dict(item) for key, item in value.items()}
        if "verifier_results" in converted and "judge_results" not in converted:
            converted["judge_results"] = converted["verifier_results"]
        if "verifier_id" in converted and "judge_agent_id" not in converted:
            converted["judge_agent_id"] = converted["verifier_id"]
        if "verifier_score" in converted and "judge_score" not in converted:
            converted["judge_score"] = converted["verifier_score"]
        if "verifier_scores" in converted and "judge_scores" not in converted:
            converted["judge_scores"] = converted["verifier_scores"]
        if "avg_verifier_score" in converted and "avg_judge_score" not in converted:
            converted["avg_judge_score"] = converted["avg_verifier_score"]
        return converted
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def extract_search_summary(network_summary: dict[str, Any]) -> dict[str, Any]:
    metadata = network_summary.get("metadata", {}) or {}
    for usage in metadata.get("tool_usage", []) or []:
        tool_name = str(usage.get("tool_name", "") or "").strip()
        raw_result = usage.get("raw_result")
        if not isinstance(raw_result, dict):
            continue
        if tool_name != "search" and not any(
            key in raw_result
            for key in ("sources", "evidence_items", "blocked_sources", "web_searches")
        ):
            continue
        diagnostics = raw_result.get("diagnostics") or {}
        final_counts = diagnostics.get("final_counts", {}) if isinstance(diagnostics, dict) else {}
        evidence_driven_search = diagnostics.get("evidence_driven_search", {})
        retrieval = raw_result.get("retrieval") or {}
        retrieval_rounds = retrieval.get("rounds", []) if isinstance(retrieval, dict) else []
        page_traces = [
            page
            for round_item in retrieval_rounds
            if isinstance(round_item, dict)
            for page in list(round_item.get("pages") or [])
            if isinstance(page, dict)
        ]
        next_hop_decisions = [
            decision
            for round_item in retrieval_rounds
            if isinstance(round_item, dict)
            for decision in [round_item.get("next_hop_decision")]
            if isinstance(decision, dict) and decision.get("decision_reason")
        ]
        queries = raw_result.get("queries") or []
        if not queries and isinstance(retrieval, dict):
            queries = [
                str(round_item.get("query", "") or "").strip()
                for round_item in retrieval_rounds
                if isinstance(round_item, dict) and str(round_item.get("query", "") or "").strip()
            ]
        evidence_items = raw_result.get("evidence_items") or []
        unverified_references = raw_result.get("unverified_references") or []
        blocked_sources = raw_result.get("blocked_sources") or []
        sources = raw_result.get("sources") or []
        return {
            "initial_web_preprocessing": diagnostics.get("initial_web_preprocessing", {}),
            "source_filter": diagnostics.get("source_filter", {})
            or (diagnostics.get("initial_web_preprocessing", {}) or {}).get("source_filter", {}),
            "full_page_fetch": diagnostics.get("full_page_fetch", {})
            or (diagnostics.get("initial_web_preprocessing", {}) or {}).get("full_page_fetch", {}),
            "coverage_summary": diagnostics.get("coverage_summary", {}),
            "initial_retrieval_decision": diagnostics.get("initial_retrieval_decision", {}),
            "final_retrieval_decision": diagnostics.get("final_retrieval_decision", {}),
            "evidence_driven_search": evidence_driven_search,
            "final_counts": final_counts,
            "pipeline_failure_stage": diagnostics.get("pipeline_failure_stage", ""),
            "queries": queries,
            "evidence_items": evidence_items,
            "unverified_references": unverified_references,
            "retrieval_rounds": retrieval_rounds,
            "page_traces": page_traces,
            "next_hop_decisions": next_hop_decisions,
            "source_count": final_counts.get("source_count", len(sources)),
            "evidence_count": final_counts.get("evidence_count", len(evidence_items)),
            "strict_evidence_count": final_counts.get(
                "strict_evidence_count",
                len(evidence_items),
            ),
            "unverified_reference_count": final_counts.get(
                "unverified_reference_count",
                len(unverified_references),
            ),
            "blocked_source_count": final_counts.get(
                "blocked_source_count",
                len(blocked_sources),
            ),
            "web_search_count": len(raw_result.get("web_searches") or []),
            "stop_reason": evidence_driven_search.get("stop_reason", "")
            or (retrieval.get("stop_reason", "") if isinstance(retrieval, dict) else ""),
        }
    return {}


def safe_filename(value: Any, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or fallback


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    log_name = safe_filename(args.log_name, "gaia_run")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT / "outputs" / log_name / "tasks"

    if args.report_md:
        report_md = Path(args.report_md)
    else:
        report_md = ROOT / "outputs" / log_name / f"{log_name}.md"

    return output_dir, report_md


def run_sample(
    sample: dict[str, Any],
    *,
    evaluator: GAIAEvaluator,
    tool_manager: ToolManager,
    stage1_runs_per_agent: int,
    model_specs: str | None,
    temperature: float,
    max_stage1_workers: int | None,
    stage2_max_tokens: int,
    enable_stage2_score: bool,
    enable_agent_tool_use: bool,
    enable_evidence_prepare: bool,
    enable_compact_search_evidence: bool,
    enable_evidence_driven_search: bool,
    bypass_search_labeler: bool,
    max_parallel_next_hop_queries: int,
    candidate_verification_max_queries: int,
    candidate_verification_workers: int,
    max_agent_tool_turns: int,
    agent_prepared_search_budget: int,
    previous_best_agent_id: str | None,
    stage1_early_stop_max_retries: int,
    stage2_verifier: str,
    versa_prm_model: str,
    versa_prm_base_model: str,
    versa_prm_device: str,
    versa_prm_dtype: str,
) -> dict[str, Any]:
    start_time = time.time()
    network = Network(
        question=str(sample.get("question", "")),
        agents=build_agents(model_specs, temperature=temperature),
        attachment=build_attachment(sample),
        tool_manager=tool_manager,
        stage1_runs_per_agent=stage1_runs_per_agent,
        max_stage1_workers=max_stage1_workers,
        stage2_max_tokens=stage2_max_tokens,
        enable_stage2_score=enable_stage2_score,
        enable_stage1_early_stop=False,
        enable_stage1_tool_use=enable_agent_tool_use,
        enable_evidence_prepare=enable_evidence_prepare,
        enable_compact_search_evidence=enable_compact_search_evidence,
        enable_evidence_driven_search=enable_evidence_driven_search,
        bypass_search_labeler=bypass_search_labeler,
        max_parallel_next_hop_queries=max_parallel_next_hop_queries,
        enable_candidate_verification_search=False,
        candidate_verification_max_queries=candidate_verification_max_queries,
        candidate_verification_workers=candidate_verification_workers,
        max_stage1_tool_turns=max_agent_tool_turns,
        stage1_prepared_search_budget=agent_prepared_search_budget,
        previous_best_agent_id=previous_best_agent_id,
        stage1_early_stop_max_retries=stage1_early_stop_max_retries,
        reference_answer=str(sample.get("final_answer", "") or ""),
        stage2_verifier=stage2_verifier,
        versa_prm_model=versa_prm_model,
        versa_prm_base_model=versa_prm_base_model,
        versa_prm_device=versa_prm_device,
        versa_prm_dtype=versa_prm_dtype,
        versa_prm_local_files_only=True,
    )

    vram_before = gpu_memory.begin_task()
    try:
        summary = network.run()
        predicted = summary.final_answer
        network_summary = dataclass_to_dict(summary)
        error = ""
    except Exception as exc:
        predicted = ""
        network_summary = {}
        error = f"{type(exc).__name__}: {exc}"
    vram_after = gpu_memory.snapshot()

    expected = str(sample.get("final_answer", "") or "")
    exact_match = evaluator._check_exact_match(predicted, expected)
    partial_match = evaluator._check_partial_match(predicted, expected)
    score = 1.0 if exact_match else 0.5 if partial_match else 0.0
    response_time_seconds = (
        (network_summary.get("metadata", {}) or {}).get("response_time_seconds", 0.0)
        if network_summary
        else 0.0
    )
    token_usage = (
        (network_summary.get("metadata", {}) or {}).get("token_usage", {})
        if network_summary
        else {}
    )
    total_tokens = int((token_usage.get("total", {}) or {}).get("total_tokens", 0) or 0)

    return {
        "task_id": sample.get("task_id", ""),
        "level": sample.get("level", ""),
        "question": sample.get("question", ""),
        "attachment": build_attachment(sample),
        "predicted": predicted,
        "expected": expected,
        "exact_match": exact_match,
        "partial_match": partial_match,
        "score": score,
        "winner_agent_id": network_summary.get("winner_agent_id", ""),
        "response_time_seconds": response_time_seconds,
        "total_tokens": total_tokens,
        "token_usage": token_usage,
        "search_summary": extract_search_summary(network_summary),
        "execution_time": time.time() - start_time,
        "error": error,
        # Kept per task so a GPU fault can be traced back to the point the
        # card started filling up, rather than only to the task that died.
        "gpu_memory": {"before": vram_before, "after": vram_after},
        "network_summary": network_summary,
    }


def write_task_json(result: dict[str, Any], output_dir: str | Path, index: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = safe_filename(result.get("task_id"), f"task_{index:03d}")
    output_path = output_dir / f"{index:03d}_{task_id}.json"
    output_path.write_text(json.dumps(dataclass_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_level_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    level_metrics: dict[str, Any] = {}
    levels = sorted({str(result.get("level", "")) for result in results if result.get("level", "")})
    for level in levels:
        level_results = [result for result in results if str(result.get("level", "")) == level]
        exact = sum(1 for result in level_results if result.get("exact_match"))
        partial = sum(1 for result in level_results if result.get("partial_match"))
        level_metrics[f"Level_{level}"] = {
            "total": len(level_results),
            "exact_matches": exact,
            "partial_matches": partial,
            "exact_match_rate": exact / len(level_results) if level_results else 0.0,
            "partial_match_rate": partial / len(level_results) if level_results else 0.0,
        }
    return level_metrics


def build_results(
    *,
    detailed_results: list[dict[str, Any]],
    level: int | None,
    stage1_runs_per_agent: int,
) -> dict[str, Any]:
    total = len(detailed_results)
    exact = sum(1 for result in detailed_results if result.get("exact_match"))
    partial = sum(1 for result in detailed_results if result.get("partial_match"))
    response_times = [
        float(result.get("response_time_seconds", 0.0) or 0.0)
        for result in detailed_results
    ]
    total_tokens = sum(int(result.get("total_tokens", 0) or 0) for result in detailed_results)
    return {
        "benchmark": "GAIA",
        "agent_name": "Network",
        "level_filter": level,
        "stage1_runs_per_agent": stage1_runs_per_agent,
        # Recorded so an output folder says whether it is comparable to another.
        "stage1_seed": describe_seed(),
        "total_samples": total,
        "exact_matches": exact,
        "partial_matches": partial,
        "exact_match_rate": exact / total if total else 0.0,
        "partial_match_rate": partial / total if total else 0.0,
        "average_response_time_seconds": (
            sum(response_times) / len(response_times) if response_times else 0.0
        ),
        "total_tokens": total_tokens,
        "average_tokens_per_task": total_tokens / total if total else 0.0,
        "level_metrics": build_level_metrics(detailed_results),
        "detailed_results": detailed_results,
    }


def short_cell(value: Any, limit: int = 80) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def fenced_text(value: Any, limit: int = 3000) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n..."
    return text


def format_step_scores(step_scores: Any) -> str:
    if not isinstance(step_scores, list) or not step_scores:
        return "-"

    parts = []
    for index, item in enumerate(step_scores, 1):
        if isinstance(item, dict):
            step = item.get("step", index)
            score = item.get(
                "reward_probability",
                item.get("score", item.get("verifier_score", item.get("judge_score", 0))),
            )
        else:
            step = index
            score = item
        support = ""
        if isinstance(item, dict):
            support_status = str(item.get("support_status") or "").strip()
            if support_status:
                support = f", support={support_status}"
        parts.append(f"step {step}: {score}{support}")
    return ", ".join(parts)


def format_evidence_support(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    support = metadata.get("evidence_support")
    if not isinstance(support, dict):
        return "-"
    status = str(support.get("status") or "no_support")
    priority = support.get("priority", 0)
    supported_steps = (support.get("metadata") or {}).get("supported_step_count", 0)
    return f"{status} (priority={priority}, supported_steps={supported_steps})"


def format_verifier_score(item: dict[str, Any]) -> str:
    step_text = format_step_scores(item.get("step_scores"))
    verifier_score = item.get("verifier_score", item.get("judge_score", 0))
    if step_text == "-":
        return str(verifier_score)
    return f"{verifier_score} ({step_text})"


def format_verifier_pairs(network_summary: dict[str, Any]) -> str:
    pairs = []
    for item in network_summary.get("verifier_results", network_summary.get("judge_results", [])) or []:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        run_index = metadata.get("target_run_index", 0)
        run_suffix = f"/run-{run_index}" if run_index else ""
        pairs.append(
            f"{item.get('verifier_id', item.get('judge_agent_id', ''))}->{item.get('target_agent_id', '')}{run_suffix}: "
            f"{format_verifier_score(item)}"
        )
    return "; ".join(pairs) if pairs else "No Stage2 verifier scores"


def write_markdown_report(results: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_results = results.get("detailed_results", [])

    lines = [
        "# GAIA Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Benchmark: {results.get('benchmark', 'GAIA')}",
        f"- Agent: {results.get('agent_name', 'Network')}",
        f"- Level filter: {results.get('level_filter') or 'all'}",
        f"- Stage1 sampling seed: {results.get('stage1_seed', 'unknown')}",
        f"- Stage1 runs per agent: {results.get('stage1_runs_per_agent')}",
        f"- Total samples: {results.get('total_samples', 0)}",
        f"- Exact matches: {results.get('exact_matches', 0)}",
        f"- Partial matches: {results.get('partial_matches', 0)}",
        f"- Exact match rate: {results.get('exact_match_rate', 0.0):.2%}",
        f"- Partial match rate: {results.get('partial_match_rate', 0.0):.2%}",
        f"- Average response time: {results.get('average_response_time_seconds', 0.0):.2f}s",
        f"- Total tokens: {results.get('total_tokens', 0)}",
        f"- Average tokens per task: {results.get('average_tokens_per_task', 0.0):.2f}",
        "",
        "## Level Metrics",
        "",
        "| Level | Total | Exact | Partial | Exact Rate | Partial Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    level_metrics = results.get("level_metrics", {})
    if level_metrics:
        for level_name, metrics in level_metrics.items():
            lines.append(
                f"| {level_name} | {metrics.get('total', 0)} | "
                f"{metrics.get('exact_matches', 0)} | {metrics.get('partial_matches', 0)} | "
                f"{metrics.get('exact_match_rate', 0.0):.2%} | "
                f"{metrics.get('partial_match_rate', 0.0):.2%} |"
            )
    else:
        lines.append("| - | 0 | 0 | 0 | 0.00% | 0.00% |")

    lines.extend(
        [
            "",
            "## Tasks",
            "",
            "| # | Task ID | Level | Exact | Partial | Winner | Predicted | Expected | JSON |",
            "|---:|---|---:|---|---|---|---|---|---|",
        ]
    )

    for index, result in enumerate(detailed_results, 1):
        predicted = str(result.get("predicted", "")).replace("|", "\\|")
        expected = str(result.get("expected", "")).replace("|", "\\|")
        if len(predicted) > 80:
            predicted = predicted[:77] + "..."
        if len(expected) > 80:
            expected = expected[:77] + "..."
        lines.append(
            f"| {index} | {result.get('task_id', '')} | {result.get('level', '')} | "
            f"{result.get('exact_match', False)} | {result.get('partial_match', False)} | "
            f"{result.get('winner_agent_id', '')} | {predicted} | {expected} | "
            f"{result.get('task_json_path', '')} |"
        )

    lines.extend(["", "## Task Details", ""])
    for index, result in enumerate(detailed_results, 1):
        lines.extend(
            [
                f"### {index}. {result.get('task_id', '')}",
                "",
                f"- Level: {result.get('level', '')}",
                f"- Exact match: {result.get('exact_match', False)}",
                f"- Partial match: {result.get('partial_match', False)}",
                f"- Score: {result.get('score', 0.0)}",
                f"- Winner agent: {result.get('winner_agent_id', '')}",
                f"- Response time: {result.get('response_time_seconds', 0.0):.2f}s",
                f"- Total tokens: {result.get('total_tokens', 0)}",
                f"- Execution time: {result.get('execution_time', 0.0):.2f}s",
                f"- JSON: {result.get('task_json_path', '')}",
                "",
                "**Question**",
                "",
                str(result.get("question", "")),
                "",
                "**Predicted**",
                "",
                str(result.get("predicted", "")),
                "",
                "**Expected**",
                "",
                str(result.get("expected", "")),
                "",
            ]
        )
        if result.get("error"):
            lines.extend(["**Error**", "", str(result["error"]), ""])

        network_summary = result.get("network_summary", {}) or {}
        network_metadata = network_summary.get("metadata", {}) or {}
        stage1_results = network_summary.get("stage1_results", []) or []
        verifier_results = network_summary.get("verifier_results", network_summary.get("judge_results", [])) or []
        agent_scores = network_summary.get("agent_scores", []) or []
        token_usage = result.get("token_usage", {}) or (
            network_metadata
        ).get("token_usage", {}) or {}
        search_summary = result.get("search_summary", {}) or {}
        winner_selection = network_metadata.get("winner_selection", {}) or {}
        winner_trace = winner_selection.get("selection_trace", {}) or {}

        lines.extend(
            [
                "**Network Metadata**",
                "",
                f"- Stage1 early stop enabled: {network_metadata.get('enable_stage1_early_stop', False)}",
                f"- Stage2 score enabled: {network_metadata.get('enable_stage2_score', True)}",
                f"- Stage1 tool use enabled: {network_metadata.get('enable_stage1_tool_use', False)}",
                f"- Evidence prepare enabled: {network_metadata.get('enable_evidence_prepare', True)}",
                f"- Compact search evidence enabled: {network_metadata.get('enable_compact_search_evidence', False)}",
                f"- Query planner: {network_metadata.get('query_planner', 'signal')}",
                f"- Evidence-driven search enabled: {network_metadata.get('enable_evidence_driven_search', False)}",
                f"- Max Stage1 workers: {network_metadata.get('max_stage1_workers', 0)}",
                f"- Max parallel next-hop queries: {network_metadata.get('max_parallel_next_hop_queries', 0)}",
                f"- Max Stage1 tool turns: {network_metadata.get('max_stage1_tool_turns', 0)}",
                f"- Stage1 prepared-search budget: {network_metadata.get('stage1_prepared_search_budget', 1)}",
                f"- Stage1 early stop used: {network_metadata.get('stage1_early_stop', False)}",
                f"- Stage1 attempts: {network_metadata.get('stage1_attempts', 0)}",
                f"- Stage1 early stop max retries: {network_metadata.get('stage1_early_stop_max_retries', 0)}",
                f"- Previous best agent: {network_metadata.get('previous_best_agent_id', '') or '-'}",
                f"- Stage2 skipped: {network_metadata.get('stage2_skipped', False)}",
                f"- Stage2 skip reason: {network_metadata.get('stage2_skip_reason', '') or '-'}",
                f"- Early stop reason: {network_metadata.get('stage1_early_stop_reason', '') or '-'}",
                "",
            ]
        )

        if token_usage:
            lines.extend(
                [
                    "**Token Usage**",
                    "",
                    "| Stage | Prompt | Completion | Total |",
                    "|---|---:|---:|---:|",
                ]
            )
            for stage_name in ("stage1", "stage2", "total"):
                usage = token_usage.get(stage_name, {}) or {}
                lines.append(
                    f"| {stage_name} | {usage.get('prompt_tokens', 0)} | "
                    f"{usage.get('completion_tokens', 0)} | {usage.get('total_tokens', 0)} |"
                )
            lines.append("")

        if winner_trace:
            lines.extend(
                [
                    "**Ordered-Gate Winner Selection**",
                    "",
                    f"- Status: {winner_trace.get('status', '')}",
                    f"- Reason: {winner_trace.get('selection_reason', '')}",
                    f"- Selected answer: {winner_trace.get('selected_answer', '') or '-'}",
                    f"- Selected path: {winner_trace.get('selected_agent_id', '') or '-'} / run {winner_trace.get('selected_run_index', 0) or '-'}",
                    "",
                    "| Candidate | Evidence Status | Requirement | Direct Support | Agents | Runs | Critical Floor | Critical Geomean | Eligible |",
                    "|---|---|---|---|---:|---:|---:|---:|---|",
                ]
            )
            for candidate in winner_trace.get("candidates", []) or []:
                lines.append(
                    f"| {short_cell(candidate.get('answer', ''), 100)} | "
                    f"{candidate.get('support_status', 'no_support')} | "
                    f"{candidate.get('requirement_status', 'not_available')} | "
                    f"{candidate.get('direct_support', False)} | "
                    f"{len(candidate.get('supporting_agent_ids', []) or [])} | "
                    f"{candidate.get('supporting_run_count', 0)} | "
                    f"{candidate.get('critical_step_floor', 0)} | "
                    f"{candidate.get('critical_step_geometric_mean', 0)} | "
                    f"{candidate.get('eligible', False)} |"
                )
            lines.append("")
            gate_trace = winner_trace.get("gate_trace", []) or []
            if gate_trace:
                lines.extend(
                    [
                        "| Ordered Gate | Survivors | Eliminated | Terminal Status |",
                        "|---|---|---|---|",
                    ]
                )
                for gate in gate_trace:
                    eliminated = [
                        f"{item.get('candidate_key', '')}: {item.get('reason', '')}"
                        for item in gate.get("eliminated", []) or []
                    ]
                    lines.append(
                        f"| {gate.get('gate_name', '')} | "
                        f"{short_cell(', '.join(gate.get('survivors', []) or []) or '-', 120)} | "
                        f"{short_cell('; '.join(eliminated) or '-', 180)} | "
                        f"{gate.get('terminal_status', '') or '-'} |"
                    )
                lines.append("")

        context_budget = network_metadata.get("stage1_context_budget", {}) or {}
        if context_budget:
            lines.extend(
                [
                    "**Stage1 Context Budget**",
                    "",
                    f"- Runs tracked: {context_budget.get('run_count', 0)}",
                    f"- Average original chars: {context_budget.get('original_chars_avg', 0)}",
                    f"- Average final chars: {context_budget.get('final_chars_avg', 0)}",
                    f"- Total char reduction: {context_budget.get('chars_reduction_total', 0)}",
                    f"- Truncation applied count: {context_budget.get('truncation_applied_count', 0)}",
                    f"- Dropped evidence count: {context_budget.get('dropped_evidence_count', 0)}",
                    f"- Truncated sections: {context_budget.get('truncated_sections', []) or '-'}",
                    "",
                ]
            )

        search_gate = network_metadata.get("stage1_search_gate", {}) or {}
        if search_gate:
            lines.extend(
                [
                    "**Stage1 Search Gate**",
                    "",
                    f"- Prepared status: {search_gate.get('prepared_status', '-')}",
                    f"- Prepared evidence available: {search_gate.get('prepared_evidence_available', False)}",
                    f"- Search requests: {search_gate.get('request_count', 0)}",
                    f"- Physical search executions: {search_gate.get('physical_execution_count', 0)}",
                    f"- Shared cache hits: {search_gate.get('cache_hit_count', 0)}",
                    f"- Blocked requests: {search_gate.get('blocked_count', 0)}",
                    f"- Blocked reasons: {search_gate.get('blocked_reasons', {}) or '-'}",
                    f"- Refinement used/budget: {search_gate.get('refinement_used', 0)}/{search_gate.get('refinement_budget', 0)}",
                    f"- Supplemental evidence count: {search_gate.get('supplemental_evidence_count', 0)}",
                    "",
                ]
            )

        attachment_reuse = network_metadata.get("stage1_attachment_reuse", {}) or {}
        if attachment_reuse:
            lines.extend(
                [
                    "**Stage1 Attachment Reuse**",
                    "",
                    f"- Prepared attachment available: {attachment_reuse.get('prepared_available', False)}",
                    f"- Reader executions: {attachment_reuse.get('reader_execution_count', 0)}",
                    f"- Vision/audio executions: {attachment_reuse.get('vision_execution_count', 0)}",
                    f"- Stage1 attachment requests: {attachment_reuse.get('stage1_request_count', 0)}",
                    f"- Typed payload reuse count: {attachment_reuse.get('payload_reuse_count', 0)}",
                    f"- Result cache hits: {attachment_reuse.get('result_cache_hit_count', 0)}",
                    f"- Handler executions: {attachment_reuse.get('handler_execution_count', 0)}",
                    f"- Blocked duplicate requests: {attachment_reuse.get('blocked_duplicate_count', 0)}",
                    f"- Selected handlers: {attachment_reuse.get('selected_handlers', []) or '-'}",
                    "",
                ]
            )

        if search_summary:
            search_queries = search_summary.get("queries", []) or []
            evidence_items = search_summary.get("evidence_items", []) or []
            unverified_references = (
                search_summary.get("unverified_references", []) or []
            )
            retrieval_rounds = search_summary.get("retrieval_rounds", []) or []
            source_filter = search_summary.get("source_filter", {}) or {}
            full_page_fetch = search_summary.get("full_page_fetch", {}) or {}
            coverage_summary = search_summary.get("coverage_summary", {}) or {}
            page_traces = search_summary.get("page_traces", []) or []
            next_hop_decisions = search_summary.get("next_hop_decisions", []) or []
            lines.extend(
                [
                    "**Search Summary**",
                    "",
                    f"- Source count: {search_summary.get('source_count', 0)}",
                    f"- Evidence count: {search_summary.get('evidence_count', 0)}",
                    f"- Strict evidence count: {search_summary.get('strict_evidence_count', 0)}",
                    f"- Unverified reference count: {search_summary.get('unverified_reference_count', 0)}",
                    f"- Blocked source count: {search_summary.get('blocked_source_count', 0)}",
                    f"- Web search count: {search_summary.get('web_search_count', 0)}",
                    f"- Query count: {len(search_queries)}",
                    f"- Retrieval rounds: {len(retrieval_rounds)}",
                    f"- Selected pages: {len(page_traces)}",
                    f"- Required next-hop decisions: "
                    f"{sum(1 for item in next_hop_decisions if item.get('required'))}",
                    f"- Evidence-driven follow-up: {search_summary.get('evidence_driven_search', {}).get('queries', []) or '-'}",
                    f"- Retrieval stop reason: {search_summary.get('stop_reason', '') or '-'}",
                    f"- Pipeline failure stage: {search_summary.get('pipeline_failure_stage', '') or '-'}",
                    f"- Coverage score: {coverage_summary.get('final_score', '-')}",
                    f"- Coverage sufficient: {coverage_summary.get('final_sufficient', '-')}",
                    f"- Coverage answer type: {coverage_summary.get('answer_type', '-')}",
                    f"- Coverage answer type covered: {coverage_summary.get('answer_type_covered', '-')}",
                    f"- Coverage next-hop trigger: {coverage_summary.get('next_hop_triggered_by', '') or '-'}",
                    f"- Rescued soft-block sources: {source_filter.get('rescued_soft_block_count', 0)}",
                    f"- Fetch attempted/fetched/failed: {full_page_fetch.get('attempted_fetch_count', 0)} / "
                    f"{full_page_fetch.get('fetched_page_count', 0)} / {full_page_fetch.get('fetch_failure_count', 0)}",
                    "",
                ]
                )
            if retrieval_rounds:
                lines.extend(["Page and next-hop trace:", ""])
                for round_item in retrieval_rounds:
                    if not isinstance(round_item, dict):
                        continue
                    round_index = round_item.get("round_index", "-")
                    pages = [
                        page
                        for page in list(round_item.get("pages") or [])
                        if isinstance(page, dict)
                    ]
                    decision = round_item.get("next_hop_decision") or {}
                    lines.append(f"- Round {round_index}:")
                    for page in pages[:4]:
                        lines.append(
                            "  - "
                            + short_cell(page.get("title") or page.get("url") or page.get("page_id"), 100)
                            + f" [{page.get('status', 'no_usable_contract')}]"
                        )
                    if isinstance(decision, dict) and decision.get("decision_reason"):
                        lines.append(
                            "  - Next-hop: "
                            f"{'yes' if decision.get('required') else 'no'}; "
                            f"reason={decision.get('decision_reason')}; "
                            f"query={short_cell(decision.get('generated_query') or '-', 140)}"
                        )
                lines.append("")
            if coverage_summary.get("missing_constraints") or coverage_summary.get("bridge_terms"):
                lines.extend(
                    [
                        f"Coverage missing constraints: {coverage_summary.get('missing_constraints') or '-'}",
                        f"Coverage bridge terms: {coverage_summary.get('bridge_terms') or '-'}",
                        "",
                    ]
                )
            if source_filter.get("block_reason_counts"):
                block_reasons = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(source_filter.get("block_reason_counts", {}).items())
                )
                lines.extend([f"Blocked source reasons: {block_reasons}", ""])
            if full_page_fetch.get("quality_counts") or full_page_fetch.get("method_counts"):
                quality = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(full_page_fetch.get("quality_counts", {}).items())
                )
                methods = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(full_page_fetch.get("method_counts", {}).items())
                )
                lines.extend(
                    [
                        f"Fetch quality counts: {quality or '-'}",
                        f"Fetch method counts: {methods or '-'}",
                        "",
                    ]
                )
            if search_queries:
                lines.extend(["Search queries:", ""])
                for query in search_queries[:8]:
                    lines.append(f"- {short_cell(query, 180)}")
                lines.append("")
            if evidence_items:
                lines.extend(
                    [
                        "Evidence items:",
                        "",
                        "| ID | Source Title | Evidence |",
                        "|---|---|---|",
                    ]
                )
                for item in evidence_items[:8]:
                    if not isinstance(item, dict):
                        continue
                    evidence_id = short_cell(item.get("evidence_id", "") or "-", 24).replace("|", "\\|")
                    title = short_cell(item.get("title", ""), 90).replace("|", "\\|")
                    evidence_text = short_cell(item.get("text", ""), 220).replace("|", "\\|")
                    lines.append(
                        f"| {evidence_id} | "
                        f"{title} | "
                        f"{evidence_text} |"
                    )
                lines.append("")
            if unverified_references:
                lines.extend(
                    [
                        "Unverified references:",
                        "",
                        "| ID | Source Title | Reference Content |",
                        "|---|---|---|",
                    ]
                )
                for item in unverified_references[:8]:
                    if not isinstance(item, dict):
                        continue
                    reference_id = short_cell(
                        item.get("reference_id", "") or "-",
                        24,
                    ).replace("|", "\\|")
                    title = short_cell(item.get("title", ""), 90).replace(
                        "|", "\\|"
                    )
                    reference_text = short_cell(
                        item.get("text", ""),
                        220,
                    ).replace("|", "\\|")
                    lines.append(
                        f"| {reference_id} | {title} | {reference_text} |"
                    )
                lines.append("")

        lines.extend(
            [
                "**Agent Scores**",
                "",
                "| Agent | Model | Active | Confidence | Verifier Scores | Avg Verifier | Penalty | Penalty Reasons | Total |",
                "|---|---|---|---:|---|---:|---:|---|---:|",
            ]
        )
        if agent_scores:
            active_by_agent = {item.get("agent_id"): item.get("active") for item in stage1_results}
            for score in agent_scores:
                verifier_scores = ", ".join(
                    str(value)
                    for value in score.get("verifier_scores", score.get("judge_scores", [])) or []
                )
                penalty_reasons = ", ".join(str(value) for value in score.get("penalty_reasons", []) or [])
                lines.append(
                    f"| {score.get('agent_id', '')} | {score.get('model_name', '')} | "
                    f"{active_by_agent.get(score.get('agent_id'), False)} | "
                    f"{score.get('confidence_score', 0)} | {verifier_scores or '-'} | "
                    f"{score.get('avg_verifier_score', score.get('avg_judge_score', 0))} | "
                    f"{score.get('penalty_score', 0)} | "
                    f"{short_cell(penalty_reasons, 120) or '-'} | {score.get('total_score', 0)} |"
                )
        else:
            lines.append("| - | - | - | 0 | - | 0 | 0 | - | 0 |")

        lines.extend(
            [
                "",
                "**Stage1 Reasoning Summaries**",
                "",
                "| Agent | Model | Active | Confidence | Compressed Answer |",
                "|---|---|---|---:|---|",
            ]
        )
        if stage1_results:
            for item in stage1_results:
                lines.append(
                    f"| {item.get('agent_id', '')} | {item.get('model_name', '')} | "
                    f"{item.get('active', False)} | {item.get('confidence_score', 0)} | "
                    f"{short_cell(item.get('compressed_answer', ''), 120)} |"
                )
        else:
            lines.append("| - | - | - | 0 | - |")

        if stage1_results:
            lines.extend(
                [
                    "",
                    "**Stage1 Answer Aggregation / Self-Review**",
                    "",
                    "| Agent | Aggregation Status | Needs Review | Review Applied | Tool Used | Review Answer | Review Error |",
                    "|---|---|---|---|---|---|---|",
                ]
            )
            for item in stage1_results:
                aggregation = item.get("aggregation_metadata", {}) or {}
                review = item.get("self_review_metadata", {}) or {}
                lines.append(
                    f"| {item.get('agent_id', '')} | "
                    f"{aggregation.get('status', '-') or '-'} | "
                    f"{aggregation.get('needs_review', False)} | "
                    f"{review.get('applied', False)} | "
                    f"{review.get('tool_used', False)} | "
                    f"{short_cell(review.get('answer', '') or '-', 100)} | "
                    f"{short_cell(review.get('error', '') or '-', 120)} |"
                )

        lines.extend(
            [
                "",
                "**Stage2 Verifier Scores**",
                "",
                "| Verifier | Target Agent | Run | Candidate | Avg Score | Critical Floor | Critical Geomean | Evidence Support | Step Scores |",
                "|---|---|---:|---|---:|---:|---:|---|---|",
            ]
        )
        if verifier_results:
            for item in verifier_results:
                verifier_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                process = verifier_metadata.get("process_verification") if isinstance(verifier_metadata.get("process_verification"), dict) else {}
                lines.append(
                    f"| {item.get('verifier_id', item.get('judge_agent_id', ''))} | "
                    f"{item.get('target_agent_id', '')} | "
                    f"{verifier_metadata.get('target_run_index', 0) or '-'} | "
                    f"{short_cell(verifier_metadata.get('target_answer', ''), 80)} | "
                    f"{item.get('verifier_score', item.get('judge_score', 0))} | "
                    f"{process.get('critical_step_floor', 0)} | "
                    f"{process.get('critical_step_geometric_mean', 0)} | "
                    f"{short_cell(format_evidence_support(item), 160)} | "
                    f"{short_cell(format_step_scores(item.get('step_scores')), 240)} |"
                )
        else:
            lines.append("| - | - | - | - | 0 | 0 | 0 | - | - |")

        if stage1_results:
            lines.extend(["", "**Stage1 Tool Summary**", ""])
            lines.extend(["| Agent | Run | Tool Calls | Cache Hits |", "|---|---:|---|---:|"])
            for item in stage1_results:
                for run in item.get("runs", []) or []:
                    tool_calls = run.get("tool_calls", []) or []
                    tool_results = run.get("tool_results", []) or []
                    tool_names = ", ".join(
                        str(call.get("tool_name", ""))
                        for call in tool_calls
                    )
                    cache_hits = sum(1 for result in tool_results if result.get("cache_hit"))
                    lines.append(
                        f"| {item.get('agent_id', '')} | {run.get('run_index', '')} | "
                        f"{short_cell(tool_names or '-', 100)} | {cache_hits} |"
                    )

            lines.extend(["", "**Reasoning Text And Received Scores**", ""])
            scores_by_target: dict[str, list[dict[str, Any]]] = {}
            for verifier in verifier_results:
                scores_by_target.setdefault(verifier.get("target_agent_id", ""), []).append(verifier)
            for item in stage1_results:
                target_id = item.get("agent_id", "")
                received_scores = scores_by_target.get(target_id, [])
                score_text = (
                    ", ".join(
                        f"{score.get('verifier_id', score.get('judge_agent_id', ''))}: "
                        f"run {(score.get('metadata') or {}).get('target_run_index', '-')}: "
                        f"{format_verifier_score(score)}"
                        for score in received_scores
                    )
                    or "No verifier score"
                )
                lines.extend(
                    [
                        f"#### Target Reason: {target_id}",
                        "",
                        f"- Model: {item.get('model_name', '')}",
                        f"- Active: {item.get('active', False)}",
                        f"- Confidence: {item.get('confidence_score', 0)}",
                        f"- Received verifier scores: {score_text}",
                        "",
                        "Final answer:",
                        "",
                        f"`{item.get('compressed_answer', '')}`",
                        "",
                        "Reasoning:",
                        "",
                        "```text",
                        fenced_text(item.get("compressed_reasoning", "")),
                        "```",
                        "",
                    ]
                )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_gaia_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    if args.stage1_seed is not None:
        os.environ[SEED_ENV_VAR] = str(args.stage1_seed)
    # Read and report it up front so a bad value fails here rather than hours
    # in, and so the log says how the run was sampled before it starts.
    seed_description = describe_seed()
    print(f"[INFO] stage1_seed={seed_description}")

    data_dir = Path(args.data_dir).resolve()
    dataset = GAIADataset(split=args.split, level=args.level, local_data_dir=data_dir)
    items = dataset.load()
    if args.task_ids:
        # Re-running a handful of named tasks is how a targeted change gets
        # checked without paying for the whole split. Prefixes are accepted so
        # the ids from an output folder can be pasted in directly.
        wanted = [value.strip() for value in args.task_ids.split(",") if value.strip()]
        items = [
            item
            for item in items
            if any(str(item.get("task_id", "")).startswith(prefix) for prefix in wanted)
        ]
        missing = [
            prefix
            for prefix in wanted
            if not any(str(item.get("task_id", "")).startswith(prefix) for item in items)
        ]
        if missing:
            raise SystemExit(f"--task-ids matched nothing for: {', '.join(missing)}")
    elif args.max_samples is not None:
        items = items[: max(0, args.max_samples)]

    evaluator = GAIAEvaluator(dataset=dataset, level=args.level)
    tool_manager = ToolManager()
    detailed_results: list[dict[str, Any]] = []
    output_dir, report_md = resolve_output_paths(args)

    print(f"[INFO] GAIA data_dir={data_dir}")
    print(f"[INFO] split={args.split} level={args.level or 'all'} samples={len(items)}")
    # print(f"[INFO] stage1_runs_per_agent={args.stage1_runs_per_agent}")
    # print(f"[INFO] max_stage1_workers={args.max_stage1_workers or 'auto'}")
    # print(f"[INFO] stage2_max_tokens={args.stage2_max_tokens}")
    # print(f"[INFO] enable_stage2_score={not args.without_stage2_score}")
    # print(f"[INFO] stage2_verifier={args.stage2_verifier}")
    # print(f"[INFO] versa_prm_model={args.versa_prm_model}")
    # print(f"[INFO] versa_prm_base_model={args.versa_prm_base_model}")
    # print(f"[INFO] versa_prm_device={args.versa_prm_device}")
    # print(f"[INFO] versa_prm_dtype={args.versa_prm_dtype}")
    # print(f"[INFO] enable_agent_tool_use={args.enable_agent_tool_use}")
    # print(f"[INFO] evidence_prepare={args.evidence_prepare}")
    # print(f"[INFO] compact_search_evidence={args.compact_search_evidence}")
    # print("[INFO] query_planner=signal")
    # print(f"[INFO] enable_evidence_driven_search={args.enable_evidence_driven_search}")
    # print(f"[INFO] max_parallel_next_hop_queries={args.max_parallel_next_hop_queries}")
    # print(f"[INFO] max_agent_tool_turns={args.max_agent_tool_turns}")
    # print(f"[INFO] stage1_early_stop_max_retries={args.stage1_early_stop_max_retries}")
    print(f"[INFO] log_name={safe_filename(args.log_name, 'gaia_run')}")
    print(f"[INFO] task_json_dir={output_dir.resolve()}")
    print(f"[INFO] report_md={report_md.resolve()}")

    versa_cache_warmup: dict[str, Any] = {}
    if not args.without_stage2_score and not args.skip_versa_prm_cache_warmup:
        print("[INFO] warming VersaPRM HuggingFace cache")
        versa_cache_warmup = ensure_versa_prm_model_cache(
            model_id=args.versa_prm_model,
            base_model_id=args.versa_prm_base_model,
            allow_download=False,
        )
        model_ok = bool((versa_cache_warmup.get("model") or {}).get("ok"))
        base_ok = bool((versa_cache_warmup.get("base_model") or {}).get("ok"))
        print(f"[INFO] versa_prm_cache_warmup model_ok={model_ok} base_model_ok={base_ok}")
        if not model_ok or not base_ok:
            raise RuntimeError(f"VersaPRM cache warm-up failed: {versa_cache_warmup}")
    previous_best_agent_id: str | None = None
    agent_exact_counts: dict[str, int] = {}
    for index, sample in enumerate(items, 1):
        print(f"\n========== GAIA {index}/{len(items)} task_id={sample.get('task_id', '')} ==========")
        result = run_sample(
            sample,
            evaluator=evaluator,
            tool_manager=tool_manager,
            stage1_runs_per_agent=args.stage1_runs_per_agent,
            model_specs=args.models,
            temperature=args.temperature,
            max_stage1_workers=args.max_stage1_workers,
            stage2_max_tokens=args.stage2_max_tokens,
            enable_stage2_score=not args.without_stage2_score,
            enable_agent_tool_use=args.enable_agent_tool_use,
            enable_evidence_prepare=args.evidence_prepare,
            enable_compact_search_evidence=args.compact_search_evidence,
            enable_evidence_driven_search=args.enable_evidence_driven_search,
            bypass_search_labeler=args.bypass_search_labeler,
            max_parallel_next_hop_queries=args.max_parallel_next_hop_queries,
            candidate_verification_max_queries=args.candidate_verification_max_queries,
            candidate_verification_workers=args.candidate_verification_workers,
            max_agent_tool_turns=args.max_agent_tool_turns,
            agent_prepared_search_budget=args.agent_prepared_search_budget,
            previous_best_agent_id=previous_best_agent_id,
            stage1_early_stop_max_retries=args.stage1_early_stop_max_retries,
            stage2_verifier=args.stage2_verifier,
            versa_prm_model=args.versa_prm_model,
            versa_prm_base_model=args.versa_prm_base_model,
            versa_prm_device=args.versa_prm_device,
            versa_prm_dtype=args.versa_prm_dtype,
        )
        if result.get("exact_match") and result.get("winner_agent_id"):
            winner_id = str(result.get("winner_agent_id") or "")
            agent_exact_counts[winner_id] = agent_exact_counts.get(winner_id, 0) + 1
            previous_best_agent_id = max(
                agent_exact_counts,
                key=lambda agent_id: (agent_exact_counts[agent_id], agent_id),
            )
        task_json_path = write_task_json(result, output_dir, index)
        result["task_json_path"] = str(task_json_path)
        detailed_results.append(result)

        print(
            f"exact={result['exact_match']} partial={result['partial_match']} "
            f"winner={result['winner_agent_id']} predicted={result['predicted']!r} "
            f"json={task_json_path}"
        )
        vram_line = gpu_memory.summarize(
            (result.get("gpu_memory") or {}).get("before") or {},
            (result.get("gpu_memory") or {}).get("after") or {},
        )
        if vram_line:
            print(vram_line)
        print(f"verifier_scores={format_verifier_pairs(result.get('network_summary', {}) or {})}")
        if result["error"]:
            print(f"[WARN] {result['error']}")

    results = build_results(
        detailed_results=detailed_results,
        level=args.level,
        stage1_runs_per_agent=args.stage1_runs_per_agent,
    )
    results["versa_prm_cache_warmup"] = versa_cache_warmup

    report_path = write_markdown_report(results, report_md)
    print(f"[OK] GAIA Markdown report exported: {report_path}")

    print("\n[OK] GAIA batch finished")
    print(f"   total={results['total_samples']}")
    print(f"   exact_match_rate={results['exact_match_rate']:.2%}")
    print(f"   partial_match_rate={results['partial_match_rate']:.2%}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIA samples with the current Network implementation.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "gaia"))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated GAIA task ids (prefixes allowed). Overrides --max-samples.",
    )
    parser.add_argument(
        "--agent-seed",
        default=None,
        help=(
            "Base seed for Stage1 sampling, so a rerun of unchanged code is "
            f"comparable. Sets {SEED_ENV_VAR}. Use 'off' for free sampling. "
            f"Omit to use the environment, which defaults to {DEFAULT_SEED}."
        ),
    )
    parser.add_argument("--stage1-runs-per-agent", type=int, default=3)
    parser.add_argument(
        "--max-stage1-workers",
        type=int,
        default=None,
        help="Maximum number of Stage1 agent runs executed in parallel. Omit for automatic.",
    )
    parser.add_argument("--stage2-max-tokens", type=int, default=512)
    parser.add_argument(
        "--stage2-verifier",
        choices=["versa"],
        default=os.getenv("STAGE2_VERIFIER", "versa"),
        help="Stage2 verifier backend. SCP currently supports VersaPRM only.",
    )
    parser.add_argument(
        "--versa-prm-model",
        default=os.getenv("VERSA_PRM_MODEL", DEFAULT_VERSA_PRM_MODEL_ID),
        help="HuggingFace model name or local path for VersaPRM.",
    )
    parser.add_argument(
        "--versa-prm-base-model",
        default=os.getenv("VERSA_PRM_BASE_MODEL", DEFAULT_VERSA_PRM_BASE_MODEL_ID),
        help="Base model id used by VersaPRM PEFT fallback.",
    )
    parser.add_argument(
        "--versa-prm-device",
        default=os.getenv("VERSA_PRM_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu"],
        help="VersaPRM torch device.",
    )
    parser.add_argument(
        "--versa-prm-dtype",
        default=os.getenv("VERSA_PRM_DTYPE", "auto"),
        choices=["auto", "float16", "bfloat16", "float32"],
        help="VersaPRM torch dtype.",
    )
    parser.add_argument(
        "--skip-versa-prm-cache-warmup",
        action="store_true",
        help="Skip startup Hugging Face cache warm-up for VersaPRM.",
    )
    parser.add_argument(
        "--without-stage2-score",
        dest="without_stage2_score",
        action="store_true",
        help="Skip Stage2 judge scoring and rank agents by Stage1 confidence plus penalties.",
    )
    parser.add_argument(
        "--enable-agent-tool-use",
        type=boolean_arg,
        const=True,
        metavar="{true,false}",
        default=True,
        help=(
            "Let agents call tools. On by default: every recorded benchmark run "
            "has it enabled (159 of 159 task runs across level1_final_13, _15 and "
            "_16), so a default of off meant that omitting the flag silently ran a "
            "different system. Disable with --no-enable-agent-tool-use."
        ),
    )
    parser.add_argument(
        "--evidence-prepare",
        type=boolean_arg,
        nargs="?",
        const=True,
        metavar="{true,false}",
        default=True,
        help=(
            "Prepare attachment, deterministic, and search evidence. "
            "Enabled by default; pass false to disable."
        ),
    )
    parser.add_argument("--compact-search-evidence", action="store_true")
    parser.add_argument("--enable-evidence-driven-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bypass-search-labeler",
        action="store_true",
        help=(
            "Skip EfficientRAG Labeler and build sentence-level evidence units "
            "for SpanRoleClassifier directly."
        ),
    )
    parser.add_argument(
        "--max-parallel-next-hop-queries",
        type=int,
        default=2,
        help="Maximum number of EfficientRAG filter queries searched in parallel.",
    )
    parser.add_argument(
        "--candidate-verification-max-queries",
        type=int,
        default=5,
        help="Maximum number of distinct candidate answers verified by recovery search.",
    )
    parser.add_argument(
        "--candidate-verification-workers",
        type=int,
        default=2,
        help="Maximum number of candidate verification searches executed in parallel.",
    )
    parser.add_argument("--max-agent-tool-turns", type=int, default=4)
    parser.add_argument(
        "--agent-prepared-search-budget",
        type=int,
        default=2,
        help=(
            "Task-level supplemental agent search budget when Evidence Prepare already "
            "has usable search evidence. Use -1 to disable the gate."
        ),
    )
    parser.add_argument("--stage1-early-stop-max-retries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--models", default=None, help="Comma-separated model aliases understood by SLM_Agent.")
    parser.add_argument("--log-name", default="gaia_run", help="Name for this run's output folder and Markdown log.")
    parser.add_argument("--output-dir", default=None, help="Override the per-task JSON output directory.")
    parser.add_argument("--report-md", default=None, help="Override the Markdown report output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_gaia_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
