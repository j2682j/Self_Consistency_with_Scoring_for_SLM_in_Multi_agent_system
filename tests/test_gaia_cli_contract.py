from __future__ import annotations

import pytest

from benchmark.gaia.gaia_runner import parse_args


def test_updated_agent_cli_defaults() -> None:
    args = parse_args([])

    assert args.evidence_prepare is True
    assert args.enable_agent_tool_use is True
    assert args.max_agent_tool_turns == 4
    assert args.agent_prepared_search_budget == 2
    assert not hasattr(args, "enable_stage1_early_stop")
    assert not hasattr(args, "candidate_verification_search")
    assert not hasattr(args, "versa_prm_local_files_only")


def test_evidence_prepare_is_one_boolean_parameter() -> None:
    assert parse_args(["--evidence-prepare"]).evidence_prepare is True
    assert parse_args(["--evidence-prepare", "false"]).evidence_prepare is False
    assert parse_args(["--evidence-prepare", "true"]).evidence_prepare is True


def test_updated_agent_cli_values() -> None:
    assert parse_args(["--enable-agent-tool-use"]).enable_agent_tool_use is True
    args = parse_args(
        [
            "--no-enable-agent-tool-use",
            "--max-agent-tool-turns",
            "7",
            "--agent-prepared-search-budget",
            "3",
        ]
    )

    assert args.enable_agent_tool_use is False
    assert args.max_agent_tool_turns == 7
    assert args.agent_prepared_search_budget == 3


@pytest.mark.parametrize(
    "removed_argv",
    [
        ["--no-evidence-prepare"],
        ["--enable-stage1-tool-use"],
        ["--max-stage1-tool-turns", "4"],
        ["--stage1-prepared-search-budget", "2"],
        ["--enable-stage1-early-stop"],
        ["--versa-prm-local-files-only"],
        ["--versa-prm-allow-download"],
        ["--enable-candidate-verification-search"],
    ],
)
def test_removed_cli_options_are_rejected(removed_argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(removed_argv)
