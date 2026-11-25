from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from driftbench.benchmarking.pgbench import (
    PgBenchConnection,
    _run_phase,
)
from driftbench.benchmarking.policy import (
    BenchmarkPolicyError,
    load_pgbench_policy,
    parse_pgbench_policy_payload,
)
from driftbench.data.pgbench import queries
from ..helpers import REPO_ROOT


POLICIES = REPO_ROOT / "driftbench" / "benchmarking" / "policies"
SELECT_POLICY = POLICIES / "pgbench_ci_v1.json"
UPDATE_POLICY = POLICIES / "pgbench_ci_v1_simple_update.json"


class SimpleUpdatePolicyTests(unittest.TestCase):
    def test_new_policy_matches_select_policy_config_and_thresholds(self) -> None:
        self.assertTrue(UPDATE_POLICY.is_file(), "simple-update policy is missing")
        select = load_pgbench_policy(SELECT_POLICY)
        update = load_pgbench_policy(UPDATE_POLICY)
        self.assertEqual(update.policy_version, "pgbench-ci-v1-simple-update")
        self.assertEqual(update.workload, "simple_update")
        self.assertEqual(update.run_config(), select.run_config())
        for field in (
            "min_tps_ratio", "max_p95_latency_ratio", "max_error_rate",
            "max_tps_relative_delta",
        ):
            self.assertEqual(getattr(update, field), getattr(select, field), field)

    def test_payload_accepts_simple_update(self) -> None:
        payload = json.loads(SELECT_POLICY.read_text(encoding="utf-8"))
        payload["workload"] = "simple_update"
        try:
            policy = parse_pgbench_policy_payload(payload)
        except BenchmarkPolicyError as exc:
            self.fail(f"simple_update was rejected: {exc}")
        self.assertEqual(policy.workload, "simple_update")

    def test_payload_rejects_tpcb(self) -> None:
        payload = json.loads(SELECT_POLICY.read_text(encoding="utf-8"))
        payload["workload"] = "tpcb"
        with self.assertRaises(BenchmarkPolicyError):
            parse_pgbench_policy_payload(payload)

    def test_generated_script_matches_pgbench_16_simple_update_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = queries(workload="simple_update", clients=2, duration=5).generate(
                output_dir=temporary, force=True
            )
            script = next(path for path in result.files if path.suffix == ".sql")
            executable_lines = [
                line.strip() for line in script.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("--")
            ]
        # Verified with PostgreSQL 16's pgbench --show-script=simple-update.
        self.assertEqual(executable_lines, [
            r"\set aid random(1, 100000 * :scale)",
            r"\set bid random(1, 1 * :scale)",
            r"\set tid random(1, 10 * :scale)",
            r"\set delta random(-5000, 5000)",
            "BEGIN;",
            "UPDATE pgbench_accounts SET abalance = abalance + :delta WHERE aid = :aid;",
            "SELECT abalance FROM pgbench_accounts WHERE aid = :aid;",
            "INSERT INTO pgbench_history (tid, bid, aid, delta, mtime) VALUES (:tid, :bid, :aid, :delta, CURRENT_TIMESTAMP);",
            "END;",
        ])

    def test_phase_uses_matching_builtin_and_candidate_file(self) -> None:
        class PhaseStarted(Exception):
            pass

        def capture(command, **kwargs):
            raise PhaseStarted(command)

        select = load_pgbench_policy(SELECT_POLICY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "candidate.sql"
            for workload, builtin in (("select_only", "select-only"),
                                      ("simple_update", "simple-update")):
                for role in ("baseline", "candidate"):
                    with self.subTest(workload=workload, role=role), patch(
                        "driftbench.benchmarking.pgbench._run_process", side_effect=capture
                    ), self.assertRaises(PhaseStarted) as started:
                        _run_phase(
                            output_root=root, phase_dir=root / workload / role,
                            role=role, duration_seconds=5, candidate_script=script,
                            policy=replace(select, workload=workload),
                            connection=PgBenchConnection(database="driftbench_ci"),
                            pgbench_binary="pgbench",
                        )
                    command = started.exception.args[0]
                    flag = "--builtin" if role == "baseline" else "--file"
                    expected = builtin if role == "baseline" else str(script.resolve())
                    self.assertEqual(command[command.index(flag) + 1], expected)
                    self.assertNotIn("--file" if role == "baseline" else "--builtin", command)

    def test_unsupported_workload_is_rejected_before_either_phase(self) -> None:
        policy = replace(load_pgbench_policy(SELECT_POLICY), workload="tpcb")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for role in ("baseline", "candidate"):
                with self.subTest(role=role), patch(
                    "driftbench.benchmarking.pgbench._run_process",
                    side_effect=AssertionError("unsupported workload started a phase"),
                ), self.assertRaises(BenchmarkPolicyError):
                    _run_phase(
                        output_root=root, phase_dir=root / role, role=role,
                        duration_seconds=5, candidate_script=root / "candidate.sql",
                        policy=policy, connection=PgBenchConnection(database="driftbench_ci"),
                        pgbench_binary="pgbench",
                    )
                self.assertFalse((root / role).exists())


@unittest.skipUnless(
    os.environ.get("DRIFTBENCH_REQUIRE_PG_INTEGRATION") == "1",
    "set DRIFTBENCH_REQUIRE_PG_INTEGRATION=1 for the real PostgreSQL gate",
)
class SimpleUpdatePostgresIntegrationTests(unittest.TestCase):
    def test_paired_gate_and_offline_bundle_verifier(self) -> None:
        env = os.environ.copy()
        self.assertTrue(env.get("PGPASSWORD"), "PGPASSWORD must be set")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, env=env,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        env.setdefault("DRIFTBENCH_GIT_SHA", head)
        self.assertEqual(env["DRIFTBENCH_GIT_SHA"], head)
        database = env.get("PGDATABASE", "driftbench_ci")
        host = env.get("PGHOST", "localhost")
        port = env.get("PGPORT", "5432")
        username = env.get("PGUSER", "driftbench")
        init = subprocess.run(
            ["pgbench", "--initialize", "--scale=1", "--host", host,
             "--port", port, "--username", username, database],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(init.returncode, 0, init.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = queries(workload="simple_update", clients=2, duration=5).generate(
                output_dir=root, force=True
            )
            candidate = next(path for path in generated.files if path.suffix == ".sql")
            bundle = root / "results"
            gate = subprocess.run(
                [sys.executable, "-m", "driftbench.cli", "benchmark", "pgbench",
                 "--policy", str(UPDATE_POLICY), "--candidate-script", str(candidate),
                 "--output-dir", str(bundle), "--database", database, "--host", host,
                 "--port", port, "--username", username, "--json"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            self.assertTrue(json.loads(gate.stdout)["ok"])
            self.assertTrue(json.loads((bundle / "decision.json").read_text())["ok"])
            verify = subprocess.run(
                [sys.executable, "-m", "driftbench.cli", "benchmark", "verify",
                 "--bundle", str(bundle), "--json"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
            self.assertTrue(json.loads(verify.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
