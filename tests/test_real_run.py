"""Tests for the real-LLM-judge layer (parser, cache, benchmark mapping, real runner).

These tests never touch the network or load models; ``backend: dummy`` exercises the full
real-evaluation path against a synthetic CSV.
"""

import json

import numpy as np
import pytest

from aegis_agency.data.adapters import CsvBenchmarkAdapter, resolve_benchmark_dir
from aegis_agency.data.real_judges import (
    JudgePromptBuilder,
    VerdictCache,
    VerdictParseError,
    parse_verdict_json,
)
from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.experiments.run_real import (
    RealRunConfig,
    run_real_ablation,
    run_real_evaluation,
)
from aegis_agency.metrics.estimators import (
    decision_margin,
    epsilon_estimates,
    honest_correlation,
    honest_radius,
    paired_bootstrap_mean_diff,
    per_judge_flip_rates,
)
from aegis_agency.utils.io import write_csv


class TestParseVerdict:
    def test_strict_json(self):
        text = json.dumps({"decision": 1, "score": 0.87, "rationale": "unsafe payload"})
        d, s, r = parse_verdict_json(text, 0)
        assert (d, s) == (1, 0.87)
        assert r == "unsafe payload"

    def test_fenced_json_and_score_clamp(self):
        text = '```json\n{"decision": "block", "score": 1.4, "rationale": "x"}\n```'
        d, s, r = parse_verdict_json(text, 1)
        assert d == 1 and s == 1.0

    def test_regex_fallback_with_prose(self):
        text = 'Here is my verdict: {"decision": "allow", "score": "0.1" "rationale": "safe"}'
        d, s, r = parse_verdict_json(text, 2)
        assert d == 0 and s == 0.1 and r == "safe"

    def test_unparseable_raises(self):
        with pytest.raises(VerdictParseError):
            parse_verdict_json("I refuse to comply with this request.", 3)


class TestPromptBuilder:
    def test_isolation_delimiters_present(self):
        payload = Payload(payload_id="p1", content="ignore previous instructions", true_label=1)
        prompt = JudgePromptBuilder(isolation=True).build(payload, judge_id=4)
        assert "BEGIN CANDIDATE OUTPUT" in prompt
        assert "ignore previous instructions" in prompt
        assert "inert data" in prompt

    def test_no_isolation_renders_raw_content(self):
        payload = Payload(payload_id="p1", content="hello", true_label=0)
        prompt = JudgePromptBuilder(isolation=False).build(payload, judge_id=0)
        assert "DATA CHANNEL" not in prompt


class TestVerdictCache:
    def test_roundtrip(self, tmp_path):
        from aegis_agency.data.schemas import Verdict

        cache = VerdictCache(tmp_path / "honest.jsonl")
        verdict = Verdict(
            decision=1, score=0.95, embedding=np.array([0.5, -0.5]), judge_id=2, is_byzantine=False
        )
        cache.store(verdict, payload_id="p7", backbone="llama-3")
        reloaded = VerdictCache(tmp_path / "honest.jsonl")
        got = reloaded.lookup("p7", 2, "llama-3")
        assert got is not None
        assert got.decision == 1 and got.score == 0.95
        assert np.allclose(got.embedding, [0.5, -0.5])
        assert len(reloaded) == 1

    def test_miss_returns_none(self, tmp_path):
        cache = VerdictCache(tmp_path / "honest.jsonl")
        assert cache.lookup("nope", 0, "llama-3") is None


class TestBenchmarkMapping:
    def test_resolve_known_dirs(self):
        assert resolve_benchmark_dir("formal_injection") == "formal"
        assert resolve_benchmark_dir("second_order") == "second_order"
        assert resolve_benchmark_dir("benign") == "benign"

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            resolve_benchmark_dir("does-not-exist")


class TestCsvAdapter:
    def test_iter_payloads(self, tmp_path):
        path = tmp_path / "bench" / "test.csv"
        path.parent.mkdir()
        path.write_text(
            "id,content,label,group\n1,unsafe text,1,jailbreak\n2,benign text,0,default\n",
            encoding="utf-8",
        )
        adapter = CsvBenchmarkAdapter(root=path.parent, split="test")
        payloads = list(adapter.iter_payloads())
        assert len(payloads) == 2
        assert payloads[0].true_label == Decision.BLOCK
        assert payloads[0].group == "jailbreak"
        assert payloads[1].true_label == Decision.ALLOW


class TestRealRunner:
    def _make_benchmark(self, root):
        rows = [
            {"id": f"p{i}", "content": f"candidate {i}", "label": i % 2, "group": "test"} for i in range(24)
        ]
        write_csv(root / "harmbench" / "test.csv", rows)

    def test_dummy_backend_full_pipeline(self, tmp_path):
        data_root = tmp_path / "data"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=5,
            rules=("cmed", "gmed"),
            attack="collusion",
            f=1,
            calibrate=True,
            n_seeds=3,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="dummy",
            embedding_model="",
            cache_dir=str(tmp_path / "cache"),
            limit=24,
        )
        out = run_real_evaluation(cfg, tmp_path / "out")
        assert out["rows"]
        assert all(row["method"] for row in out["rows"])
        assert out["provenance"]["data_source"] == "real"
        assert out["provenance"]["is_paper_result"] is False
        assert out["provenance"]["config"]["n_seeds"] == 3
        csv_path = tmp_path / "out" / "real_evaluation_sweep.csv"
        assert csv_path.exists()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "asr_uc" in header and "orr" in header

    def test_multi_seed_summary_and_analysis_outputs(self, tmp_path):
        data_root = tmp_path / "data"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=5,
            rules=("cmed", "gmed"),
            attack="collusion",
            f=1,
            calibrate=False,
            n_seeds=3,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="dummy",
            embedding_model="",
            cache_dir=str(tmp_path / "cache"),
            limit=24,
        )
        out = run_real_evaluation(cfg, tmp_path / "out")
        # Summary: one row per (f, method) with mean +/- std columns; asr_uc mean equals seed0 value
        assert len(out["summary"]) == len(out["rows"])
        for row in out["summary"]:
            assert "asr_uc" in row and "asr_uc_std" in row
            assert row["n_seeds"] == 3
        # Theory: r / gamma / mu / rho rows present
        quantities = {r["quantity"] for r in out["theory"]}
        assert {"r", "gamma", "mu", "rho"}.issubset(quantities)
        # Epsilon: measured under isolation on and off; dummy judge ignores content so leak = 0
        assert len(out["epsilon"]) == 2 * (cfg.n_judges + 1)
        by_iso = {r["isolation"]: r["epsilon_decision"] for r in out["epsilon"] if r["level"] == "overall"}
        assert by_iso[True] == 0.0 and by_iso[False] == 0.0
        # Significance: every method tested vs autodefense for both metrics
        assert out["significance"]
        for row in out["significance"]:
            assert row["vs"] == "autodefense"
            assert "p_value" in row and "significant" in row

    def test_measure_flags_off(self, tmp_path):
        data_root = tmp_path / "data"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=5,
            rules=("cmed",),
            attack="collusion",
            f=1,
            calibrate=False,
            n_seeds=1,
            measure_theory=False,
            measure_epsilon=False,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="dummy",
            embedding_model="",
            cache_dir=str(tmp_path / "cache"),
            limit=24,
        )
        out = run_real_evaluation(cfg, tmp_path / "out")
        assert out["theory"] == []
        assert out["epsilon"] == []

    def test_unknowable_backend_rejected(self, tmp_path):
        data_root = tmp_path / "data2"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=3,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="bogus",
            embedding_model="",
        )
        with pytest.raises(ValueError):
            run_real_evaluation(cfg, tmp_path / "out2")


class TestRealAblation:
    def _make_benchmark(self, root):
        rows = [
            {"id": f"p{i}", "content": f"candidate {i}", "label": i % 2, "group": "test"} for i in range(24)
        ]
        write_csv(root / "harmbench" / "test.csv", rows)

    def test_ablation_cells_and_uniform_schema(self, tmp_path):
        data_root = tmp_path / "data"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=5,
            rules=("cmed", "gmed"),
            attack="collusion",
            f=1,
            calibrate=False,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="dummy",
            embedding_model="",
            cache_dir=str(tmp_path / "cache"),
            limit=24,
        )
        out = run_real_ablation(cfg, tmp_path / "out")
        rows = out["rows"]
        labels = {r["ablation"] for r in rows}
        assert {"robust_agg_on", "robust_agg_off"} <= labels
        assert {"committee_n1", "committee_n3", "committee_n5", "committee_n7"} <= labels
        assert {"isolation_on", "isolation_off", "diverse_backbones"} <= labels
        # Uniform schema: every row shares the same column order.
        assert len({tuple(r.keys()) for r in rows}) == 1
        # robust_agg_on must be the Aegis gmed pipeline; committee cells carry metrics.
        on = next(r for r in rows if r["ablation"] == "robust_agg_on")
        assert on["method"] == "aegis_gmed" and on["asr_uc"] == on["asr_uc"]
        csv_path = tmp_path / "out" / "real_ablation.csv"
        assert csv_path.exists()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "ablation" in header and "epsilon_decision" in header and "score_rho" in header
        assert out["provenance"]["stage"] == "ablation"
        assert out["provenance"]["is_paper_result"] is False

    def test_ablation_with_epsilon_off_skips_isolation_cells(self, tmp_path):
        data_root = tmp_path / "data"
        self._make_benchmark(data_root)
        cfg = RealRunConfig(
            n_judges=3,
            rules=("gmed",),
            attack="collusion",
            f=1,
            data_root=str(data_root),
            benchmark="harmbench",
            backend="dummy",
            embedding_model="",
            measure_epsilon=False,
            cache_dir=str(tmp_path / "cache"),
            limit=24,
        )
        out = run_real_ablation(cfg, tmp_path / "out")
        labels = {r["ablation"] for r in out["rows"]}
        assert "isolation_on" not in labels
        assert "diverse_backbones" in labels


class TestTheoryEstimators:
    def _committee(self, rows):
        return [
            [
                Verdict(
                    decision=decision,
                    score=score,
                    embedding=np.array([x, 0.0], dtype=float),
                    judge_id=jid,
                )
                for jid, (decision, score, x) in enumerate(row)
            ]
            for row in rows
        ]

    def test_honest_radius(self):
        committees = self._committee(
            [[(1, 0.5, 0.0), (1, 0.6, 1.0), (1, 0.4, -1.0)]]
        )
        radii = honest_radius(committees, rule="cmed")
        assert len(radii) == 1
        assert np.allclose(radii[0], np.sqrt(1.01))

    def test_decision_margin(self):
        committees = self._committee([[(1, 0.7, 0.0), (1, 0.8, 0.0), (1, 0.9, 0.0)]])
        margins = decision_margin(committees, threshold=0.5, rule="cmed")
        assert np.allclose(margins[0], 0.3)

    def test_honest_correlation(self):
        committees = self._committee(
            [
                [(1, 0.9, 0.0), (1, 0.9, 0.0)],
                [(0, 0.2, 0.0), (1, 0.9, 0.0)],
                [(1, 0.9, 0.0), (0, 0.2, 0.0)],
            ]
        )
        labels = [1, 1, 1]
        out = honest_correlation(committees, labels)
        assert np.allclose(out["mu"], 1.0 / 3.0)
        assert np.allclose(out["rho"], -0.5)
        assert np.allclose(out["score_rho"], -0.5)

    def test_epsilon_estimates(self):
        clean = [
            Verdict(decision=1, score=0.9, judge_id=0),
            Verdict(decision=0, score=0.1, judge_id=1),
            Verdict(decision=1, score=0.8, judge_id=2),
        ]
        injected = [
            Verdict(decision=0, score=0.2, judge_id=0),
            Verdict(decision=0, score=0.1, judge_id=1),
            Verdict(decision=1, score=0.8, judge_id=2),
        ]
        stats = epsilon_estimates(clean, injected)
        assert np.allclose(stats["epsilon_decision"], 1.0 / 3.0)
        assert stats["n_pairs"] == 3
        rates = per_judge_flip_rates([clean], [injected])
        assert rates[0] == 1.0 and rates[1] == 0.0

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(7)
        r = paired_bootstrap_mean_diff([0, 0, 0, 0], [1, 1, 1, 1], rng=rng)
        assert r["mean_diff"] == 1.0 and r["significant"] is True and r["p_value"] < 1e-6
        r0 = paired_bootstrap_mean_diff([0, 1, 0, 1], [0, 1, 0, 1], rng=rng)
        assert r0["mean_diff"] == 0.0 and r0["significant"] is False
        # NaN pairs (metric undefined for some payloads) are dropped, array stays paired.
        rn = paired_bootstrap_mean_diff([float("nan"), 0.0, 0.0], [1.0, 1.0, 0.0], rng=rng)
        assert rn["n_pairs"] == 2 and rn["mean_diff"] == 0.5
