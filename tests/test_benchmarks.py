"""Tests for benchmark table extraction from model card text."""

from __future__ import annotations

from model_atlas.extraction.benchmarks import derive_benchmark_anchors, extract_benchmarks


class TestExtractBenchmarks:
    def test_simple_table(self):
        card = """\
## Benchmark Results

| Benchmark | Score |
|-----------|-------|
| MMLU | 72.5 |
| HellaSwag | 85.3 |
| ARC | 63.1 |
"""
        results = extract_benchmarks(card)
        assert results["benchmark:mmlu"] == ("72.5", "str")
        assert results["benchmark:hellaswag"] == ("85.3", "str")
        assert results["benchmark:arc"] == ("63.1", "str")

    def test_no_table(self):
        card = "This is a model card with no tables."
        results = extract_benchmarks(card)
        assert results == {}

    def test_non_benchmark_table_ignored(self):
        card = """\
| Feature | Description |
|---------|-------------|
| Language | English |
| License | MIT |
"""
        results = extract_benchmarks(card)
        assert results == {}

    def test_multiple_tables(self):
        card = """\
## Overview

| Feature | Value |
|---------|-------|
| Size | 7B |

## Evaluation Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 72.5 | 5-shot |
| GSM8K | 68.0 | 8-shot |
"""
        results = extract_benchmarks(card)
        # First table should be ignored (no benchmark keywords)
        assert "benchmark:size" not in results
        # Second table should be parsed
        assert results["benchmark:mmlu"] == ("72.5", "str")
        assert results["benchmark:gsm8k"] == ("68.0", "str")

    def test_empty_input(self):
        assert extract_benchmarks("") == {}

    def test_table_with_percentage(self):
        card = """\
| Eval | Accuracy |
|------|----------|
| MMLU | 72.5% |
"""
        results = extract_benchmarks(card)
        assert results["benchmark:mmlu"] == ("72.5%", "str")

    def test_name_normalization(self):
        card = """\
| Benchmark | Score |
|-----------|-------|
| Hella Swag | 85.3 |
"""
        results = extract_benchmarks(card)
        assert "benchmark:hella-swag" in results


class TestDeriveBenchmarkAnchors:
    def test_high_mmlu(self):
        benchmarks = {"benchmark:mmlu": ("75.2", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        labels = {a.label for a in anchors}
        assert "high-mmlu" in labels

    def test_below_threshold(self):
        benchmarks = {"benchmark:mmlu": ("45.0", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        assert anchors == []

    def test_strong_humaneval(self):
        benchmarks = {"benchmark:humaneval": ("55.3%", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        labels = {a.label for a in anchors}
        assert "strong-humaneval" in labels

    def test_strong_gsm8k(self):
        benchmarks = {"benchmark:gsm8k": ("68.0", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        labels = {a.label for a in anchors}
        assert "strong-gsm8k" in labels

    def test_unknown_benchmark_ignored(self):
        benchmarks = {"benchmark:winogrande": ("80.0", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        assert anchors == []

    def test_multiple_thresholds(self):
        benchmarks = {
            "benchmark:mmlu": ("72.5", "str"),
            "benchmark:gsm8k": ("65.0", "str"),
            "benchmark:humaneval": ("30.0", "str"),  # below threshold
        }
        anchors = derive_benchmark_anchors(benchmarks)
        labels = {a.label for a in anchors}
        assert "high-mmlu" in labels
        assert "strong-gsm8k" in labels
        assert "strong-humaneval" not in labels

    def test_empty_benchmarks(self):
        assert derive_benchmark_anchors({}) == []

    def test_confidence_075(self):
        benchmarks = {"benchmark:mmlu": ("75.0", "str")}
        anchors = derive_benchmark_anchors(benchmarks)
        assert anchors[0].confidence == 0.75
