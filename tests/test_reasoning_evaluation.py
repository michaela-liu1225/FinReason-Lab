from __future__ import annotations

import json

import pytest

import finreason.reasoning_evaluation as reasoning_evaluation
from finreason.generator import CompletionMetadata
from finreason.reasoning_evaluation import run_reasoning_evaluation, score_answer
from finreason.schema import RetrievalHit
from finreason.workflow import GeneratedProgram, GenerationRequest


def _write_dataset(path, *, include_unanswered: bool = False) -> None:
    payload = [
        {
            "id": "report-1-q1",
            "filename": "report-1.pdf",
            "pre_text": [],
            "post_text": [],
            "table": [["metric", "2021", "2022"], ["revenue", "100", "125"]],
            "qa": {
                "question": "By how much did revenue increase in 2022?",
                "answer": "25",
                "exe_ans": 25,
                "program": "subtract(125, 100)",
                "gold_inds": {"table_1": "SECRET_GOLD_MUST_NOT_ENTER_PROMPT"},
            },
        }
    ]
    if include_unanswered:
        payload.append(
            {
                "id": "report-2-q1",
                "filename": "report-2.pdf",
                "pre_text": ["Completely unrelated words."],
                "post_text": [],
                "table": [],
                "qa": {
                    "question": "unmatchedquasar",
                    "answer": "7",
                    "exe_ans": 7,
                    "program": "add(7, const_0)",
                    "gold_inds": {"text_0": "ANOTHER_SECRET_GOLD_LABEL"},
                },
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


class GroundedGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        self.requests.append(request)
        assert not hasattr(request, "gold")
        assert "SECRET_GOLD" not in " ".join(hit.chunk.text for hit in request.evidence)
        citation = next(
            hit.chunk.evidence_id for hit in request.evidence if hit.chunk.evidence_id == "table_1"
        )
        return GeneratedProgram("subtract(125, 100)", (citation,))


def test_end_to_end_report_uses_strict_denominator_and_never_prompts_with_gold(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset, include_unanswered=True)
    generator = GroundedGenerator()

    report = run_reasoning_evaluation(
        dataset,
        generator=generator,
        top_k=5,
        max_repairs=0,
        model_config={
            "model": "fixed-test-model",
            "revision": "abc123",
            "api_key": "must-not-leak",
            "nested": {"access_token": "also-secret", "max_tokens": 512},
        },
    )

    summary = report["summary"]
    assert summary["examples"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert 0.0 <= summary["accuracy_ci95_low"] < summary["accuracy"]
    assert summary["accuracy"] < summary["accuracy_ci95_high"] <= 1.0
    assert summary["coverage"] == 0.5
    assert summary["answered_accuracy"] == 1.0
    assert summary["program_success_rate"] == 0.5
    assert summary["execution_success_rate"] == 0.5
    assert summary["citation_precision"] == 1.0
    assert summary["citation_recall"] == 0.5
    assert summary["citation_f1"] == pytest.approx(2 / 3)
    assert summary["stop_counts"] == {"completed": 1, "no_evidence": 1}
    assert summary["error_counts"] == {"no_evidence": 1}
    assert summary["model_calls"] == 1
    assert summary["model_completions_reported"] == 0
    assert len(generator.requests) == 1

    answered = report["per_query"]["report-1-q1"]
    abstained = report["per_query"]["report-2-q1"]
    assert answered["prediction"] == "25"
    assert answered["gold"] == "25"
    assert answered["correct"] is True
    assert answered["retrieved_hits"]
    assert answered["trace"][-1]["outcome"] == "answered"
    assert abstained["prediction"] is None
    assert abstained["correct"] is False
    assert report["by_gold_kind"]["table"]["accuracy"] == 1.0
    assert report["by_gold_kind"]["text"]["accuracy"] == 0.0
    assert report["config"]["model_config"] == {
        "model": "fixed-test-model",
        "revision": "abc123",
        "api_key": "[REDACTED]",
        "nested": {"access_token": "[REDACTED]", "max_tokens": 512},
    }
    assert report["config"]["gold_policy"] == ("execution_answer_then_display_answer_fallback")


def test_scoring_starts_only_after_all_predictions_finish(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset, include_unanswered=True)
    completed_predictions = 0
    original_run = reasoning_evaluation.FinReasonWorkflow.run
    original_gold_target = reasoning_evaluation._gold_target

    def recording_run(self, question):
        nonlocal completed_predictions
        result = original_run(self, question)
        completed_predictions += 1
        return result

    def guarded_gold_target(example):
        assert completed_predictions == 2
        return original_gold_target(example)

    monkeypatch.setattr(reasoning_evaluation.FinReasonWorkflow, "run", recording_run)
    monkeypatch.setattr(reasoning_evaluation, "_gold_target", guarded_gold_target)

    report = run_reasoning_evaluation(
        dataset,
        generator=GroundedGenerator(),
        max_repairs=0,
    )

    assert report["summary"]["examples"] == 2


def test_score_answer_supports_tolerance_categories_and_percent_scaling() -> None:
    assert score_answer("yes", "YES")
    assert score_answer("10.0005", "10", atol=0.001, rtol=0)
    assert score_answer("12.5", "12.5%", percent_auto_scale=True)
    assert score_answer("0.125", "12.5%", percent_auto_scale=True)
    assert not score_answer("12.5", "12.5%", percent_auto_scale=False)
    assert not score_answer("1", "100", percent_auto_scale=True)
    assert not score_answer("NaN", "NaN")
    assert not score_answer("Infinity", "Infinity")
    assert not score_answer(None, "12.5")
    with pytest.raises(ValueError, match="atol"):
        score_answer("1", "1", atol=-1)


def test_end_to_end_uses_execution_answer_despite_scoreable_display_mismatch(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    payload = [
        {
            "id": "report-1-q1",
            "filename": "report-1.pdf",
            "pre_text": [],
            "post_text": [],
            "table": [["metric", "value"], ["ratio", "24.69136"]],
            "qa": {
                "question": "What was the ratio percentage?",
                "answer": "24.69%",
                "exe_ans": 24.69136,
                "program": "add(24.69136, const_0)",
                "gold_inds": {"table_1": "the ratio was 24.69136"},
            },
        }
    ]
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    class ExecutionAnswerGenerator:
        def generate(self, request):
            return GeneratedProgram("add(24.69136, const_0)", ("table_1",))

    report = run_reasoning_evaluation(
        dataset,
        generator=ExecutionAnswerGenerator(),
        max_repairs=0,
    )

    row = report["per_query"]["report-1-q1"]
    assert row["gold"] == "24.69136"
    assert row["gold_source"] == "execution_answer"
    assert row["correct"] is True


class TelemetryGenerator(GroundedGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.completions: list[CompletionMetadata] = []

    @property
    def telemetry_count(self) -> int:
        return len(self.completions)

    def telemetry_since(self, index: int) -> tuple[CompletionMetadata, ...]:
        return tuple(self.completions[index:])

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        result = super().generate(request)
        self.completions.append(
            CompletionMetadata(
                request_id="request-1",
                model="fixed-model-revision",
                finish_reason="stop",
                prompt_tokens=40,
                completion_tokens=10,
                total_tokens=50,
                latency_ms=12.5,
            )
        )
        return result


def test_reasoning_evaluation_aggregates_model_telemetry(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)

    report = run_reasoning_evaluation(dataset, generator=TelemetryGenerator())

    summary = report["summary"]
    assert summary["model_calls"] == 1
    assert summary["model_completions_reported"] == 1
    assert summary["prompt_tokens"] == 40
    assert summary["completion_tokens"] == 10
    assert summary["total_tokens"] == 50
    assert summary["mean_model_latency_ms"] == 12.5
    assert summary["reported_models"] == {"fixed-model-revision": 1}
    assert summary["finish_reason_counts"] == {"stop": 1}
    assert report["per_query"]["report-1-q1"]["model_telemetry"][0]["request_id"] == "request-1"


class RepairingGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        self.requests.append(request)
        if request.feedback is None:
            return GeneratedProgram("subtract(999, 100)", ("table_1",))
        assert request.feedback.error_code == "ungrounded_numeric_argument"
        return GeneratedProgram("subtract(125, 100)", ("table_1",))


def test_reasoning_evaluation_reports_repair_rate_and_success(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)
    generator = RepairingGenerator()

    report = run_reasoning_evaluation(
        dataset,
        generator=generator,
        max_repairs=1,
    )

    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["repair_rate"] == 1.0
    assert report["summary"]["repair_answer_rate"] == 1.0
    assert report["summary"]["repair_success_rate"] == 1.0
    assert report["summary"]["repair_successes"] == 1
    assert report["per_query"]["report-1-q1"]["repairs_used"] == 1
    assert len(generator.requests) == 2


class WrongRepairGenerator:
    def generate(self, request: GenerationRequest) -> GeneratedProgram:
        if request.feedback is None:
            return GeneratedProgram("subtract(999, 100)", ("table_1",))
        return GeneratedProgram("add(125, const_0)", ("table_1",))


def test_repair_success_requires_a_correct_answer(tmp_path) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset)

    report = run_reasoning_evaluation(
        dataset,
        generator=WrongRepairGenerator(),
        max_repairs=1,
    )

    assert report["summary"]["coverage"] == 1.0
    assert report["summary"]["accuracy"] == 0.0
    assert report["summary"]["repair_answer_rate"] == 1.0
    assert report["summary"]["repair_success_rate"] == 0.0
    assert report["per_query"]["report-1-q1"]["repair_answered"] is True
    assert report["per_query"]["report-1-q1"]["repair_success"] is False


def test_hybrid_models_initialize_once_and_candidate_depth_is_bound(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset, include_unanswered=True)
    initializations = {"encoder": 0, "reranker": 0}
    candidate_depths: list[int] = []

    class FakeEncoder:
        def __init__(self, model_name, **kwargs):
            initializations["encoder"] += 1
            assert model_name == "dense-model"
            assert kwargs == {"revision": "dense-rev"}

    class FakeReranker:
        def __init__(self, model_name, **kwargs):
            initializations["reranker"] += 1
            assert model_name == "reranker-model"
            assert kwargs == {"revision": "reranker-rev"}

    class FakeHybridRetriever:
        def __init__(self, chunks, *, dense_encoder, reranker):
            self.chunks = tuple(chunks)
            assert isinstance(dense_encoder, FakeEncoder)
            assert isinstance(reranker, FakeReranker)

        def retrieve(self, query, *, k=5, candidate_k=None):
            candidate_depths.append(candidate_k)
            preferred = next(
                (chunk for chunk in self.chunks if chunk.evidence_id == "table_1"),
                self.chunks[0],
            )
            return (RetrievalHit(preferred, score=1.0, rank=1),)

    class HybridGenerator:
        def generate(self, request):
            citation = request.evidence[0].chunk.evidence_id
            if "revenue" in request.question:
                return GeneratedProgram("subtract(125, 100)", (citation,))
            return GeneratedProgram("add(const_7, const_0)", (citation,))

    monkeypatch.setattr(reasoning_evaluation, "SentenceTransformerEncoder", FakeEncoder)
    monkeypatch.setattr(reasoning_evaluation, "SentenceTransformerCrossEncoder", FakeReranker)
    monkeypatch.setattr(reasoning_evaluation, "HybridRetriever", FakeHybridRetriever)

    report = run_reasoning_evaluation(
        dataset,
        generator=HybridGenerator(),
        method="hybrid",
        embedding_model="dense-model",
        embedding_revision="dense-rev",
        reranker_model="reranker-model",
        reranker_revision="reranker-rev",
        candidate_k=17,
        max_repairs=0,
    )

    assert initializations == {"encoder": 1, "reranker": 1}
    assert candidate_depths == [17, 17]
    assert report["config"]["candidate_k"] == 17


def test_hybrid_document_setup_failure_counts_as_incorrect_instead_of_aborting(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dev.json"
    _write_dataset(dataset, include_unanswered=True)

    class FakeEncoder:
        def __init__(self, model_name, **kwargs):
            pass

    class FailingHybridRetriever:
        def __init__(self, chunks, *, dense_encoder, reranker):
            raise RuntimeError("simulated document encoder OOM")

    class NeverCalledGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            raise AssertionError("generator must not run after retriever setup failure")

    monkeypatch.setattr(reasoning_evaluation, "SentenceTransformerEncoder", FakeEncoder)
    monkeypatch.setattr(reasoning_evaluation, "HybridRetriever", FailingHybridRetriever)
    generator = NeverCalledGenerator()

    report = run_reasoning_evaluation(
        dataset,
        generator=generator,
        method="hybrid",
        embedding_model="dense-model",
        max_repairs=0,
    )

    assert report["summary"]["examples"] == 2
    assert report["summary"]["accuracy"] == 0.0
    assert report["summary"]["stop_counts"] == {"retrieval_error": 2}
    assert report["summary"]["error_counts"] == {"retrieval_error": 2}
    assert generator.calls == 0
