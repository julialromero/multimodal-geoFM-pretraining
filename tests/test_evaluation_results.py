import json

import pytest

from ciip.evaluation.result_records import (
    RESULT_SCHEMA,
    EvaluationResult,
    discover_evaluation_results,
    read_evaluation_result,
    write_evaluation_result,
)


def _result():
    return EvaluationResult(
        checkpoint="epoch-10.pt",
        dataset="EuroSAT",
        split="test",
        modality="s2",
        bands=("B02", "B03", "B04"),
        feature_space="projected",
        seed=7,
        arguments={"neighbors": 5},
        metrics={"accuracy": 0.8},
    )


def test_evaluation_result_round_trip_and_discovery(tmp_path):
    path = write_evaluation_result(tmp_path / "run.json", _result())
    assert read_evaluation_result(path) == _result()
    assert discover_evaluation_results(tmp_path) == [(path, _result())]


def test_discovery_ignores_other_json(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"task": "evaluation"}))
    assert discover_evaluation_results(tmp_path) == []


def test_evaluation_result_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported evaluation schema"):
        EvaluationResult.from_dict({**_result().to_dict(), "schema": RESULT_SCHEMA + "-future"})


def test_plotting_input_can_discover_versioned_fewshot_result(tmp_path):
    pytest.importorskip("matplotlib")
    from ciip.evaluation.plot_downstream import _iter_eurosat_fewshot_results

    result = _result()
    result = EvaluationResult.from_dict(
        {
            **result.to_dict(),
            "arguments": {
                **result.arguments,
                "model_type": "ciip_checkpoint",
                "model_path": "example",
                "k_shot": 5,
                "knn_k": 1,
            },
            "metrics": {"accuracy_mean": 0.8},
        }
    )
    (tmp_path / "run").mkdir()
    write_evaluation_result(tmp_path / "run" / "evaluation_result.json", result)
    records = list(_iter_eurosat_fewshot_results(tmp_path, knn_k=1))
    assert records == [("CIIP example", 5.0, 0.8)]
