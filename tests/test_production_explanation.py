import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.explain import build_explanation, validate_explanation, validate_packet, describe_feature
from reproduce import prepare_inputs


def packet(n=5, transform="identity", value=1.306):
    raw = math.expm1(value) if transform == "log1p" else value
    return {"endpoint": "tlr8_potentiation", "response_direction": "higher = stronger potentiation",
            "prediction_value_scale": "relative reporter response", "prediction": raw,
            "query": {"sequence_5to3": "ATGC", "helm": "synthetic fixture"},
            "provenance": {"source": "synthetic fixture, not measured"},
            "selected_transform": transform,
            "model_scale": "log1p(y)" if transform == "log1p" else "y",
            "members": {str(i): {"prediction": raw, "model_scale_prediction": value, "expected_value": 0.2} for i in range(n)},
            "member_contribution_sums": {str(i): value - 0.2 for i in range(n)},
            "branch_evidence": [{"member": str(i), "component": "fixture", "weight": 1,
                                 "prediction": value, "expected_value": 0.2, "contribution_sum": value - 0.2} for i in range(n)],
            "top_feature_absolute_mass_fraction": 1.0,
            "feature_evidence": [{"member": str(i), "feature": "ucore__n_TCC", "family": "motif", "value": 0,
                                  "contribution": value - 0.2} for i in range(n)],
            "top_features": [{"feature": "ucore__n_TCC", "family": "motif", "value": 0,
                              "signed_mean_model_scale": value - 0.2,
                              "absolute_mean_model_scale": abs(value - 0.2)}]}


@pytest.mark.parametrize("n", [1, 5, 15])
@pytest.mark.parametrize("transform", ["identity", "log1p"])
@pytest.mark.parametrize("value", [0, -0.1, 1.306])
def test_dynamic_rendering(n, transform, value):
    p = packet(n, transform, value)
    result = build_explanation(p)
    assert validate_explanation(p, result)["status"] == "PASS"
    assert f"{n} 个成员" in result["text_zh"]
    assert "5′-ATGC-3′" in result["text_en"]
    assert "1.695" not in result["text_en"]
    assert result["free_form_agent_semantics_validated"] is False


@pytest.mark.parametrize("field", ["prediction", "expected_value", "model_scale_prediction"])
def test_inconsistent_member_rejected(field):
    p = packet()
    p["members"]["0"][field] += 0.2
    with pytest.raises(ValueError):
        validate_packet(p)


@pytest.mark.parametrize("kind", ["ensemble", "branch", "coverage", "nan", "duplicate"])
def test_packet_corruption(kind):
    p = packet()
    if kind == "ensemble": p["prediction"] = 1.271
    if kind == "branch": p["branch_evidence"][0]["weight"] = 0.5
    if kind == "coverage": p["top_feature_absolute_mass_fraction"] = 1.2
    if kind == "nan": p["top_features"][0]["value"] = float('nan')
    if kind == "duplicate": p["branch_evidence"].append(copy.deepcopy(p["branch_evidence"][0]))
    with pytest.raises(ValueError): validate_packet(p)


@pytest.mark.parametrize("field", ["value", "signed_mean_model_scale", "absolute_mean_model_scale"])
def test_feature_summary_corruption(field):
    p = packet()
    p["top_features"][0][field] += 1
    with pytest.raises(ValueError): validate_packet(p)


@pytest.mark.parametrize("key", ["text_zh", "text_en", "agent_prompt", "packet_sha256"])
def test_old_numbers_or_extra_prose_rejected(key):
    p = packet()
    result = build_explanation(p)
    result[key] += " 1.271 1.70 1.695 establishes causality"
    with pytest.raises(ValueError): validate_explanation(p, result)


def test_bad_cluster_not_overwritten(tmp_path):
    dest = tmp_path / "clusters.csv"
    dest.write_bytes(b"wrong")
    with pytest.raises(ValueError): prepare_inputs.ensure_clusters(dest)
    assert dest.read_bytes() == b"wrong"


def test_feature_glossary_is_not_a_mechanism_claim():
    assert '至少一个' in describe_feature('pint_motif_TT__mod_O', 'zh')
    assert '5′-TT-3′' in describe_feature('pint_motif_TT__mod_O', 'zh')
    assert '第 1 位' in describe_feature('uohe__raw_pos_01_base_G', 'zh')
    assert '不据此推断' in describe_feature('new_unmapped_feature', 'zh')


def test_cluster_creation_hash_and_resume(tmp_path, monkeypatch):
    payload = b"fixture\n"
    monkeypatch.setattr(prepare_inputs, 'EXPECTED', prepare_inputs.hashlib.sha256(payload).hexdigest())
    dest = tmp_path / "clusters.csv"
    assert prepare_inputs.ensure_clusters(dest, lambda: payload) == "regenerated_and_verified"
    assert prepare_inputs.ensure_clusters(dest) == "verified_existing"
    bad = tmp_path / 'bad.csv'
    with pytest.raises(ValueError): prepare_inputs.ensure_clusters(bad, lambda: b'bad')
    assert not bad.exists()


@pytest.mark.parametrize('fail_prepare', [False, True])
def test_workflow_prerequisite_precedes_probe(tmp_path, monkeypatch, fail_prepare):
    from reproduce import workflow
    from types import SimpleNamespace
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    original_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k: 'MemAvailable: 73400320 kB\n'
                        if str(p) == '/proc/meminfo' else original_read(p, *a, **k))
    monkeypatch.setattr(workflow.shutil, 'disk_usage', lambda p: SimpleNamespace(free=40 * 1024**3))
    monkeypatch.setattr(sys, 'argv', ['workflow.py', 'source_oligogym', '--run'])
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if fail_prepare:
            raise workflow.subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    if fail_prepare:
        with pytest.raises(workflow.subprocess.CalledProcessError): workflow.main()
        assert len(calls) == 1
    else:
        workflow.main()
        assert len(calls) == 3
        assert '--probe' in calls[1] and '--probe' not in calls[2]
        assert calls[2][-1] == 'cpu'
    assert 'reproduce/prepare_inputs.py' in calls[0]
