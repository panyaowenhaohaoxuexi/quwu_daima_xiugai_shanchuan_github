from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_entrypoints_do_not_import_legacy_model_or_clip():
    for path in ("Teacher.py", "EMA.py", "Eval.py", "Eval_EMA.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "VIFNetInconsistencyTeacher" not in source
        assert "import clip" not in source
