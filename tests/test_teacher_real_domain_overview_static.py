from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_entrypoints_construct_models_through_formal_checkpointing_only():
    for path in ("Teacher.py", "EMA.py", "Eval.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "build_model_from_config" in source
        assert "import clip" not in source
