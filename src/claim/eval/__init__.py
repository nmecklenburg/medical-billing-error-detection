from claim.eval.contracts import EvalCase, EvaluationRunConfig, ModelSpec

__all__ = [
    "EvalCase",
    "EvaluationRunConfig",
    "ModelSpec",
    "main",
    "parse_args",
]


def __getattr__(name: str) -> object:
    if name == "main":
        from claim.eval.run import main

        return main
    if name == "parse_args":
        from claim.eval.run import parse_args

        return parse_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
