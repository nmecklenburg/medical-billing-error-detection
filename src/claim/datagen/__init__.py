from claim.datagen.contracts import GenerationRunConfig, WorkerConfiguration

__all__ = [
    "GenerationRunConfig",
    "WorkerConfiguration",
    "main",
    "parse_args",
]


def __getattr__(name: str) -> object:
    if name == "main":
        from claim.datagen.clean import main

        return main
    if name == "parse_args":
        from claim.datagen.clean import parse_args

        return parse_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
