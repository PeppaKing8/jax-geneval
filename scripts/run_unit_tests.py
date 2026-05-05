from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tests = [
        ROOT / "tests" / "test_deformable_attention.py",
        ROOT / "tests" / "test_swin.py",
    ]
    total = 0
    for path in tests:
        module = _load(path)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            print(f"[run] {path.name}::{name}", flush=True)
            fn()
            total += 1
    print(f"[ok] {total} tests passed")


if __name__ == "__main__":
    main()

