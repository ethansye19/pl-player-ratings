"""Why this rating? Build the grounded input, render it with the selected provider and
verify that every number in the text comes from the input.

    python -m src.explain.explain --match 3754079 --player 3237 --method baseline
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.console import use_utf8_output
from src.explain.inputs import PROCESSED, explanation_input
from src.explain.llm import TemplateProvider, get_provider
from src.explain.verify import unsupported_numbers, verify_numbers

log = logging.getLogger(__name__)


@dataclass
class Explanation:
    text: str
    input: dict[str, Any]
    provider: str
    verified: bool
    llm_text: str | None = None
    rejected_numbers: list[float] = field(default_factory=list)


def explain(match_id: int, player_id: int, method: str, k: int = 5, processed_dir: Path | str = PROCESSED) -> Explanation:
    data = explanation_input(match_id, player_id, method, k=k, processed_dir=processed_dir)
    provider = get_provider()
    text = provider.explain(data)
    if verify_numbers(text, data):
        return Explanation(text=text, input=data, provider=provider.name, verified=True)
    bad = unsupported_numbers(text, data)
    log.warning("%s provider wrote numbers not in the input %s; falling back to the template", provider.name, bad)
    fallback = TemplateProvider().explain(data)
    if not verify_numbers(fallback, data):  # should be impossible; make it loud rather than silent
        raise RuntimeError(f"template explanation failed verification: {unsupported_numbers(fallback, data)}")
    return Explanation(text=fallback, input=data, provider=provider.name, verified=False, llm_text=text, rejected_numbers=bad)


def main() -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", type=int, required=True)
    parser.add_argument("--player", type=int, required=True)
    parser.add_argument("--method", choices=["baseline", "vaep", "both"], default="both")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for method in (["baseline", "vaep"] if args.method == "both" else [args.method]):
        result = explain(args.match, args.player, method, k=args.k, processed_dir=args.processed)
        print(f"[{method} | provider={result.provider} | verified={result.verified}]")
        print(result.text)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
