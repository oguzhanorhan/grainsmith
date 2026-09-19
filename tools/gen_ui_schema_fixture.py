"""Regenerate the frontend schema fixture from the real backend schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from grainsmith.ui.schema_json import schema_to_json

FIXTURE = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "builder"
           / "__fixtures__" / "schemaData.ts")
HEADER = """// AUTO-GENERATED from the full RunConfig tree.
// Regenerate: python -m tools.gen_ui_schema_fixture
import type { SchemaDoc } from '../../api/types'

export const realSchemaData: SchemaDoc = """


def render() -> str:
    return HEADER + json.dumps(schema_to_json(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        return int(not FIXTURE.is_file() or FIXTURE.read_text(encoding="utf-8") != text)
    FIXTURE.write_text(text, encoding="utf-8")
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())