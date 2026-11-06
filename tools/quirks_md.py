"""Write docs/quirks/<instrument>.md (human page) from docs/quirks/<instrument>.jsonl (the source).

Each record has a `kind` (review 2026-10-16):
  quirk        the equipment misbehaves, for certain (against its manual or SCPI rules)
  limit        a spec or design limit the software must respect
  observation  documented or design behaviour worth knowing; not a fault
  withdrawn    registered earlier, not reproduced on retest; kept so old citations resolve
Run: .venv/Scripts/python tools/quirks_md.py
"""
import json
import pathlib

KINDS = [("quirk", "Quirks: the equipment misbehaves"), ("limit", "Limits: spec or design limits"),
         ("observation", "Observations: documented or design behaviour"), ("withdrawn", "Withdrawn: not reproduced")]

for src in sorted(pathlib.Path("docs/quirks").glob("*.jsonl")):
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = [f"# {src.stem} quirks, limits and observations", "",
           f"Generated from `{src.name}` by `tools/quirks_md.py`. Edit the .jsonl, then regenerate.",
           "A quirk is filed only when the equipment misbehaves for certain. Review 2026-10-16: every record",
           "retested where the wiring allows; the `review` field gives the result.", ""]
    for kind, head in KINDS:
        sel = [r for r in rows if r.get("kind", "quirk") == kind]
        if not sel:
            continue
        out += [f"## {head} ({len(sel)})", "", "| ID | Area | Fact | What happens | Handling | Review |", "|---|---|---|---|---|---|"]
        for r in sel:
            out.append(f"| {r['id']} | {r['area']} | {r['title']}. {r['fact']} | {r['consequence']} | "
                       f"{r['handling']}: {r['where']} | {r.get('review', '')} |")
        out.append("")
    out += ["Evidence for each row: fields `evidence` and `review` in the .jsonl."]
    src.with_suffix(".md").write_text("\n".join(out) + "\n", encoding="utf-8")
    counts = {k: sum(r.get("kind", "quirk") == k for r in rows) for k, _ in KINDS}
    print(f"{src.with_suffix('.md')}: {counts}")
