#!/usr/bin/env python3
"""Run the search script against samples/ and compare with the expectations.

    python3 check_only_connected.py                       # thread pool
    python3 check_only_connected.py --pool process -j 4   # extra args pass through

1. Every marker in expected.json is searched in each mode (no flag /
   --only_connected / --connected-mode output), once over all metadata and
   once each restricted to the `prompt` and to the `workflow` chunk, so a
   filter that works on one format can't hide a bug in the other.
2. The multi-term cases in expected_multi.json (mostly --scope node) are
   checked for the files they match and, where given, the matching fields.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAMPLES = Path(__file__).with_name("samples")
SCRIPT = os.environ.get(
    "IMGMETA_SCRIPT",
    "search_string_image_meta.py")

MODES = {"all": [], "linked": ["--only_connected"],
         "output": ["--connected-mode", "output"]}
# WebP keeps prompt/workflow in EXIF Model/Make
CHUNKS = {"any field": [], "prompt": ["--fields", "prompt", "Model"],
          "workflow": ["--fields", "workflow", "Make"]}


def run(patterns, args):
    cmd = [sys.executable, SCRIPT, str(SAMPLES), *patterns, "--json",
           "--progress", "never", "--errors", *args, *sys.argv[1:]]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, 1):
        sys.exit(f"search failed ({res.returncode}): {cmd}\n{res.stderr}")
    report = json.loads(res.stdout)
    if report["unreadable"]:
        sys.exit(f"unreadable files: {cmd}\n{res.stderr}")
    return report["results"]


def search(patterns, args):
    return {Path(r["path"]).name for r in run(patterns, args)}


def check_markers(failures):
    expect = json.loads((SAMPLES / "expected.json").read_text())
    markers = sorted({m for marks in expect.values() for m in marks})
    jobs = [(m, chunk, mode) for m in markers for chunk in CHUNKS for mode in MODES]
    with ThreadPoolExecutor(8) as pool:
        found = dict(zip(jobs, pool.map(
            lambda job: search([job[0]], CHUNKS[job[1]] + MODES[job[2]]), jobs)))

    for marker in markers:
        wanted = {mode: {f for f, marks in expect.items()
                         if marks.get(marker, "nnn")[i] == "y"}
                  for i, mode in enumerate(MODES)}
        row = []
        for chunk in CHUNKS:
            # within one chunk, only files whose chunk holds the marker can hit
            present = found[(marker, chunk, "all")]
            for mode in MODES:
                want = wanted[mode] if chunk == "any field" else wanted[mode] & present
                got = found[(marker, chunk, mode)]
                row.append("ok " if got == want else "BAD")
                if got != want:
                    failures.append(f"{marker!r} [{chunk} / {mode}]: "
                                    f"missing {sorted(want - got)}, "
                                    f"unexpected {sorted(got - want)}")
        print(f"{marker:<34} " + " ".join(row))
    head = "  ".join(f"{c}: all/linked/output" for c in CHUNKS)
    print(f"\ncolumns -> {head}")
    return len(jobs)


def check_multi(failures):
    cases = json.loads((SAMPLES / "expected_multi.json").read_text())
    with ThreadPoolExecutor(8) as pool:
        reports = list(pool.map(lambda c: run(c["patterns"], c["args"]), cases))

    print("\nmulti-term searches")
    for case, results in zip(cases, reports):
        got = {Path(r["path"]).name: sorted(m["field"] for m in r["matches"])
               for r in results}
        ok = set(got) == set(case["files"])
        for name, fields in (case["fields"] or {}).items():
            ok = ok and got.get(name) == sorted(fields)
        label = (" ".join(f'"{p}"' for p in case["patterns"]) + " "
                 + " ".join(case["args"])).strip()
        print(f"  {'ok ' if ok else 'BAD'} {label}")
        if not ok:
            failures.append(f"{label}: expected {sorted(case['files'])} "
                            f"{case['fields'] or ''}, got {got}")
    return len(cases)


def main():
    failures: list = []
    total = check_markers(failures) + check_multi(failures)
    if failures:
        print(f"\n{len(failures)} failure(s):")
        print("\n".join("  " + f for f in failures))
        return 1
    print(f"\nall {total} searches matched expectations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
