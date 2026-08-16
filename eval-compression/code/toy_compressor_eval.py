"""
toy_compressor_eval.py — Minimal probe-based eval for context compression
Style: mirrors Hermes harness but zero external deps, heuristic QA.

Usage:
  python toy_compressor_eval.py --fixture sample

Shows:
- compress() simulation: truncate + keep artifacts
- 6-dim probe scoring (EM + simple heuristic)
- ratio / quality output

No API key needed for offline mode. For LLM judge, set OPENAI_API_KEY and --use-llm-judge.
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict
import re

# fake fixture like SWE session
SAMPLE_FIXTURE = """
SYSTEM: You are coding agent, no push without tests passing.
USER: Fix bug in auth.py: token expiry not respected, add unit test.
TURN 1: read auth.py -> found token_check uses naive datetime.
TURN 2: search docs -> found JWT spec requires exp claim validation.
TURN 3: edit auth.py: add exp check if exp < now: raise Expired.
TURN 4: ran pytest tests/test_auth.py -k expiry -> FAILED AssertionError still.
TURN 5: debugged: timezone aware vs naive compare -> fixed with utcnow().
TURN 6: ran pytest again -> PASSED 3 tests.
Artifact: modified files [auth.py], added [tests/test_token_expiry.py]
Remaining: need to ensure PR description covers CVE-2024-1234.
"""

@dataclass
class Probe:
    id: str
    dim: str  # one of 6 dims
    q: str
    gold: str

PROBES = [
    Probe("p1","accuracy","What was failing test name?","test_auth expiry"),
    Probe("p2","context_awareness","What is overall user goal?","fix token expiry bug in auth.py"),
    Probe("p3","artifact_trail","Which files were modified?","auth.py, tests/test_token_expiry.py"),
    Probe("p4","completeness","What remaining constraint unmet?","PR must cover CVE-2024-1234"),
    Probe("p5","continuity","After adding exp check, what happened before fix to timezone?","test failed due to timezone aware vs naive"),
    Probe("p6","instruction_following","What system instruction about push?","no push without tests passing"),
]

def naive_compressor(text: str, keep_ratio=0.25) -> str:
    """
    Simulates context compression: keeps first 200 chars (system/task) + last tool outputs verbatim (artifact).
    Bad compressor would keep only middle summary and drop artifact list.
    """
    lines = text.splitlines()
    head = "\n".join(lines[:2])
    tail = "\n".join(lines[-4:])  # artifact + remaining
    middle_len = len(lines)
    mid_keep = int(middle_len*keep_ratio)
    mid = "\n".join(lines[2:2+mid_keep])
    return head + "\n...[compressed]...\n" + mid + "\n" + tail

def bad_compressor(text: str) -> str:
    """Over-aggressive summarizer that drops artifacts"""
    return "User wanted fix bug in auth. We edited code and ran tests. Tests passed."

def simple_qa(compressed: str, q: str) -> str:
    """Heuristic QA: substring search, no LLM"""
    compressed_lower = compressed.lower()
    # naive: if gold keywords appear, return substring, else fallback
    # for demo, just return compressed truncated
    # Real impl would call llm: openai.ChatCompletion
    if "which files" in q.lower():
        m = re.search(r"modified files.*", compressed, re.I)
        return m.group(0) if m else compressed[:200]
    if "remaining" in q.lower() or "cve" in q.lower():
        return "PR covers CVE" if "CVE" in compressed else "unknown"
    if "push" in q.lower():
        return "no push without tests" if "no push" in compressed_lower else "unknown"
    return compressed[:150]

def em_score(gold: str, ans: str) -> int:
    """0-5 heuristic: token overlap, punctuation stripped"""
    import string
    def clean(s):
        s = s.lower()
        # keep alphanumeric, slash, dot, hyphen
        s = re.sub(r"[^\w\./-]", " ", s)
        return s.split()
    gold_tokens = set(clean(gold))
    ans_tokens = set(clean(ans))
    if not gold_tokens:
        return 3
    overlap = len(gold_tokens & ans_tokens)/len(gold_tokens)
    # bonus for exact substring
    if gold.lower() in ans.lower():
        return 5
    # map to 0-5
    if overlap >= 0.9: return 5
    if overlap >= 0.7: return 4
    if overlap >= 0.5: return 3
    if overlap >= 0.3: return 2
    if overlap > 0.0: return 1
    return 0

def evaluate(compressor_fn, fixture: str, probes: List[Probe]) -> Dict:
    comp = compressor_fn(fixture)
    ratio = len(fixture)/max(1,len(comp))
    by_dim = {}
    total = 0
    rows = []
    for p in probes:
        ans = simple_qa(comp, p.q)
        score = em_score(p.gold, ans)
        by_dim.setdefault(p.dim, []).append(score)
        total += score
        rows.append({"id":p.id,"dim":p.dim,"q":p.q,"gold":p.gold,"ans":ans[:120],"score":score})
    avg = total/len(probes) if probes else 0
    by_dim_avg = {k: sum(v)/len(v) for k,v in by_dim.items()}
    return {"ratio": ratio, "avg": avg, "by_dim": by_dim_avg, "compressed": comp, "rows": rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad", action="store_true", help="use bad compressor to show drop")
    args = parser.parse_args()

    fn = bad_compressor if args.bad else naive_compressor
    res = evaluate(fn, SAMPLE_FIXTURE, PROBES)
    print(f"Compression ratio: {res['ratio']:.2f}x")
    print(f"Avg quality: {res['avg']:.2f}/5")
    print("By dim:", json.dumps(res["by_dim"], indent=2, ensure_ascii=False))
    print("\n-- compressed preview --")
    print(res["compressed"][:800])
    print("\n-- probe details --")
    for r in res["rows"]:
        print(f"{r['id']:>2} [{r['dim'][:4]}] score={r['score']} Q={r['q'][:50]} ans={r['ans'][:60]}")

if __name__ == "__main__":
    main()
