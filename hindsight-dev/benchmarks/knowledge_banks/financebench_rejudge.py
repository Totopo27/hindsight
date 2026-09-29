"""Re-grade a finished FinanceBench run the way PageIndex graded Mafin 2.5.

VectifyAI/Mafin2.5-FinanceBench ``eval.py``: the same prompt on three judges
(gpt-4o-2024-11-20, o1-mini, o3-mini), and an answer is correct when ANY of them says so
(``judge_benchmark_results_from_file_hybrid``). Their headline also used human
re-annotation of ambiguous questions, which this does not reproduce.

    uv run python scripts/financebench_rejudge.py outputs/financebench/<run>/<mode>/<split>.json

Prints the single-judge (gpt-4o) and three-judge-OR accuracy, overall and per category,
and writes the verdicts next to the run as ``*.rejudge.json``.
"""

import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

from memory_bench.dataset.financebench import _JUDGE

JUDGES = ["gpt-4o-2024-11-20", "o1-mini", "o3-mini"]


async def verdict(client: AsyncOpenAI, sem: asyncio.Semaphore, model: str, query: str, answer: str, gold: str) -> bool | None:
    # Their prompt asks for a bare True/False; the dataset's copy ends with a JSON hint for
    # AMB's schema'd judge, so the original last line is restored here.
    prompt = _JUDGE.format(query=query, answer=answer, gold=gold).replace(
        'Set "correct" to true or false by the criteria above, and give a one-line reason.',
        "Your output should be ONLY a boolean value: `True` or `False`, nothing else.",
    )
    kwargs = {} if model.startswith("o") else {"temperature": 0}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], **kwargs
                )
                text = (r.choices[0].message.content or "").lower()
                if "true" in text:
                    return True
                if "false" in text:
                    return False
                return None
            except Exception as e:  # a retired model or a rate limit: record, do not crash
                if attempt == 2:
                    print(f"{model}: {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 * (attempt + 1))
    return None


async def main(path: Path) -> None:
    run = json.loads(path.read_text())
    rows = run["results"]
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(16)
    verdicts: dict[str, list[bool | None]] = {}
    for model in JUDGES:
        verdicts[model] = await asyncio.gather(
            *(verdict(client, sem, model, r["query"], r["answer"], r["gold_answers"][0]) for r in rows)
        )
    single = [bool(v) for v in verdicts[JUDGES[0]]]
    hybrid = [any(bool(verdicts[m][i]) for m in JUDGES) for i in range(len(rows))]
    unavailable = [m for m in JUDGES if all(v is None for v in verdicts[m])]

    def report(label: str, marks: list[bool]) -> None:
        by_cat: dict[str, list[bool]] = {}
        for r, ok in zip(rows, marks):
            by_cat.setdefault(r["meta"].get("category", "?"), []).append(ok)
        cats = "  ".join(f"{c}={sum(v)}/{len(v)}" for c, v in sorted(by_cat.items()))
        print(f"{label:28} {sum(marks)}/{len(marks)} = {100 * sum(marks) / len(marks):.1f}%   {cats}")

    print(f"run: {path}  (AMB judge: {run.get('correct')}/{run.get('total_queries')})")
    report(f"single judge ({JUDGES[0]})", single)
    report("any of 3 judges (PageIndex)", hybrid)
    if unavailable:
        print(f"judges unavailable (counted as False): {unavailable}")
    out = path.with_suffix(".rejudge.json")
    out.write_text(json.dumps({"judges": JUDGES, "verdicts": verdicts, "single": single, "hybrid": hybrid}, indent=1))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
