import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI
from tqdm.auto import tqdm

# Config
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/private.jsonl"
RESULTS_DIR = Path("results")

MAX_TOKENS = 4096

# Retry output budgets. Pass 0 intentionally stays at MAX_TOKENS.
# Pass 1 and Pass 2 get larger budgets so truncated reasoning can finish.
MAX_TOKENS_PASS1 = 8192
MAX_TOKENS_PASS2 = 12288
MAX_TOKENS_RETRY = 1024  # kept for compatibility; retry configs below use PASS1/PASS2.

# Try 12 or 16 once stable.
CONCURRENCY = 8

ITERATIONS = 2

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=1200,
)

# Prompts
# Shared final-answer formatting instructions.
ANSWER_FORMATTING_BLURB = (
    "Put the final answer in \\boxed{}. "
    "For MCQ, output only the option letter, e.g. \\boxed{C}. "
    "For multiple blanks or subparts, put answers in the order asked, "
    "separated by commas inside one box, e.g. \\boxed{3, 7, B}. "
    "For select-all-that-apply, list all chosen letters comma-separated, "
    "e.g. \\boxed{A, C, D}. "
    "Do not include explanations, units, or option text inside the box unless required."
)

DECIMAL_ROUNDING_BLURB = (
    " If the final answer is a decimal and the problem statement does not "
    "specify a rounding precision, round the final answer to 6 decimal places. "
    "If the problem statement specifies a number of decimal places, significant "
    "figures, or another rounding rule, the problem statement takes priority."
)

SYSTEM_MATH_NORMAL = (
    "Solve the problem inside <think> tags. "
    "Do only the reasoning needed to determine the final answer. "
    "Once a final answer is reached, stop reasoning. "
    "After </think>, output only the final answer. "
    "The text after </think> must contain a boxed answer. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MCQ_NORMAL = (
    "Solve the multiple-choice problem inside <think> tags. "
    "Do only the reasoning needed to identify the correct option. "
    "Once one option is determined to be correct, stop reasoning. "
    "After </think>, output only the final answer. "
    "Do not write anything else after </think>. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MATH_DECISIVE = (
    "Your goal is to reach a final boxed answer, not to write a long proof. "
    "Solve inside <think> tags. "
    "As soon as a final answer is found, stop reasoning immediately. "
    "Do not keep checking alternate cases, do not revisit earlier steps, and do not continue explaining. "
    "After </think>, output only the final boxed answer. "
    "The text after </think> must contain exactly the final boxed answer. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MCQ_DECISIVE = (
    "Your goal is to choose the correct multiple-choice option. "
    "Solve inside <think> tags. "
    "As soon as one option is determined to be correct, stop reasoning immediately. "
    "Ignore the remaining options unless they are needed to identify the answer. "
    "Do not do extra checks, do not test different cases, and do not continue explaining. "
    "After </think>, output only the final boxed answer. "
    "Nothing else after </think>. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MATH_SHORT = (
    "Solve briefly inside <think>. "
    "Stop as soon as the final answer is known. "
    "After </think>, write only the final boxed answer. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MCQ_SHORT = (
    "Choose the correct option briefly inside <think>. "
    "Stop as soon as the correct letter is known. "
    "After </think>, write only the final boxed answer. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MATH_ULTRA_SHORT = (
    "Find the answer. Use minimal reasoning inside <think>. "
    "Then output only the final boxed answer after </think>. "
    + ANSWER_FORMATTING_BLURB
)

SYSTEM_MCQ_ULTRA_SHORT = (
    "Find the correct option. Use minimal reasoning inside <think>. "
    "Then output only the final boxed answer after </think>. "
    + ANSWER_FORMATTING_BLURB
)


def build_prompt(
    question: str,
    options: Optional[list],
    prompt_style: str = "normal",
    round_decimal_answer: bool = False,
) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_str = "\n".join(
            f"{label}. {option.strip()}"
            for label, option in zip(labels, options)
        )

        if prompt_style == "decisive":
            sys_p = SYSTEM_MCQ_DECISIVE
        elif prompt_style == "short":
            sys_p = SYSTEM_MCQ_SHORT
        elif prompt_style == "ultra_short":
            sys_p = SYSTEM_MCQ_ULTRA_SHORT
        else:
            sys_p = SYSTEM_MCQ_NORMAL

        if round_decimal_answer:
            sys_p += DECIMAL_ROUNDING_BLURB

        return sys_p, f"{question}\n\nOptions:\n{opts_str}"

    if prompt_style == "decisive":
        sys_p = SYSTEM_MATH_DECISIVE
    elif prompt_style == "short":
        sys_p = SYSTEM_MATH_SHORT
    elif prompt_style == "ultra_short":
        sys_p = SYSTEM_MATH_ULTRA_SHORT
    else:
        sys_p = SYSTEM_MATH_NORMAL

    if round_decimal_answer:
        sys_p += DECIMAL_ROUNDING_BLURB

    return sys_p, question


# Answer extraction: mirrors judger.py behavior
def _extract_all_boxed(text: str) -> list[str]:
    """
    Returns the last contiguous group of \\boxed{} contents.
    """
    entries = []
    start = 0

    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break

        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content:
                entries.append((idx, i, content))

        start = i

    if not entries:
        return []

    last_group = [entries[-1]]

    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]

        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break

    return [e[2] for e in last_group]


def judger_extract(text: str) -> str | None:
    """
    Approximate mirror of judger.extract_ans + extract_explicit_ans.
    Returns extracted answer string, or None if nothing is found.
    """
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    working = text

    if "herefore" in working:
        working = working.split("herefore")[-1].strip()

    if "oxed{" in search_text:
        boxed = _extract_all_boxed(search_text)
        if boxed:
            return ", ".join(boxed) if len(boxed) > 1 else boxed[0]

    for marker in ["answer is", "answer:", "answer :"]:
        if marker in text.lower():
            idx = text.lower().rfind(marker)
            after_orig = text[idx + len(marker):].strip()
            if after_orig:
                return after_orig[:200]

    matches = re.findall(r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\)|\\\])", text, re.DOTALL)
    if matches:
        return matches[-1]

    matches = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    if matches:
        return matches[-1]

    return None


def has_valid_answer(text: str) -> bool:
    result = judger_extract(text)
    return bool(result and result.strip())


def has_boxed_after_think(text: str) -> bool:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    return bool(_extract_all_boxed(search_text))


def invalid_ids(responses: dict[int, str]) -> list[int]:
    return [
        item_id
        for item_id, resp in responses.items()
        if not has_boxed_after_think(resp)
    ]


def unscorable_ids(responses: dict[int, str]) -> list[int]:
    return [
        item_id
        for item_id, resp in responses.items()
        if not has_valid_answer(resp)
    ]


DECIMAL_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9])"
)


def answer_has_decimal(text: str) -> bool:
    answer = judger_extract(text)
    if not answer:
        return False

    # Match decimal numeric answers such as 0.25, .25, 3., or 1.2e-4.
    return bool(DECIMAL_NUMBER_RE.search(answer.replace(",", "")))


def decimal_answer_ids(responses: dict[int, str]) -> list[int]:
    return [
        item_id
        for item_id, resp in responses.items()
        if answer_has_decimal(resp)
    ]


# vLLM call helpers
async def call_one(system: str, user: str, cfg: dict, item_id: int | None = None) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error = None

    for attempt in range(3):
        try:
            result = await client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                extra_body={
                    "top_k": cfg["top_k"],
                    "repetition_penalty": cfg["repetition_penalty"],
                },
            )

            content = result.choices[0].message.content

            if content is None:
                return ""

            return content.strip()

        except Exception as e:
            last_error = e
            await asyncio.sleep(2 * (attempt + 1))

    return f"ERROR item_id={item_id}: {type(last_error).__name__}: {last_error}"


async def run_items(
    items: list[dict],
    cfg: dict,
    desc: str,
    csv_path: Path | None = None,
    existing_rows: dict[int, str] | None = None,
    on_result: Callable[[int, str], None] | None = None,
) -> dict[int, str]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: dict[int, str] = dict(existing_rows or {})

    if csv_path is not None:
        write_csv_atomic(csv_path, results)

    async def worker(item: dict) -> tuple[int, str]:
        async with sem:
            system, user = build_prompt(
                question=item["question"],
                options=item.get("options"),
                prompt_style=cfg["prompt_style"],
                round_decimal_answer=cfg.get("round_decimal_answer", False),
            )

            response = await call_one(
                system=system,
                user=user,
                cfg=cfg,
                item_id=item["id"],
            )

            return item["id"], response

    tasks = [asyncio.create_task(worker(item)) for item in items]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc, unit="q"):
        item_id, response = await fut
        results[item_id] = response

        if csv_path is not None:
            # Persist each completed generation immediately.
            # CSV rows can vary in length, so the safest "line update" is an
            # atomic rewrite of the file with this row replaced/added.
            write_csv_atomic(csv_path, results)

        if on_result is not None:
            on_result(item_id, response)

    return results


# CSV helpers
def write_csv_atomic(path: Path, rows: dict[int, str]) -> None:
    """
    Rewrite the CSV with the latest rows, then atomically replace the old file.

    This lets us update the CSV after every completed generation without risking
    a half-written file if the process is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"])

        for item_id in sorted(rows):
            writer.writerow([item_id, rows[item_id]])

    tmp_path.replace(path)


def save_csv(path: Path, rows: dict[int, str]) -> None:
    write_csv_atomic(path, rows)
    print(f"  → saved {len(rows)} rows to {path}")


def load_csv(path: Path) -> dict[int, str]:
    results: dict[int, str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results[int(row["id"])] = row["response"]

    return results


async def check_vllm_server() -> None:
    """Fail early with a useful message if the OpenAI-compatible vLLM server is unavailable."""
    try:
        await client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to the local vLLM server at http://localhost:8000/v1. "
            "Start the server from the README before calling run_inference()."
        ) from exc


# Main
async def _run_inference_async() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    await check_vllm_server()

    with open(DATA_PATH, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    by_id = {item["id"]: item for item in data}
    all_ids = list(by_id.keys())

    pass_configs = [
        # Pass 0: normal attempt.
        {
            "max_tokens": MAX_TOKENS,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.00,
            "prompt_style": "normal",
            "round_decimal_answer": False,
        },

        # Pass 1: same solving style as pass 0, but with a larger output budget.
        # This targets pass-0 responses that were cut off before </think> + \boxed{}.
        {
            "max_tokens": MAX_TOKENS_PASS1,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.00,
            "prompt_style": "normal",
            "round_decimal_answer": False,
        },

        # Pass 2: one more larger-window retry for anything still lacking a valid box.
        # Lower temperature makes this pass a little more stable while preserving reasoning room.
        {
            "max_tokens": MAX_TOKENS_PASS2,
            "temperature": 0.35,
            "top_p": 0.90,
            "top_k": 10,
            "repetition_penalty": 1.05,
            "prompt_style": "normal",
            "round_decimal_answer": False,
        },

        # Pass 3: same generation settings and prompt style as pass 2, but with
        # an added rounding instruction. Only run on rows whose parsed answer
        # currently contains a decimal.
        {
            "max_tokens": MAX_TOKENS_PASS2,
            "temperature": 0.35,
            "top_p": 0.90,
            "top_k": 10,
            "repetition_penalty": 1.05,
            "prompt_style": "normal",
            "round_decimal_answer": True,
        },
    ]

    master: dict[int, str] = {}
    final_path = RESULTS_DIR / "submission_final.csv"

    # Pass 0
    pass0_path = RESULTS_DIR / "submission_pass0.csv"

    if pass0_path.exists():
        print(f"Pass 0 exists, loading {pass0_path}")
        master = load_csv(pass0_path)

    pass0_missing = [item_id for item_id in all_ids if item_id not in master]

    if pass0_missing:
        print(f"\n{'=' * 60}")
        print(f"Pass 0 — {len(pass0_missing)} questions")
        print(f"Concurrency = {CONCURRENCY}")
        print(f"{'=' * 60}")

        def accept_pass0_result(item_id: int, response: str) -> None:
            master[item_id] = response
            write_csv_atomic(final_path, master)

        pass0_items = [by_id[item_id] for item_id in pass0_missing]
        rows = await run_items(
            pass0_items,
            pass_configs[0],
            "Pass 0",
            csv_path=pass0_path,
            existing_rows=master,
            on_result=accept_pass0_result,
        )
        master.update(rows)
        save_csv(pass0_path, rows)
    else:
        print("Pass 0 complete — no missing rows")

    write_csv_atomic(final_path, master)

    no_boxed = invalid_ids(master)
    no_score = unscorable_ids(master)

    print(f"Pass 0 done: {len(no_boxed)} without \\boxed{{}} after </think>")
    print(f"             {len(no_score)} truly unscorable")
    print(f"             {len(no_boxed) - len(no_score)} saved by judger fallbacks")

    # Strict-format mode: retry everything without boxed answer.
    # Faster score-only mode: use bad = no_score instead.
    bad = no_boxed

    # Retry passes
    for pass_num in range(1, ITERATIONS + 1):
        if not bad:
            print(f"\nAll answered — stopping at pass {pass_num - 1}")
            break

        cfg = pass_configs[min(pass_num, len(pass_configs) - 1)]
        pass_path = RESULTS_DIR / f"submission_pass{pass_num}.csv"

        print(f"\n{'=' * 60}")
        print(f"Pass {pass_num} — {len(bad)} IDs to retry")
        print(f"cfg = {cfg}")
        print(f"Concurrency = {CONCURRENCY}")
        print(f"{'=' * 60}")

        retry_ids = list(bad)

        if pass_path.exists():
            print(f"  Exists, loading {pass_path}")
            rows = load_csv(pass_path)
        else:
            rows = {}

        def maybe_accept_retry_result(item_id: int, response: str) -> None:
            old = master.get(item_id, "")

            if has_boxed_after_think(response):
                master[item_id] = response
                write_csv_atomic(final_path, master)
            elif has_valid_answer(response) and not has_valid_answer(old):
                master[item_id] = response
                write_csv_atomic(final_path, master)
            elif not old:
                master[item_id] = response
                write_csv_atomic(final_path, master)

        missing_retry_ids = [item_id for item_id in retry_ids if item_id not in rows]

        if missing_retry_ids:
            retry_items = [by_id[item_id] for item_id in missing_retry_ids]
            rows = await run_items(
                retry_items,
                cfg,
                f"Pass {pass_num}",
                csv_path=pass_path,
                existing_rows=rows,
                on_result=maybe_accept_retry_result,
            )
            save_csv(pass_path, rows)
        else:
            print("  Pass CSV complete — no missing rows")

        for item_id in retry_ids:
            response = rows.get(item_id)
            if response is None:
                continue

            old = master.get(item_id, "")

            if has_boxed_after_think(response):
                master[item_id] = response
            elif has_valid_answer(response) and not has_valid_answer(old):
                master[item_id] = response
            elif not old:
                master[item_id] = response

        write_csv_atomic(final_path, master)

        bad = invalid_ids(master)
        no_score = unscorable_ids(master)

        print(f"  After pass {pass_num}: {len(bad)} no \\boxed{{}} | {len(no_score)} truly unscorable")

        if bad[:20]:
            suffix = "..." if len(bad) > 20 else ""
            print(f"  Remaining IDs: {bad[:20]}{suffix}")

    # Decimal rounding pass: after pass 2, rerun any row whose parsed answer
    # contains a decimal. This uses pass-2 generation settings plus an instruction
    # to round to 6 decimal places unless the problem statement says otherwise.
    decimal_ids = decimal_answer_ids(master)
    decimal_pass_num = ITERATIONS + 1

    if decimal_ids:
        cfg = pass_configs[3]
        pass_path = RESULTS_DIR / f"submission_pass{decimal_pass_num}_round6.csv"

        print(f"\n{'=' * 60}")
        print(f"Pass {decimal_pass_num} decimal rounding — {len(decimal_ids)} IDs to retry")
        print(f"cfg = {cfg}")
        print(f"Concurrency = {CONCURRENCY}")
        print(f"{'=' * 60}")

        if pass_path.exists():
            print(f"  Exists, loading {pass_path}")
            rows = load_csv(pass_path)
        else:
            rows = {}

        def maybe_accept_decimal_result(item_id: int, response: str) -> None:
            old = master.get(item_id, "")

            if has_boxed_after_think(response):
                master[item_id] = response
                write_csv_atomic(final_path, master)
            elif has_valid_answer(response) and not has_boxed_after_think(old):
                master[item_id] = response
                write_csv_atomic(final_path, master)
            elif not old:
                master[item_id] = response
                write_csv_atomic(final_path, master)

        missing_decimal_ids = [item_id for item_id in decimal_ids if item_id not in rows]

        if missing_decimal_ids:
            decimal_items = [by_id[item_id] for item_id in missing_decimal_ids]
            rows = await run_items(
                decimal_items,
                cfg,
                f"Pass {decimal_pass_num} round6",
                csv_path=pass_path,
                existing_rows=rows,
                on_result=maybe_accept_decimal_result,
            )
            save_csv(pass_path, rows)
        else:
            print("  Decimal rounding pass CSV complete — no missing rows")

        for item_id in decimal_ids:
            response = rows.get(item_id)
            if response is None:
                continue

            old = master.get(item_id, "")

            if has_boxed_after_think(response):
                master[item_id] = response
            elif has_valid_answer(response) and not has_boxed_after_think(old):
                master[item_id] = response
            elif not old:
                master[item_id] = response

        write_csv_atomic(final_path, master)

        bad = invalid_ids(master)
        no_score = unscorable_ids(master)

        print(f"  After pass {decimal_pass_num}: {len(bad)} no \\boxed{{}} | {len(no_score)} truly unscorable")
    else:
        print("\nDecimal rounding pass skipped — no parsed decimal answers")

    # Final CSV
    save_csv(final_path, master)

    final_no_boxed = invalid_ids(master)
    final_no_score = unscorable_ids(master)

    print(f"\n{'=' * 60}")
    print(f"FINAL: {len(all_ids) - len(final_no_boxed)}/{len(all_ids)} have \\boxed{{}} after </think>")
    print(f"       {len(all_ids) - len(final_no_score)}/{len(all_ids)} scoreable by judger")
    print(f"Final submission → {final_path}")


def run_inference() -> None:
    """
    Single public entry point required for the final code submission.

    This synchronous wrapper lets graders reproduce the full pipeline with:

        from TAKE2 import run_inference
        run_inference()

    The async implementation is kept private so the existing concurrent
    inference pipeline does not need to change.
    """
    asyncio.run(_run_inference_async())


if __name__ == "__main__":
    run_inference()
