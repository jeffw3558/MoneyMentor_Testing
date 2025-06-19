# qa/runner.py
import asyncio, httpx, yaml, pandas as pd, time, json
from pathlib import Path
from tqdm.asyncio import tqdm
from qa.validator import grade  # your existing grading logic

# ---------- load accounts & test prompts ----------
ACCTS = yaml.safe_load(Path("config/accounts.yaml").read_text())
DF    = pd.read_csv("prompts/tests.csv").fillna("")

# group rows by account → (placeholder)conversation → turn
tests: dict[str, dict[str, list]] = {}
for row in DF.to_dict("records"):
    acct = row["account"] or list(ACCTS.keys())[0]
    cid  = row.get("conversation_id") or "placeholder"
    tests.setdefault(acct, {}).setdefault(cid, []).append(row)

for conv in tests.values():                       # keep turn order
    for turns in conv.values():
        turns.sort(key=lambda r: int(r["turn"]))


# ---------- helper: fetch (or create) conversation ID ----------
async def get_conversation_id(cli: httpx.AsyncClient, base_url: str, headers: dict):
    conv_url = base_url.replace("/chat/messages", "/chat/conversations")
    resp     = await cli.get(conv_url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data[0]["id"] if data else None


# ---------- run one conversation ----------
async def run_conv(acct_key: str, cid: str, rows: list[dict]):
    conf    = ACCTS[acct_key]
    headers = {
    "X-Host-Origin": "dev-chat-api.c64f.com",   # ← exactly as Swagger shows
    "sessionToken":  conf["session_token"],     # capital T
    "Content-Type":  "application/json"
}


    results = []
    async with httpx.AsyncClient(timeout=30) as cli:
        cid = await get_conversation_id(cli, conf["base_url"], headers) or "new_conv"

        for r in rows:
            t0   = time.perf_counter()
            resp = await cli.post(
                conf["base_url"],
                headers=headers,
                json={"conversation_id": cid, "message": r["prompt"]}
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            if resp.status_code != 200:
                data   = {"message": f"[HTTP {resp.status_code}] {resp.text}"}
            else:
                data   = resp.json()

            passed = grade(
                reply    = data.get("message", ""),
                prompt   = r["prompt"],
                keywords = r["keywords"],
                expected = r["expected"],
                use_llm  = bool(r["use_llm"])
            )

            results.append({
                "id"        : f"{acct_key}:{cid}:{r['turn']}",
                "prompt"    : r["prompt"],
                "reply"     : data.get("message", ""),
                "pass"      : passed,
                "latency_ms": latency_ms,
                "tokens"    : data.get("tokens_used_message", 0)
            })
    return results


# ---------- main ----------
async def main():
    tasks = [
        run_conv(acct, cid, rows)
        for acct, convs in tests.items()
        for cid,  rows in convs.items()
    ]

    # tqdm.gather shows progress bar √
    results = [item for batch in await tqdm.gather(*tasks) for item in batch]

    Path("reports").mkdir(exist_ok=True)
    fname = time.strftime("reports/%Y-%m-%dT%H-%M-%S.json")
    Path(fname).write_text(json.dumps(results, indent=2))

    passed = sum(r["pass"] for r in results)
    print(f"✅ {passed}/{len(results)} passed – report saved to {fname}")

if __name__ == "__main__":
    asyncio.run(main())
