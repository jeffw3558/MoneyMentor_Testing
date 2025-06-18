import asyncio, httpx, yaml, pandas as pd, time, json
from pathlib import Path
from tqdm.asyncio import tqdm
from qa.validator import grade


ACCTS = yaml.safe_load(Path("config/accounts.yaml").read_text())
DF    = pd.read_csv("prompts/tests.csv").fillna("")

# -------- organise by account & conversation --------
tests = {}
for row in DF.to_dict("records"):
    acct = row["account"] or list(ACCTS.keys())[0]   # default acct
    cid  = row["conversation_id"] or f'cid-{int(time.time()*1e6)}-{acct}'
    tests.setdefault(acct, {}) \
         .setdefault(cid, []) \
         .append(row)
for conv in tests.values():
    for turns in conv.values():             # ensure turn order
        turns.sort(key=lambda r: int(r["turn"]))

# -------- async caller --------
async def run_conv(acct_key, cid, rows):
    conf = ACCTS[acct_key]
    headers = {
    "origin": "qa-harness",  # or whatever your backend expects
    "Authorization": f"Bearer {conf['session_token']}",
    "Content-Type": "application/json"
}

    out=[]
    async with httpx.AsyncClient(timeout=30) as cli:
        for r in rows:
            t0=time.perf_counter()
            resp = await cli.post(
                conf["base_url"],
                headers=headers,
                json={
                    "conversation_id": cid,
                    "prompt": r["prompt"]          # ← change “message” → “prompt”
                 })

            latency_ms=int((time.perf_counter()-t0)*1000)
            data = resp.json()
            if "message" not in data:
                print(f"[ERROR] API call failed: {data}")

            passed = grade(
                reply     = data["message"],
                prompt    = r["prompt"],
                keywords  = r["keywords"],
                expected  = r["expected"],
                use_llm   = bool(r["use_llm"])
            )
            out.append({
                "id": f'{acct_key}:{cid}:{r["turn"]}',
                "prompt": r["prompt"],
                "reply": data["message"],
                "pass": passed,
                "latency_ms": latency_ms,
                "tokens": data.get("tokens_used_message",0)
            })
    return out

async def main():
    tasks=[run_conv(a,c,rows)
           for a, convs in tests.items()
           for c, rows in convs.items()]
    results=[item for batch in await tqdm.gather(*tasks)
                   for item in batch]
    Path("reports").mkdir(exist_ok=True)
    fname=time.strftime("reports/%Y-%m-%dT%H-%M-%S.json")
    Path(fname).write_text(json.dumps(results,indent=2))
    total=len(results); passed=sum(1 for r in results if r["pass"])
    print(f"✅ {passed}/{total} passed  ·  report → {fname}")

if __name__=="__main__":
    asyncio.run(main())
