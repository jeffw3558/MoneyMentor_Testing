import re, json, os, boto3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------- basic rules --------
_model = SentenceTransformer("all-MiniLM-L6-v2")

def _kw_ok(reply: str, kws: str) -> bool:
    """All semi-colon keywords must appear (case-insensitive)."""
    if not kws: return True
    return all(re.search(k.strip(), reply, re.I) for k in kws.split(";") if k)

def _sim_ok(reply: str, gold: str, thr: float = 0.85) -> bool:
    if not gold: return True
    sim = cosine_similarity(_model.encode([reply]),
                            _model.encode([gold]))[0][0]
    return sim >= thr

# -------- Bedrock judge --------
REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL  = os.getenv("BEDROCK_MODEL",
                   "anthropic.claude-3-haiku-20240307-v1:0")
_br = boto3.client("bedrock-runtime", region_name=REGION)

def _llm_judge(prompt: str, reply: str) -> bool:
    system = ("You are an evaluation assistant.\n"
              "Return JSON {\"pass\": true/false}.")
    body = {
        "messages":[
            {"role":"system","content":system},
            {"role":"user","content":f"USER: {prompt}\nBOT: {reply}"}
        ],
        "temperature":0.0, "max_tokens":50
    }
    resp = _br.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    text = json.loads(resp["body"].read())["content"][0]["text"]
    try:
        return json.loads(text).get("pass", False)
    except Exception:
        return False

def grade(reply:str, prompt:str,
          keywords:str, expected:str, use_llm:bool)->bool:
    basic = _kw_ok(reply, keywords) and _sim_ok(reply, expected)
    if not use_llm or not prompt:       # no judge requested
        return basic
    return _llm_judge(prompt, reply)    # may override
