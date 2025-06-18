"""
Usage:
python excel_to_txt.py questions.xlsx
→ writes prompts/tests.csv (account column blank)
"""
import sys, pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python excel_to_txt.py <questions.xlsx>")
    sys.exit(1)

src = Path(sys.argv[1])
df  = pd.read_excel(src)
df["account"] = ""
df["conversation_id"] = ""
df["turn"] = range(1, len(df) + 1)
df["keywords"] = ""
df["expected"] = ""
df["use_llm"] = 1
df[["account","conversation_id","turn","prompt",
    "keywords","expected","use_llm"]].to_csv("prompts/tests.csv", index=False)
print("Wrote prompts/tests.csv with", len(df), "rows")
