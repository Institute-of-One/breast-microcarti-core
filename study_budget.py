"""Append-only compute allocation ledger; no external calls."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'results/full-study'


def entries():
    p=FOLDER/'resource.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def consumed(): return 1800+sum(r['device_seconds'] for r in entries())


def check(reserve=0,ongoing=0):
    if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-11-18T18:00:00+09:00'):
        raise TimeoutError('Calendar deadline reached')
    if consumed()+ongoing+reserve>=86400:
        raise TimeoutError('24 device-hour total cap/reserved completion budget reached')


def record(phase,seconds,**metadata):
    FOLDER.mkdir(exist_ok=True)
    row={'at_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'device_seconds':seconds,**metadata}
    with (FOLDER/'resource.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
    return row
