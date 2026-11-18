"""Local serial continuation: wait for generator, train, evaluate, report. No automation service."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import study_budget as budget

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/full-study'

def main():
    freeze=json.loads((OUT/'analysis-freeze.json').read_text())
    for name,sha in freeze['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha: raise RuntimeError('Analysis implementation changed: '+name)
    while not (OUT/'generation-complete.json').exists():
        budget.check(reserve=3600)
        # A stale ledger is an execution failure, never a negative scientific result.
        ledger=OUT/'resource.jsonl'
        if ledger.exists() and time.time()-ledger.stat().st_mtime>900: raise RuntimeError('Generator has stopped/stalled; inspect logs before resuming')
        time.sleep(30)
    for name in ('train_full_study.py','evaluate_full_study.py','report_full_study.py'):
        with (OUT/(name.removesuffix('.py')+'.log')).open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    (OUT/'pipeline-complete.json').write_text(json.dumps({'status':'COMPLETED','device_hours':budget.consumed()/3600},indent=2))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        (OUT/'pipeline-error.json').write_text(json.dumps({'status':'TECHNICAL_UNEVALUATED','error':repr(exc)},indent=2))
        raise
