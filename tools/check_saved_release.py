"""Run saved-output figure/table regeneration on an isolated CPU-only copy."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    tables = ['tables.json', 'supplement-tables.json']
    expected = {name: (ROOT / 'paper/data' / name).read_bytes() for name in tables}
    with tempfile.TemporaryDirectory(prefix='microcalcification-cpu-') as temp:
        out = Path(temp) / 'source'
        shutil.copytree(ROOT, out, ignore=shutil.ignore_patterns('.git', '.cache', '__pycache__', '.venv'))
        audit_path = Path('results/publication-task-audit/paired-task-audit.json')
        original = json.loads((out / audit_path).read_text())
        subprocess.run([sys.executable, 'publication_task_audit.py', '--reproduce-saved'], cwd=out, check=True, stdout=subprocess.PIPE, text=True)
        regenerated = json.loads((out / audit_path).read_text())
        # Clock provenance differs; all numerical values and source hashes must match.
        del original['at_utc'], regenerated['at_utc']
        assert regenerated == original, 'saved-array paired audit mismatch'
        subprocess.run([sys.executable, 'paper/build_figures_tables.py'], cwd=out, check=True)
        for name, raw in expected.items():
            assert (out / 'paper/data' / name).read_bytes() == raw, name + ' byte mismatch'
        supplement = json.loads((out / 'paper/data/supplement-tables.json').read_text())
        assert [len(table['rows']) for table in supplement] == [24, 72]
        for number in range(2, 5):
            for ext in ['png', 'pdf']:
                target = out / f'paper/figures/Figure{number}.{ext}'
                assert target.stat().st_size > 1000, str(target)
        print(json.dumps({'cpu_regeneration': 'PASS', 'gpu_calls': 0,
                          'byte_identical_tables': {n: hashlib.sha256(v).hexdigest() for n, v in expected.items()}}, indent=2))


if __name__ == '__main__':
    main()
