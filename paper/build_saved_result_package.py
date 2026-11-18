"""Assemble a deterministic, local-only Online Resource 2 without model execution."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    names = ['README.md', 'LICENSE', 'requirements.txt', 'pilot.py', 'cluster_physics.py',
             'cluster_fast.py', 'gpu_reconstruct.py', 'study_reconstruct.py', 'study_models.py',
             'study_budget.py', 'generate_full_study.py', 'train_full_study.py',
             'evaluate_full_study.py', 'publication_task_audit.py',
             'test_publication_task_audit.py', 'configs/pilot-v1.json',
             'docs/PUBLICATION_SCOPE_PLAN_20261115.md',
             'paper/build_figures_tables.py', 'paper/build_saved_result_package.py',
             'paper/supplement_tables.py',
             'paper/FIGURE_PROVENANCE.md', 'paper/sources/clinical-illustration-provenance.json',
             'paper/data/tables.json', 'paper/data/supplement-tables.json',
             'results/publication-task-audit/paired-task-audit.json']
    names += ['results/full-study/' + n for n in ['evaluation.json', 'phantoms.json',
              'sr-localization-counts.npy', 'direct-test-predictions.npy',
              'enhanced-test-predictions.npy', 'xray_sr-test-predictions.npy',
              'normalization.json', 'freeze.json', 'analysis-freeze.json', 'final-tests.json',
              'final-integrity.json', 'resource.jsonl', 'interruption-20261115.json']]
    names += [f'paper/figures/Figure{i}.{ext}' for i in range(1, 5) for ext in ['png', 'pdf']]
    manifest = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    instructions = '''Online Resource 2 saved-result reproducibility

Software: MIT (LICENSE). Original documentation, numerical outputs and figures:
CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . Figure1 is adapted
from Loizidou et al., Zenodo version3, DOI10.5281/zenodo.14859694, CC BY4.0;
see paper/FIGURE_PROVENANCE.md for attribution and modifications.

This archive reproduces analyses from saved results, not raw RF/model training.
Install requirements.txt in a separate environment. CPU checks:
  python -m unittest test_publication_task_audit.py -v
Numerical figure/table generation from saved JSON:
  python paper/build_figures_tables.py
The latter overwrites generated figure/table files and appends CPU timing locally.
Use an extracted copy. It does not execute a neural network or GPU simulation.
Forward/training code is supplied for method inspection; do not run those files
to reproduce the saved-result paper summaries. PyTorch is needed for new training,
but is not needed by the two commands above and is not installed automatically.

Original clinical DICOM, raw RF and reconstructed image arrays, model checkpoints,
credentials, private AI reviews and human approval records are excluded.
Known experimental accounting is17.18+U device-hours; U remains unknown.
No Git/CI/public URL or DOI for this study is claimed. This is a local submission
artifact and has not been uploaded, published or submitted.
'''
    target = ROOT/'paper/Online_Resource_2.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        contents = {n: (ROOT/n).read_bytes() for n in names}
        contents['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
        contents['REPRODUCING.txt'] = instructions.encode()
        for name, data in contents.items():
            info = ZipInfo(name, (2026, 11, 16, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, data)
    with ZipFile(target) as z:
        assert z.testzip() is None
        for n, h in manifest.items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h
        assert not any(n.endswith('.dcm') or '/reviews/' in n or 'credentials' in n
                       for n in z.namelist())
    print({'files': len(manifest)+2, 'bytes': target.stat().st_size,
           'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})

if __name__ == '__main__':
    main()
