# Ultrasound microcalcification cluster imaging: detection and localization

Saved-result reproducibility companion to **Detection and localization dissociation
in physics simulated ultrasound imaging of microcalcification clusters**.

Version: **0.1.0**. Author: Shuji Yamamoto, Institute of One, LISIT Co., Ltd.,
Tokyo 150-0044, Japan. ORCID: https://orcid.org/0000-0001-9211-1071 .

## Scope and findings

This exploratory computational study compares direct ultrasound, conventional
enhancement, and a two-channel ultrasound-plus-learned-occupancy-map pathway
(X-ray-style SR). No actual mammogram is used as model input or training teacher.
There are 480 synthetic phantoms and eight acquisition/noise conditions, producing
3,840 reconstructed maps. Splits are disjoint by phantom. These are 2D simulations,
not clinical validation, FWI, or a proof of a fundamental detectability limit.

At 1 MHz/full-view/40 dB reference SNR, all 60 positive test phantoms were detected
but no particle was correctly localized at the audited thresholds. At 3 MHz under
the corresponding favorable condition, 268/300 particles were matched, with one
unmatched candidate across 90 images. Detection, configuration classification,
localization and false structures therefore require separate, joint assessment.
SR superiority is not assumed: direct configuration classification outperformed
the SR pathway in the favorable condition. Post-hoc analyses reuse consumed test
outputs and are exploratory. Higher reference SNR was not uniformly beneficial.

## Reproduce saved summaries on CPU

Use a separate environment with Python 3.14 and the pinned dependencies:

```sh
python -m pip install -r requirements.txt
python -m unittest test_publication_task_audit.py -v
python tools/check_saved_release.py
```

The last command re-derives the paired detection/localization audit from saved .npy arrays, compares every field except the timestamp, then runs the numerical figure/table generation path in a
temporary copy, checks both generated table JSON files byte-for-byte against
the archived outputs (24 supplementary S1 rows and 72 S2 rows), and verifies
that all generated numerical figures are nonempty. It does not train models,
run GPU reconstruction, download clinical images, or change the saved ledger.
GitHub Actions executes these CPU checks. Check the actual Actions run status.

The release contains saved predictions, localization counts, evaluation results,
phantom definitions, physics/reconstruction/training source and provenance.
Raw RF/reconstructed image arrays and trained checkpoints are excluded; full
simulation/model retraining is not reproduced by the CPU commands. PyTorch
would be required for new training and is not installed by requirements.txt.
The inherited experimental resource account is approximately **17.18 + U device-hours**;
U is unrecorded interrupted-phase computation, not zero. Compliance with the
24 device-hour experimental ceiling cannot be established exactly.

## Clinical context and licensing

Code: MIT, see LICENSE. Original documentation, numerical outputs and Figures 2–4:
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Figure 1 adapts a real public clinical mammogram and source annotation by
Kosmia Loizidou, Galateia Skouroumouni, Christos Nikolaou and Costas Pitris,
Breast Micro-Calcifications Dataset with Precisely Annotated Sequential Mammograms,
version 3, https://doi.org/10.5281/zenodo.14859694 , CC BY 4.0.
Changes: windowing, cropping, scale bar and panel/ROI labels. See
paper/FIGURE_PROVENANCE.md and paper/sources/clinical-illustration-provenance.json.
It was used only for explanation, never training, selection or evaluation; no
paired clinical ultrasound is assumed. The original clinical DICOM is excluded.
The optional acquisition/display source is included for provenance inspection;
it is not part of the offline CPU test and requires additional pydicom/Pillow
and access to the original dataset for execution.

No credentials, private AI review records or author-approval records are included.
This software is for research and does not establish a clinical alternative to
mammography. The journal manuscript has not been submitted.

Historical status fields inside frozen analysis JSON/source describe the original execution time. They are retained unchanged for provenance and do not represent current review/publication status. The original submission-package builder is supplied as historical source; use the CPU commands above for this public release.

Associated dataset article: Loizidou K, Skouroumouni G, Pitris C, Nikolaou C. Digital subtraction of temporally sequential mammograms for improved detection and classification of microcalcifications. European Radiology Experimental. 2021;5(1):40. https://doi.org/10.1186/s41747-021-00238-w .

The saved audit accepts --reproduce-saved solely for offline CPU verification, retaining its five-minute per-run cap while bypassing the historical experiment deadline. Experimental GPU entry points keep their original deadline and resource checks. Running generation scripts in place appends to the historical ledger; use tools/check_saved_release.py to preserve it. Frozen final-tests.json is a historical 24-test execution receipt; current CI runs the two named audit tests and saved-result regeneration, not the historical GPU-related suite. audit_bcmid.py is included as the RemoteZip dependency of the optional clinical illustration downloader; its separate dataset eligibility audit is not used in this study.
