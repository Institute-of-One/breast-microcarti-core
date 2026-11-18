# Clinical figure provenance

Figure 1 uses a public clinical mammogram and its original annotation from Loizidou K, Skouroumouni G, Nikolaou C, Pitris C, Breast Micro-Calcifications Dataset with Precisely Annotated Sequential Mammograms, version 3. Source: https://zenodo.org/records/14859694 ; DOI https://doi.org/10.5281/zenodo.14859694 ; license CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ .

Selected pair: `Dataset_v3/Suspicious_cases/1/CC_recent.dcm` and `CC_recent_GT.jpg`. This is a public clinical image, not a synthetic or generatively edited image. The source annotation is reproduced as an explanatory overlay, not as a new clinical diagnosis. Neither image was used for training, model selection or performance evaluation. No paired clinical ultrasound acquisition is assumed.

Changes comprise display windowing, region cropping, a 5 mm scale bar, panel labels and an ROI rectangle. The raw DICOM pixel spacing is 0.07 mm. Full source hashes, selection rule and display parameters are in `sources/clinical-illustration-provenance.json`; processing is reproducible with `acquire_clinical_illustration.py` and `build_clinical_figure.py`. Acquisition defaults to metadata-only unless `--fetch` is explicitly passed. Raw DICOM and original annotation files remain outside the review archive in `.cache/public-clinical/`.

Figures 2–4 are programmatically generated from saved synthetic-study outputs, with no new inference, training or GPU reconstruction.

Associated dataset article: Loizidou K, Skouroumouni G, Pitris C, Nikolaou C. Digital subtraction of temporally sequential mammograms for improved detection and classification of microcalcifications. European Radiology Experimental. 2021;5(1):40. https://doi.org/10.1186/s41747-021-00238-w .
