"""Pure saved-result construction shared by table and manuscript generators."""

def build_supplement_tables(evaluation, audit):
    classification = []
    for condition in evaluation['conditions']:
        for method in ('direct', 'enhanced', 'xray_sr'):
            value = condition['models'][method]
            auc, sensitivity, fpr, accuracy = value['metrics']
            classification.append([
                f"{condition['frequency_hz']//1000000} MHz {condition['view_noise']}",
                {'direct': 'Direct US', 'enhanced': 'Enhanced', 'xray_sr': 'US + SR map'}[method],
                f'{auc:.4f}', f'{sensitivity:.3f}', f'{fpr:.3f}', f'{accuracy:.3f}',
                str(value['pattern_confusion'])])
    localization = []
    for condition in audit['conditions']:
        for value in condition['all_prespecified_thresholds']:
            localization.append([
                f"{condition['frequency_hz']//1000000} MHz {condition['view_noise']}",
                str(value['threshold']), f"{value['matched_particles']}/300",
                f"{value['localization_sensitivity']:.3f}",
                f"{value['fp_per_positive_image']:.3f}",
                f"{value['fp_per_negative_image']:.3f}", f"{value['fp_per_image']:.3f}"])
    assert len(classification) == 24 and len(localization) == 72
    return [
        {'title': 'All prespecified classification endpoints',
         'headers': ['Condition','Pathway','AUC','Sensitivity','FPR','Pattern BA','Confusion line/ring'],
         'rows': classification,
         'note': 'Validation-calibrated nominal FPR 5%; test FPR is empirical. Pattern is forced line/ring classification among positive cases, not malignancy classification.'},
        {'title': 'All nine prespecified localization operating points',
         'headers': ['Condition','Threshold','Matched','Sensitivity','FP / positive','FP / negative','FP / all'],
         'rows': localization,
         'note': 'Stratified false-candidate denominators are 60 positive, 30 negative, and 90 total test phantoms. Strata are exploratory; thresholds were prespecified, and no optimum was selected on test outputs.'}]
