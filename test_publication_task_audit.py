import unittest
import numpy as np
from publication_task_audit import summarize

class PairedTaskAuditChecks(unittest.TestCase):
    def test_presence_success_can_coexist_with_no_localization(self):
        c=np.array([[0,0,0],[0,5,10],[5,5,1]])
        result=summarize(c,np.array([False,True,True]),np.array([False,True,True]))
        self.assertEqual(result['localization_sensitivity'],.5)
        self.assertEqual(result['fraction_detected_positive_with_no_matched_particle'],.5)
        self.assertEqual(result['fp_per_negative_image'],0)
        self.assertEqual(result['fp_per_positive_image'],5.5)
    def test_no_true_positive_is_not_zero_conditional_failure(self):
        r=summarize(np.array([[0,0,1],[0,5,0]]),np.array([False,True]),np.array([True,False]))
        self.assertIsNone(r['fraction_detected_positive_with_no_matched_particle'])
        self.assertEqual(r['fp_negative_total'],1)

if __name__=='__main__': unittest.main()
