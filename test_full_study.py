import unittest
import numpy as np
import torch
from generate_full_study import make_manifest,occupancy
from study_models import StudyModel
from evaluate_full_study import auc,localize

class FullStudyChecks(unittest.TestCase):
    def test_phantom_split_and_target(self):
        rows=make_manifest()
        self.assertEqual(rows,make_manifest())
        self.assertEqual(len({r['id'] for r in rows}),480)
        for label in range(3):
            self.assertEqual([sum(r['class']==label and r['split']==s for r in rows) for s in ('train','validation','test')],[100,30,30])
        for row in (rows[0],rows[160],rows[319],rows[320],rows[479]):
            target=occupancy(row)
            self.assertEqual(target.shape,(256,256))
            if row['class']:
                self.assertLess(abs(target.sum()/(5*np.pi*(50/32)**2)-1),.05)
                matched,total,fp=localize(target,row,.5)
                self.assertEqual((matched,total,fp),(5,5,0))
            else: self.assertEqual(target.sum(),0)
    def test_three_input_paths(self):
        torch.set_num_threads(1)
        x=torch.randn(2,1,128,128)
        for method in ('direct','enhanced','xray_sr'):
            model=StudyModel(method)
            self.assertLess(sum(p.numel() for p in model.parameters()),250000)
            p,s=model(x); self.assertEqual(tuple(p.shape),(2,3))
            self.assertEqual(s is None,method!='xray_sr')
            if s is not None: self.assertEqual(tuple(s.shape),(2,1,256,256))
    def test_auc_ties_and_direction(self):
        y=np.array([0,0,1,2])
        self.assertEqual(auc(y,np.array([0,0,1,1])),1)
        self.assertEqual(auc(y,np.ones(4)),.5)
        self.assertEqual(auc(y,np.array([1,1,0,0])),0)

if __name__=='__main__': unittest.main()
