import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import full_credit, normalized, parse_jev
from score_core import compare, metrics, threshold_passes, vector

class ProtocolTest(unittest.TestCase):
    def test_obs_linear_not_softmax(self):
        mapping={'parent_id':'p','option_to_query_id':{'A':'qa','B':'qb','C':'qc'}}
        parsed={i:{'probabilities':{'A':q,'B':1-q}} for i,q in [('qa',.8),('qb',.1),('qc',.1)]}
        v=vector(mapping,parsed,'binary_ovr')
        self.assertEqual(v['prediction'],'A');self.assertAlmostEqual(v['confidence'],.8)
    def test_all_zero_obs_is_invalid(self):
        with self.assertRaises(ValueError):normalized({'A':0,'B':0})
    def test_noul_uses_probability_not_confidence(self):
        r={'model':'pinned','answers':{'decision':{'type':'noul','noul':.82,'confidence':.2}}}
        p,pred=parse_jev(r,{'type':'noul','criteria':{'true':'Full proposition','false':'Full negation'}},'pinned')
        self.assertEqual(p['A'],.82);self.assertEqual(pred,'A')
        with self.assertRaises(ValueError):parse_jev(r,{'type':'noul'},'wrong')
    def test_shuffled_alignment(self):
        mp={'parent_id':'p','display_to_canonical':{'A':'C','B':'A','C':'B'}}
        v=vector(mp,{'p':{'prediction':'B','probabilities':{'A':.1,'B':.7,'C':.2}}},'choice_shuffled')
        self.assertEqual(v['prediction'],'A');self.assertEqual(v['probabilities'],{'C':.1,'A':.7,'B':.2})
    def test_numeric10_alignment(self):
        for start in [0,1]:
            align={str(i+start):chr(65+i) for i in range(10)}
            ps={k:float(v=='J') for k,v in align.items()}
            v=vector({'parent_id':'x','display_to_canonical':align},
                     {'x':{'prediction':str(start+9),'probabilities':ps}},'choice_numbers')
            self.assertEqual(v['prediction'],'J')
    def test_none_not_abstain(self):
        v=vector({'parent_id':'p','display_to_canonical':{'A':'A','B':'B'}},
            {'p':{'prediction':'A','probabilities':{'A':1.,'B':0.}}},'choice_none')
        self.assertFalse(v['abstained'])
    def test_abstain(self):
        v=vector({'parent_id':'p','none_id':'C'},
            {'p':{'prediction':'C','probabilities':{'A':.1,'B':.1,'C':.8}}},'choice_abstain')
        self.assertTrue(v['abstained'])
    def test_accepted_reference_not_partial_credit(self):
        g={'option_scores':{'A':.5,'B':1.,'C':1.}}
        self.assertEqual(full_credit(g),{'B','C'})
        m=metrics({'x':g},{'x':{'prediction':'A','abstained':False}},['A','B','C'])
        self.assertEqual(m['reference_hit_accuracy_nonabstained'],0)
        self.assertEqual(m['f1_n'],0)
    def test_tv_and_reversal(self):
        b={'x':{'prediction':'A','probabilities':{'A':.8,'B':.2}}}
        v={'x':{'prediction':'B','probabilities':{'A':.3,'B':.7}}}
        m,_=compare(b,v,{'x':{'label':'A'}})
        self.assertEqual(m['flipped'],1);self.assertAlmostEqual(m['tv'],.5)
    def test_threshold_boundary(self):
        self.assertTrue(threshold_passes(.9-1e-13,.9));self.assertFalse(threshold_passes(.899,.9))
    def test_invalid_probabilities(self):
        for p in [{'A':float('nan')},{'A':1.01},{'A':-.1},{'A':True}]:
            with self.assertRaises(ValueError):normalized(p)

if __name__=='__main__':unittest.main()
