import unittest
from qwen_decision.v3.leader_engines import gliclass_options,gliclass_state
from qwen_decision.v3.leader_compare import response_values,summarize,smoke_samples


class NativeComparisonTest(unittest.TestCase):
    def test_smoke_uses_present_image_counts_without_inventing_a_two_image_case(self):
        items=[{"id":"a","images":["one"]},{"id":"b","images":["other"]}]
        self.assertEqual(smoke_samples(items,"vision"),[items[0]])
        items.append({"id":"c","images":["x","y"]})
        self.assertEqual(smoke_samples(items,"vision"),[items[0],items[2]])

    def test_noul_native_true_first_maps_back_to_false_true_and_threshold_tie(self):
        question={"type":"noul","instructions":"True?","criteria":{"false":"wrong","true":"right"}}
        self.assertEqual(gliclass_options(question),(["true","false"],["right","wrong"]))
        for p,prediction in ((.2,0),(.8,1),(.5,1)):
            pred,probs=response_values(question,{"answers":{"answer":{"type":"noul","noul":p}}})
            self.assertEqual(pred,prediction);self.assertEqual(probs,[1-p,p])

    def test_choice_keys_and_author_label_mapping_are_preserved(self):
        q={"type":"choice","criteria":{"arbitrary_key":"Blue sky","another":None}}
        self.assertEqual(gliclass_options(q),(["arbitrary_key","another"],["Blue sky","another"]))
        pred,probs=response_values(q,{"answers":{"answer":{"type":"choice","choice":"another",
                                                  "probabilities":{"another":.8,"arbitrary_key":.2}}}})
        self.assertEqual(pred,1);self.assertEqual(probs,[.2,.8])

    def test_structured_state_follows_published_native_renderer(self):
        self.assertEqual(gliclass_state({"facts":["a","b"],"score":3}),"facts:\n- a\n- b\nscore: 3")
        self.assertEqual(gliclass_state(["a",{"x":2}]),'- a\n- {"x": 2}')

    def test_unsupported_kept_in_accuracy_denominator_and_sources_balanced(self):
        records=[{"source":"a","correct":True,"joint_correct":False,"status":"ok","nll":.2},
                 {"source":"a","correct":False,"joint_correct":True,"status":"unsupported","nll":None},
                 {"source":"b","correct":True,"joint_correct":True,"status":"ok","nll":.1}]
        report=summarize(records)
        self.assertEqual(report["leader_accuracy"],.75)
        self.assertEqual(report["unsupported"],1)
        self.assertEqual(report["sources"]["a"]["coverage"],.5)


if __name__=="__main__":unittest.main()
