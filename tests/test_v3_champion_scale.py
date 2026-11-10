import unittest
from qwen_decision.v3 import champion_scale_data as data,format as fmt


class ScaleDataTests(unittest.TestCase):
    def test_capped_quota_redistributes_without_repeating_small_sources(self):
        self.assertEqual(data.allocate(10,{"a":4.,"b":1.,"c":1.},{"a":2,"b":10,"c":10}),{"a":2,"b":4,"c":4})
        with self.assertRaises(ValueError):data.allocate(30,{"a":1.},{"a":4})

    def test_presentation_preserves_gold_options_and_all_context(self):
        row={"id":"example","state":{"email":"real context"},"question":{"type":"choice","instructions":"Choose.",
             "criteria":{"safe":None,"bad":"malicious"}},"target":[0.,1.],"ordered":False}
        result=data.presentation(row,1,force=True)
        self.assertEqual(result["target"],row["target"])
        self.assertEqual(list(result["question"]["criteria"].values()),fmt.options(row["question"])[1])
        self.assertIn(fmt.describe(row["state"]),result["question"]["instructions"])
        self.assertIn("Choose.",result["question"]["instructions"])
        self.assertEqual(row["question"]["criteria"],{"safe":None,"bad":"malicious"})

    def test_positional_choice_retains_original_references(self):
        row={"id":"positional","state":"","ordered":False,"target":[0.,1.],
             "question":{"type":"choice","instructions":"Pick.","criteria":{"A":"red","B":"both A and C"}}}
        self.assertEqual(data.presentation(row,1,force=True)["question"],row["question"])

    def test_boolean_polarity_is_not_reinterpreted(self):
        row={"id":"bool","source":"s","state":"evidence","ordered":False,"target":[1.,0.],"target_kind":"single",
             "question":{"type":"noul","instructions":"The claim is not supported."}}
        result=data.presentation(row,1,force=True)
        self.assertEqual(result["question"]["type"],"noul")
        self.assertEqual(result["target"],[1.,0.])
        self.assertEqual(data.stratum(row)[-1],0)

    def test_each_epoch_visits_every_modality_row_once_with_fixed_ratio(self):
        from qwen_decision.v3.champion_scale_train import Schedule
        from collections import Counter
        rows=[{"id":str(i),"modality":"text"} for i in range(32)]
        rows += [{"id":"image-"+str(i),"modality":"image"} for i in range(8)]
        schedule=Schedule(rows,epochs=2,seed=5)
        self.assertEqual(len(schedule),4)
        for epoch in range(2):
            visited=[]
            for i in range(epoch*2,(epoch+1)*2):
                batch=schedule[i]
                self.assertEqual(Counter(r["modality"] for r,_ in batch),{"text":16,"image":4})
                visited.extend(r["id"] for r,_ in batch)
            self.assertEqual(Counter(visited),{r["id"]:1 for r in rows})

    def test_checkpoint_resume_replays_optimizer_masters_and_dropout(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from safetensors.torch import load_file
        from qwen_decision.v3 import champion_scale_train as scale,train,lora_experiment
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                base=torch.nn.Linear(3,2,dtype=torch.bfloat16).requires_grad_(False)
                self.layer=lora_experiment.LoRALinear(base)
            def forward(self,x):return self.layer(x)
        def update(model,opt,x):
            opt.zero_grad();loss=model(x).float().square().mean();loss.backward()
            opt.clip_grad_norm(1.);opt.step()
        torch.manual_seed(11);model=Toy();opt=train.MasterAdamW(model,2e-4)
        x=torch.randn(4,3,dtype=torch.bfloat16);update(model,opt,x)
        with tempfile.TemporaryDirectory() as tmp,patch.object(scale,"ROOT",Path(tmp)),\
             patch.object(torch.cuda,"get_rng_state_all",return_value=[]),\
             patch.object(torch.cuda,"set_rng_state_all"),\
             patch.object(scale,"load_file",side_effect=lambda p,device:load_file(p,device="cpu")):
            scale.save_checkpoint(model,opt,1,200)
            update(model,opt,x)
            # Frozen base is always reloaded from the same original, separately from the adapter checkpoint.
            resumed=Toy();resumed.layer.base.load_state_dict(model.layer.base.state_dict())
            resumed_opt=train.MasterAdamW(resumed,2e-4)
            self.assertEqual(scale.restore_checkpoint(resumed,resumed_opt),(1,200))
            update(resumed,resumed_opt,x)
            for (_,a),(_,b) in zip(opt.pairs,resumed_opt.pairs):self.assertTrue(torch.equal(a,b))
            for a,b in zip(model.parameters(),resumed.parameters()):self.assertTrue(torch.equal(a,b))


if __name__=="__main__":unittest.main()
