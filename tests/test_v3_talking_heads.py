"""Mechanism, padding isolation, gradients, interventions and export safety."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from qwen_decision.v3 import talking_heads as th
from qwen_decision.v3 import talking_heads_experiment as exp


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module(); self.model.language_model = nn.Module()
        text = self.model.language_model
        text.config = SimpleNamespace(num_attention_heads=4, _attn_implementation="sdpa")
        text.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Module(); layer.block_type = "full_attention"
            layer.self_attn = nn.Module(); layer.self_attn.q_proj = nn.Linear(8,8,bias=False)
            text.layers.append(layer)


class TalkingHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(407)
        self.q = torch.randn(2,4,7,8,requires_grad=True)
        self.k = torch.randn(2,2,7,8,requires_grad=True)
        self.v = torch.randn(2,2,7,8,requires_grad=True)
        self.mask = torch.ones(2,1,1,7,dtype=torch.bool)
        self.mask[1,:,:,5:] = False

    def test_initial_identity_and_nonzero_immediate_cross_head_gradients(self):
        native = th.HeadMix(4,"native")
        mix = th.HeadMix(4,"talking")
        a = native(self.q,self.k,self.v,self.mask,.25)
        b = mix(self.q,self.k,self.v,self.mask,.25)
        self.assertTrue(torch.equal(a,b))
        b.square().sum().backward()
        for p in (mix.pre,mix.post):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater((p.grad*(1-mix.eye)).norm().item(),0)
        for p in (self.q,self.k,self.v):
            self.assertGreater(p.grad.norm().item(),0)

    def test_grouped_query_matches_direct_attention(self):
        mix = th.HeadMix(4,"native")
        scores = self.q @ self.k.repeat_interleave(2,1).transpose(-1,-2)*.25
        expected = (scores.masked_fill(~self.mask,-torch.inf).softmax(-1) @ self.v.repeat_interleave(2,1)).transpose(1,2)
        torch.testing.assert_close(mix(self.q,self.k,self.v,self.mask,.25),expected,atol=0,rtol=0)

    def test_padding_isolation_even_with_signed_head_mixing(self):
        mix = th.HeadMix(4,"talking")
        with torch.no_grad():
            mix.pre.add_(torch.randn_like(mix.pre)*.2); mix.post.add_(torch.randn_like(mix.post)*.2)
        a = mix(self.q,self.k,self.v,self.mask,.25)
        k,v = self.k.detach().clone(),self.v.detach().clone()
        k[1,:,5:] = 999;v[1,:,5:] = -999
        b = mix(self.q,k,v,self.mask,.25)
        self.assertTrue(torch.equal(a[1,:5],b[1,:5]))
        cropped = mix(self.q[1:,:,:5],self.k[1:,:,:5],self.v[1:,:,:5],self.mask[1:,:,:,:5],.25)
        torch.testing.assert_close(a[1:,:5],cropped,atol=1e-6,rtol=1e-6)

    def test_no_cross_preserves_diagonal_scaling_and_context_restores(self):
        m = FakeModel(); th.install(m.model.language_model,"talking")
        mods = th.modules(m.model.language_model)
        with torch.no_grad():
            for x in mods:
                x.pre.add_(.1);x.post.add_(.2)
        with th.intervention(m.model.language_model,"no_cross"):
            for x in mods:
                p,w=x.matrices()
                self.assertEqual((p*(1-x.eye)).abs().sum().item(),0)
                self.assertGreater((p.diagonal()-1).abs().sum().item(),0)
        self.assertTrue(all(x.mode=="active" for x in mods))
        with self.assertRaises(RuntimeError):
            with th.intervention(m.model.language_model,"identity"):
                raise RuntimeError("interruption")
        self.assertTrue(all(x.mode=="active" for x in mods))

    def test_diagonal_control_and_checkpointed_gradients(self):
        diag = th.HeadMix(4,"diagonal")
        diag(self.q,self.k,self.v,self.mask,.25).square().sum().backward()
        self.assertEqual((diag.pre.grad*(1-diag.eye)).abs().sum().item(),0)
        self.assertEqual((diag.post.grad*(1-diag.eye)).abs().sum().item(),0)
        ref = th.HeadMix(4,"talking")
        other = copy.deepcopy(ref)
        loss = ref(self.q,self.k,self.v,self.mask,.25).square().mean();loss.backward()
        loss2 = checkpoint(other,self.q,self.k,self.v,self.mask,.25,use_reentrant=False).square().mean();loss2.backward()
        for a,b in zip(ref.parameters(),other.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)

    def test_export_excludes_modules_and_reload_restores_all_weights(self):
        for arm in th.ARMS:
            model = FakeModel(); th.install(model.model.language_model,arm)
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if ".head_mix." in n:p.add_(.07)
            expected = copy.deepcopy(model.state_dict())
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                def save(m,t,p,c):
                    self.assertTrue(all(".head_mix." not in n for n in m.state_dict()))
                    torch.save(m.state_dict(),p/"stock.pt")
                def load(*args,**kwargs):
                    new=FakeModel();new.load_state_dict(torch.load(out/"stock.pt",weights_only=True));return new
                with patch.object(th.modeling,"save",side_effect=save):th.save(model,None,out,{})
                with patch.object(th.modeling,"load",side_effect=load):other=th.load(out,device="cpu")
                self.assertEqual(other.state_dict().keys(),expected.keys())
                for n,v in other.state_dict().items():self.assertTrue(torch.equal(v,expected[n]))

    def test_statistics_separate_architecture_and_cross_head_dependence(self):
        def modes(active,off):
            def rows(v):return [{"id":f"x{i}","source":"s","correct":v} for i in range(8)]
            return {"active":rows(active),"no_cross":rows(off),"identity":rows(False)}
        p={s:{"stock_sdpa":modes(False,False),"native":modes(False,False),"diagonal":modes(False,False),"talking":modes(True,False)} for s in exp.SEEDS}
        r=exp.summarize_contrasts(p,draws=32)
        self.assertEqual(len(r),4)
        self.assertTrue(all(x["delta"]==1 for x in r.values()))
        for s in p:p[s]["talking"]=modes(True,True)
        self.assertEqual(exp.summarize_contrasts(p,draws=32)["cross_head_removal_harm"]["delta"],0)

    def test_distractors_never_change_the_query_or_unique_gold_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/"inputs.json").write_text(json.dumps({"files":{},"summary":{}}))
            exp.build_distractors(out)
            rows=[json.loads(l) for l in (out/"distractor-test.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),512)
        for i in range(128):
            rr=rows[i*4:i*4+4]
            self.assertTrue(all(r["target"]==rr[0]["target"] and r["state"]["question"]==rr[0]["state"]["question"] for r in rr))
            fact=rr[0]["state"]["passage"]
            queried=fact.split(" has status ")[0]
            for r,level in zip(rr,(0,8,32,96)):
                lines=r["state"]["passage"].splitlines()
                self.assertEqual(len(lines),level+1)
                self.assertEqual(sum(line.startswith(queried+" has status ") for line in lines),1)
                self.assertIn(fact,lines)


if __name__ == "__main__":
    unittest.main()
