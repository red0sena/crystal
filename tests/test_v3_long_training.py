"""Long fits must resume FP32 optimizer evolution, not just BF16 predictions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_decision.v3 import long_training as experiment,train


class ResumeContractTest(unittest.TestCase):
    def make(self):
        model = torch.nn.Linear(3,2,dtype=torch.bfloat16)
        return model,train.MasterAdamW(model,3e-5)

    def step(self,model,opt,lr=None):
        opt.zero_grad()
        if lr is not None:
            for group in opt.param_groups:group['lr'] = lr
        x = torch.tensor([[.3,-.2,.7],[.1,.9,-.4]],dtype=torch.bfloat16)
        loss = model(x).float().square().sum();loss.backward()
        opt.clip_grad_norm(1.);opt.step()
        return loss.item()

    def write(self,path,payload):
        torch.save(payload,path/'optimizer.pt')
        (path/'complete.json').write_text(json.dumps({'step':payload['step'],
            'files':{'optimizer.pt':train.sha256_file(path/'optimizer.pt')}}))

    @patch('torch.cuda.set_rng_state_all')
    @patch('torch.cuda.get_rng_state_all',return_value=[])
    def test_next_fp32_update_exact_after_resume(self,*_):
        torch.manual_seed(17);model,opt = self.make()
        for i in range(3):self.step(model,opt,lr=3e-5*(i+1)/4)
        payload = experiment.training_state(model,opt,3)
        # CPU tensors alias their owners; emulate the independent disk snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp);self.write(path,payload)
            expected_loss = self.step(model,opt,lr=3e-5);expected = experiment.master_digest(opt)
            restored,newopt = self.make()
            self.assertEqual(experiment.restore_optimizer(restored,newopt,path),3)
            self.assertIs(newopt.param_groups,newopt.optimizer.param_groups)
            self.assertEqual(self.step(restored,newopt,lr=3e-5),expected_loss)
            self.assertEqual(experiment.master_digest(newopt),expected)
            for a,b in zip(model.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))

    @patch('torch.cuda.set_rng_state_all')
    @patch('torch.cuda.get_rng_state_all',return_value=[])
    def test_reordered_parameters_and_tampered_state_rejected(self,*_):
        model,opt = self.make();self.step(model,opt)
        payload = experiment.training_state(model,opt,1)
        payload['names'] = list(reversed(payload['names']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp);self.write(path,payload)
            restored,newopt = self.make()
            with self.assertRaisesRegex(ValueError,'parameter order'):
                experiment.restore_optimizer(restored,newopt,path)
            with (path/'optimizer.pt').open('ab') as f:f.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                experiment.restore_optimizer(restored,newopt,path)


if __name__=='__main__':unittest.main()
