import copy
import json
import shutil
import unittest
import uuid
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from transformers import RobertaConfig, RobertaModel
from src.multitask_baseline import (TASKS, FourTaskRoberta, EarlyStopping, batch_plan, batch_semantics,
    optimizer_groups, evaluate_task, binary_metrics, save_checkpoint, load_checkpoint)
from src.training_gate import gate_errors, fingerprint, object_hash
from src.device_runtime import grad_scaler, optimizer_update
from src.roberta_baseline import learning_rate_factor


def tiny_model():
    torch.manual_seed(17)
    encoder = RobertaModel(RobertaConfig(vocab_size=32, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=16, max_position_embeddings=20,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0), add_pooling_layer=False)
    return FourTaskRoberta(encoder, {'hidden_size': 12, 'intermediate_size': 6,
        'dropout_before_first': 0, 'dropout_before_second': 0})


def batch(n=5):
    return {'input_ids': torch.tensor([[0, 4 + i % 8, 5, 2] for i in range(n)]),
            'attention_mask': torch.ones(n, 4, dtype=torch.long),
            'labels': torch.tensor([i % 2 for i in range(n)])}


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    model, reference = tiny_model(), tiny_model()
    wrapped = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    for index, task in enumerate(TASKS):
        data = batch(4)
        local = {k: v[rank*2:(rank+1)*2] for k, v in data.items()}
        (wrapped(task, **local)['loss'] / 2).backward()
        (reference(task, **data)['loss'] / 2).backward()
        if (index+1) % 2 == 0:
            for p, r in zip(model.parameters(), reference.parameters()):
                if r.grad is not None:
                    torch.testing.assert_close(p.grad, r.grad, atol=1e-7, rtol=2e-5)
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, r, atol=1e-7, rtol=2e-5)
    dist.destroy_process_group()


class MultiTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = Path('tests') / ('tmp_multitask_' + uuid.uuid4().hex)
        self.directory.mkdir()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_four_heads_route_and_shared_encoder_gradients(self):
        model = tiny_model()
        self.assertEqual(set(model.heads), set(TASKS))
        for task in TASKS:
            model.zero_grad(set_to_none=True)
            output = model(task, **batch())
            self.assertEqual(tuple(output['logits'].shape), (5, 2))
            self.assertTrue(torch.isfinite(output['loss']))
            output['loss'].backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.encoder.parameters()
                                   if p.grad is not None), 0)
            for other in TASKS:
                grads = [p.grad for p in model.heads[other].parameters()]
                if other == task:
                    self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
                else:
                    self.assertTrue(all(g is None for g in grads))
        with self.assertRaises(ValueError):
            model('bad', **batch())

    def test_routing_matches_direct_head(self):
        model = tiny_model().eval()
        b = batch()
        hidden = model.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state[:, 0]
        for t in TASKS:
            torch.testing.assert_close(model(t, **b)['logits'], model.heads[t](hidden))

    def test_scheduling_determinism_coverage_and_balance(self):
        sizes = dict(zip(TASKS, [19, 7, 11, 5]))
        for policy in ('proportional', 'task_balanced'):
            a = batch_plan(sizes, 2, 2, 2, policy, 0, 0)
            self.assertEqual(a, batch_plan(sizes, 2, 2, 2, policy, 0, 0))
            self.assertNotEqual(a, batch_plan(sizes, 2, 2, 2, policy, 1, 0))
            self.assertNotEqual(a, batch_plan(sizes, 2, 2, 2, policy, 0, 1))
            self.assertEqual(len(a) % 2, 0)
            for t in TASKS:
                self.assertEqual({i for task, ids in a if task == t for i in ids}, set(range(sizes[t])))
            if policy == 'task_balanced':
                self.assertEqual([sum(t == task for task, _ in a) for t in TASKS], [5]*4)

    def test_effective_batch_and_tail(self):
        sizes = dict(zip(TASKS, [6899, 1400, 2214, 3500]))
        plan = batch_plan(sizes, 16, 2, 2, 'task_balanced', 0, 0)
        info = batch_semantics(plan, 16, 2, 2, 64)
        self.assertEqual(info['updates_per_epoch'], 432)
        self.assertEqual(info['example_exposures_per_epoch'], 27648)
        proportional = batch_plan(sizes, 16, 2, 2, 'proportional', 0, 0)
        self.assertEqual(len(proportional), 440)
        with self.assertRaises(ValueError):
            batch_semantics(plan, 16, 1, 2, 64)

    def test_accumulation_matches_concatenated_global_objective(self):
        # Simulate 2 ranks x 2 examples x 2 micros; average rank+micro losses.
        a, b = tiny_model(), tiny_model()
        data = batch(8)
        a('D1', **data)['loss'].backward()
        for micro in range(2):
            for rank in range(2):
                start = (micro * 2 + rank) * 2
                small = {k: v[start:start+2] for k, v in data.items()}
                (b('D1', **small)['loss'] / 4).backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.grad is not None:
                torch.testing.assert_close(pa.grad, pb.grad, atol=1e-7, rtol=2e-5)
        oa, ob = torch.optim.SGD(a.parameters(), lr=.1), torch.optim.SGD(b.parameters(), lr=.1)
        oa.step(); ob.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, atol=1e-7, rtol=2e-5)

    def test_complete_evaluation_including_short_tail_and_auc(self):
        model = tiny_model()
        for task in TASKS:
            small = evaluate_task(model, task, batch(5), 'cpu', {'precision':'fp32'}, 2)
            whole = evaluate_task(model, task, batch(5), 'cpu', {'precision':'fp32'}, 5)
            self.assertEqual(small['examples'], 5)
            for metric in ('loss', 'accuracy', 'macro_f1', 'roc_auc'):
                self.assertAlmostEqual(small[metric], whole[metric], places=6)
        # Each batch could be single-class, but the aggregate has a defined AUC.
        self.assertEqual(binary_metrics([0,0,1,1], [.1,.4,.35,.8], 2)['roc_auc'], .75)

    def test_single_class_auc_is_null_not_nan(self):
        result = binary_metrics([1,1], [.7,.8], 1)
        self.assertIsNone(result['roc_auc'])
        self.assertEqual(result['roc_auc_undefined_reason'], 'single_class_split')
        json.dumps(result, allow_nan=False)

    def test_decay_exclusions(self):
        model = tiny_model()
        groups = optimizer_groups(model, .01, ['bias', 'LayerNorm.weight'])
        excluded = {id(p) for p in groups[1]['params']}
        for name, p in model.named_parameters():
            self.assertEqual(id(p) in excluded, name.endswith(('bias', 'LayerNorm.weight')))

    def test_early_stopping_ties_and_reset(self):
        stopping = EarlyStopping(2)
        self.assertEqual(stopping.observe(.5), (True, False))
        self.assertEqual(stopping.observe(.5), (False, False))
        self.assertEqual(stopping.observe(.6), (True, False))
        self.assertEqual(stopping.observe(.4), (False, False))
        self.assertEqual(stopping.observe(.6), (False, True))

    def test_end_to_end_cpu_trainer_with_tiny_fixtures(self):
        from scripts.train_four_task import run
        from src.task_data import TaskExample
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/baseline_roberta_four_task.json').read_text())
        cfg['model']['head'].update(hidden_size=12, intermediate_size=6)
        for task in TASKS:
            cfg['tasks'][task]['max_length'] = 4
        cfg_path = self.directory / 'config.json'
        cfg_path.write_text(json.dumps(cfg))
        rows = {t:{s:[TaskExample('test text', i%2, t, t+s+str(i), i) for i in range(n)]
                   for s,n in [('train',8),('validation',5),('test',3)]} for t in TASKS}
        class FixtureTokenizer:
            def __call__(self, texts, **kwargs):
                return {k:v for k,v in batch(len(texts)).items() if k != 'labels'}
        output = root / 'results' / ('cpu-fixture-' + uuid.uuid4().hex)
        args = Namespace(config=cfg_path, seed=0, mode='smoke', cpu_smoke=True,
                         output=output, cuda_certificate=output/'absent.json')
        try:
            with patch('scripts.train_four_task.verified_data', return_value=(rows, {'fixture':'test-only'})), \
                 patch('scripts.train_four_task.AutoTokenizer.from_pretrained', return_value=FixtureTokenizer()), \
                 patch('scripts.train_four_task.RobertaModel.from_pretrained', return_value=tiny_model().encoder):
                run(args)
            report = json.loads((output/'run_summary.json').read_text())
            self.assertTrue(report['all_heads_updated'])
            self.assertTrue(report['best_restored'])
            self.assertTrue(report['complete_validation'])
            self.assertFalse((output/'cuda_smoke_pass.json').exists())
            results = json.loads((output/'test_metrics.json').read_text())
            self.assertTrue(all(v['examples']==3 for v in results['tasks'].values()))
            # Persist small, explicitly fixture-only CPU smoke evidence.
            (root/'results/trainer-cpu-fixture-smoke.json').write_text(json.dumps(
                {'scope':'tiny random encoder and synthetic fixtures; not pretrained performance',
                 'run':report, 'test':results}, indent=2))
        finally:
            if output.exists():
                shutil.rmtree(output)

    def test_cpu_four_task_optimizer_checkpoint_smoke(self):
        model = tiny_model()
        initial = {t: model.heads[t][-1].weight.clone() for t in TASKS}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, 4, 0))
        scaler = grad_scaler({'precision':'fp32'})
        for task in TASKS:
            loss = model(task, **batch())['loss']
            scaler.scale(loss).backward()
            _, skipped = optimizer_update(model, optimizer, scaler, 1)
            self.assertFalse(skipped)
            scheduler.step()
        self.assertTrue(all(not torch.equal(initial[t], model.heads[t][-1].weight) for t in TASKS))
        expected = copy.deepcopy(model.state_dict())
        path = self.directory / 'best.pt'
        save_checkpoint(path, model, optimizer, scheduler, scaler, 1, .4, {'smoke':True})
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1)
        checkpoint = load_checkpoint(path, model)
        self.assertEqual(checkpoint['epoch'], 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertEqual(checkpoint['scheduler'], scheduler.state_dict())

    def test_cuda_gate_missing_stale_or_incomplete(self):
        p = self.directory / 'certificate.json'
        self.assertTrue(gate_errors(p, 'abc'))
        good = {'status':'PASSED', 'fingerprint':'abc', 'world_size':2, 'precision':'fp16',
            'devices':['Tesla T4','Tesla T4'], 'all_heads_updated':True,
            'complete_validation':True, 'best_restored':True,
            'mode':'smoke', 'full_training_started':False,
            'environment':{'test':'fixture'}, 'environment_hash':object_hash({'test':'fixture'})}
        p.write_text(json.dumps(good))
        self.assertEqual(gate_errors(p, 'abc'), [])
        self.assertTrue(gate_errors(p, 'changed'))
        self.assertTrue(gate_errors(p, 'abc', object_hash({'test':'changed'})))
        good['world_size'] = 1
        p.write_text(json.dumps(good))
        self.assertTrue(gate_errors(p, 'abc'))

    def test_certificate_changes_with_every_code_config_and_data_identity(self):
        # Regression: the original six-file allowlist omitted other project code/config.
        root = Path(__file__).resolve().parents[1]
        fixture = self.directory / 'identity'
        fixture.mkdir()
        for directory in ('src', 'scripts', 'tests', 'configs'):
            (fixture/directory).mkdir()
        for name in ('src/multitask_baseline.py', 'src/training_gate.py', 'src/device_runtime.py',
                     'src/task_data.py', 'src/roberta_baseline.py', 'scripts/train_four_task.py'):
            (fixture/name).write_text((root/name).read_text())
        extra = fixture/'scripts/extra.py'
        extra.write_text('first')
        initial = fingerprint(fixture, {'seed':0}, {'D1':'original'})
        extra.write_text('changed')
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':0}, {'D1':'original'}))
        initial = fingerprint(fixture, {'seed':0}, {'D1':'original'})
        (fixture/'configs/other.json').write_text('{}')
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':0}, {'D1':'original'}))
        initial = fingerprint(fixture, {'seed':0}, {'D1':'original'})
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':1}, {'D1':'original'}))
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':0}, {'D1':'changed'}))
        (fixture/'data').mkdir()
        (fixture/'data/raw.csv').write_text('unchanged source fixture')
        initial = fingerprint(fixture, {'seed':0}, {'D1':'original'})
        (fixture/'data/raw.csv').write_text('changed')
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':0}, {'D1':'original'}))
        initial = fingerprint(fixture, {'seed':0}, {'D1':'original'})
        extra.unlink()
        self.assertNotEqual(initial, fingerprint(fixture, {'seed':0}, {'D1':'original'}))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_process_ddp_changing_heads_accumulation(self):
        mp.spawn(ddp_worker, args=((self.directory.resolve() / 'rendezvous').as_uri(),),
                 nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
