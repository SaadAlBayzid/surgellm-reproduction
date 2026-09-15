"""Baseline-only trainer. Full execution requires a matching successful CUDA smoke."""
import argparse
import copy
import json
import math
import os
import platform
import subprocess
import time
from datetime import timedelta
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, RobertaModel
from src.device_runtime import resolve_runtime, autocast_context, grad_scaler, optimizer_update
from src.roberta_baseline import seed_everything, learning_rate_factor
from src.multitask_baseline import (TASKS, FourTaskRoberta, EarlyStopping, batch_plan, batch_semantics,
    optimizer_groups, tensor_batch, evaluate_task, save_checkpoint, load_checkpoint)
from src.training_gate import verified_data, fingerprint, identity, object_hash, gate_errors


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def software_versions():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    for package in ('torch', 'transformers', 'numpy', 'scikit-learn', 'tokenizers', 'huggingface-hub'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = 'not installed'
    try:
        result['nvidia_driver'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        result['nvidia_driver'] = 'unavailable'
    return result


def validate_config(cfg):
    """Reject unsupported semantics rather than silently ignoring config fields."""
    expected = [(cfg['training']['optimizer'], 'AdamW'),
        (cfg['training']['scheduler']['name'], 'linear'),
        (cfg['model']['head']['activation'], 'gelu'),
        (cfg['model']['head']['num_labels'], 2),
        (cfg['evaluation']['metric'], 'macro_f1'),
        (cfg['evaluation']['strict_improvement'], True),
        (cfg['evaluation']['restore_best_before_test'], True),
        (cfg['tokenizer']['padding'], 'max_length')]
    if any(a != b for a,b in expected):
        raise ValueError('Unsupported baseline config semantics')
    if any(cfg['model'].get(k) for k in ('task_embeddings','surgical_features','prefix_tokens','feature_gate','iwn')):
        raise ValueError('Only the plain baseline is implemented')
    if any(v != 1 for v in cfg['training']['task_weights'].values()):
        raise ValueError('Only unit task loss weights are implemented')
    if cfg['training']['task_schedule'] not in ('proportional', 'task_balanced'):
        raise ValueError('Unknown task scheduling policy')


def encode(rows, tokenizer, length, chunk_size):
    chunks = []
    for start in range(0, len(rows), chunk_size):
        chunks.append(tokenizer([r.text for r in rows[start:start + chunk_size]],
            max_length=length, padding='max_length', truncation=True, return_tensors='pt'))
    result = {k: torch.cat([chunk[k] for chunk in chunks]) for k in ('input_ids', 'attention_mask')}
    result['labels'] = torch.tensor([r.label for r in rows], dtype=torch.long)
    if result['input_ids'].shape != (len(rows), length):
        raise ValueError('Tokenization shape mismatch')
    return result


def run(args):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    # Override seed only for initialization/scheduling; data selection is immutable.
    source_cfg = copy.deepcopy(cfg)
    cfg['seed'] = args.seed
    data, hashes = verified_data(root, cfg)
    signature = fingerprint(root, source_cfg, hashes)
    runtime = resolve_runtime('cpu' if args.cpu_smoke else cfg['runtime']['device'],
                              'fp32' if args.cpu_smoke else cfg['runtime']['precision'])
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.mode == 'full':
        errors = gate_errors(args.cuda_certificate, signature)
        if runtime['device'] != 'cuda' or world != 2:
            errors.append('Full training requires two CUDA ranks')
        if errors:
            raise RuntimeError('FULL RUN BLOCKED: ' + '; '.join(errors))
    if runtime['device'] == 'cpu' and (args.mode != 'smoke' or not args.cpu_smoke):
        raise RuntimeError('CPU execution requires explicit --mode smoke --cpu-smoke')
    if world > 1:
        if runtime['device'] != 'cuda':
            raise RuntimeError('CLI distributed path requires CUDA; CPU DDP is tested separately')
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    device = torch.device('cuda', local_rank) if runtime['device'] == 'cuda' else torch.device('cpu')
    seed_everything(args.seed, cfg['runtime']['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    output = args.output.resolve()
    if (root / 'results').resolve() not in output.parents:
        raise ValueError('Run output must be a new directory under results/')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)  # never overwrite previous evidence
    if world > 1:
        dist.barrier()
    batch_cfg = cfg['training']['batch']
    per_device, accumulation = batch_cfg['per_device'], batch_cfg['accumulation']
    epochs = cfg['training']['max_epochs']
    sizes = {t: len(data[t]['train']) for t in TASKS}
    policy = cfg['training']['task_schedule']
    if args.mode == 'smoke':
        epochs = 1
        if args.cpu_smoke:
            per_device, accumulation = 1, 2
        # Four full update windows with the selected full-run batch/length/FP16.
        # Temporary index view only; no frozen data is written or resampled.
        sizes = {t: min(sizes[t], per_device * world * accumulation) for t in TASKS}
    expected = batch_cfg['global_effective'] if not args.cpu_smoke else per_device * world * accumulation
    plan = batch_plan(sizes, per_device, world, accumulation, policy, args.seed, 0)
    semantics = batch_semantics(plan, per_device, world, accumulation, expected)
    total_updates = semantics['updates_per_epoch'] * epochs
    warmup = math.ceil(cfg['training']['scheduler']['warmup_fraction'] * total_updates)
    devices = [torch.cuda.get_device_name(local_rank)] if runtime['device'] == 'cuda' else ['cpu']
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, devices[0])
        devices = gathered
    software = software_versions()
    gpu_details = []
    if runtime['device'] == 'cuda':
        gpu_details = [{'local_device': i, 'name': torch.cuda.get_device_name(i),
                        'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
                        'capability': list(torch.cuda.get_device_capability(i))}
                       for i in range(world)]
    environment = {'software':software, 'devices':devices, 'gpu_details':gpu_details,
                   'world_size':world, 'precision':runtime['precision']}
    environment_hash = object_hash(environment)
    if args.mode == 'full':
        errors = gate_errors(args.cuda_certificate, signature, environment_hash)
        if errors:
            raise RuntimeError('FULL RUN BLOCKED: ' + '; '.join(errors))
    cfg['resolved'] = {'mode': args.mode, 'batch': semantics, 'epochs': epochs,
        'scheduler_total_updates': total_updates, 'warmup_steps': warmup,
        'runtime': runtime, 'devices': devices, 'software': software, 'dataset_manifest_hashes': hashes,
        'fingerprint': signature, 'environment':environment, 'environment_hash':environment_hash,
        'source_identity':identity(root, source_cfg, hashes),
        'evaluation': 'complete frozen validation/test on rank zero',
        'train_view_counts': sizes, 'seed': args.seed}
    if rank == 0:
        write_json(output / 'resolved_config.json', cfg)
    token_cfg = cfg['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(token_cfg['name'], revision=token_cfg['revision'],
        use_fast=token_cfg['use_fast'], add_prefix_space=token_cfg['add_prefix_space'])
    tokenizer.padding_side = token_cfg['padding_side']
    tokenizer.truncation_side = token_cfg['truncation_side']
    encoded = {}
    for t in TASKS:
        encoded[t] = {'train': encode(data[t]['train'][:sizes[t]], tokenizer,
                       cfg['tasks'][t]['max_length'], token_cfg['cache_chunk_size'])}
        if rank == 0:
            for split in ('validation', 'test'):
                encoded[t][split] = encode(data[t][split], tokenizer,
                    cfg['tasks'][t]['max_length'], token_cfg['cache_chunk_size'])
    encoder = RobertaModel.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'],
        add_pooling_layer=False, hidden_dropout_prob=cfg['model']['encoder_hidden_dropout'],
        attention_probs_dropout_prob=cfg['model']['encoder_attention_dropout'])
    model = FourTaskRoberta(encoder, cfg['model']['head']).to(device)
    wrapped = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    # Same initialization across ranks, distinct but repeatable dropout streams.
    seed_everything(args.seed + rank, cfg['runtime']['cpu_threads'])
    train_cfg = cfg['training']
    optimizer = torch.optim.AdamW(optimizer_groups(model, train_cfg['weight_decay'],
        train_cfg['weight_decay_exclusions']), lr=train_cfg['learning_rate'],
        betas=tuple(train_cfg['betas']), eps=train_cfg['epsilon'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: learning_rate_factor(step, total_updates, warmup))
    scaler = grad_scaler(runtime)
    before = {t: model.heads[t][-1].weight.detach().clone() for t in TASKS}
    stopping = EarlyStopping(cfg['evaluation']['patience'])
    update_count = 0
    epoch_logs = []
    started = time.perf_counter()
    if runtime['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        wrapped.train()
        plan = batch_plan(sizes, per_device, world, accumulation, policy, args.seed, epoch)
        batch_semantics(plan, per_device, world, accumulation, expected)
        if rank == 0:
            write_json(output / ('epoch_%d_batch_plan.json' % (epoch + 1)),
                       {'task_and_global_row_positions': plan, 'training_view_sizes': sizes})
        if runtime['device'] == 'cuda':
            torch.cuda.synchronize()
        train_start = time.perf_counter()
        loss_sum, skipped_count = 0.0, 0
        for index, (task, ids) in enumerate(plan):
            batch = tensor_batch(encoded[task]['train'], ids[rank * per_device:(rank + 1) * per_device], device)
            with autocast_context(runtime):
                result = wrapped(task_id=task, **batch)
            if not torch.isfinite(result['loss']):
                raise RuntimeError('Nonfinite training loss')
            loss_sum += result['loss'].detach().item()
            # DDP averages every microbatch across ranks; /K averages full windows.
            # Deliberately synchronize every backward: heads change between tasks.
            scaler.scale(result['loss'] / accumulation).backward()
            if (index + 1) % accumulation == 0:
                norm, skipped = optimizer_update(wrapped, optimizer, scaler, train_cfg['gradient_clip_norm'])
                skipped_count += int(skipped)
                if not skipped:
                    scheduler.step()
                    update_count += 1
        if runtime['device'] == 'cuda':
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        totals = torch.tensor([loss_sum, train_seconds], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(totals[:1], op=dist.ReduceOp.SUM)
            dist.all_reduce(totals[1:], op=dist.ReduceOp.MAX)
        validation, score = {}, 0.0
        eval_start = time.perf_counter()
        if rank == 0:
            validation = {t: evaluate_task(model, t, encoded[t]['validation'], device, runtime,
                          cfg['evaluation']['batch_size']) for t in TASKS}
            score = sum(x['macro_f1'] for x in validation.values()) / 4
        score_tensor = torch.tensor(score, dtype=torch.float64, device=device)
        if world > 1:
            dist.broadcast(score_tensor, src=0)
        score = score_tensor.item()
        improved, stop = stopping.observe(score)
        if improved:
            if rank == 0:
                save_checkpoint(output / 'best.pt', model, optimizer, scheduler, scaler,
                                epoch + 1, score, cfg['resolved'])
        if world > 1:
            dist.barrier()
        if rank == 0:
            log = {'epoch': epoch + 1, 'training_loss': totals[0].item() / world / len(plan),
                'training_seconds': totals[1].item(), 'evaluation_checkpoint_seconds': time.perf_counter() - eval_start,
                'global_examples_per_second': semantics['example_exposures_per_epoch'] / totals[1].item(),
                'microbatches_per_second': len(plan) / totals[1].item(), 'optimizer_updates': update_count,
                'amp_skipped_updates': skipped_count, 'learning_rate': scheduler.get_last_lr()[0],
                'validation': validation, 'mean_task_macro_f1': score, 'best': improved,
                'batch_semantics': semantics}
            epoch_logs.append(log)
            with (output / 'epochs.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(log, allow_nan=False) + '\n')
            print(json.dumps(log), flush=True)
        if stop:
            break
    heads_updated = all(not torch.equal(before[t], model.heads[t][-1].weight) for t in TASKS)
    if args.mode == 'smoke' and (not heads_updated or skipped_count or update_count != total_updates):
        raise RuntimeError('Smoke failed: all heads must update, with no AMP-skipped updates')
    # Shared output directory on Kaggle; barrier above guarantees checkpoint complete.
    checkpoint = load_checkpoint(output / 'best.pt', model)
    restored = all(torch.equal(v.cpu(), checkpoint['model'][k]) for k, v in model.state_dict().items())
    if not restored:
        raise RuntimeError('Best checkpoint restore mismatch')
    test = {}
    if rank == 0:
        test = {t: evaluate_task(model, t, encoded[t]['test'], device, runtime,
                 cfg['evaluation']['batch_size']) for t in TASKS}
        write_json(output / 'test_metrics.json', {'best_epoch': checkpoint['epoch'], 'tasks': test,
                    'mean_task_macro_f1': sum(x['macro_f1'] for x in test.values()) / 4})
    if world > 1:
        dist.barrier()
    peak = torch.cuda.max_memory_allocated(device) if runtime['device'] == 'cuda' else 0
    peaks = [peak]
    if world > 1:
        peaks = [None] * world
        dist.all_gather_object(peaks, peak)
    if rank == 0:
        report = {'status': 'PASSED', 'mode':args.mode, 'fingerprint': signature, 'world_size': world,
            'source_identity':cfg['resolved']['source_identity'],
            'environment':environment, 'environment_hash':environment_hash,
            'precision': runtime['precision'], 'devices': devices, 'software': software,
            'all_heads_updated': heads_updated, 'complete_validation': all(
                epoch_logs[-1]['validation'][t]['examples'] == len(data[t]['validation']) for t in TASKS),
            'best_restored': restored, 'peak_allocated_bytes_per_rank': peaks,
            'elapsed_seconds': time.perf_counter() - started, 'full_training_started': args.mode == 'full'}
        write_json(output / 'run_summary.json', report)
        if args.mode == 'smoke' and world == 2 and runtime['precision'] == 'fp16' and all('T4' in d for d in devices):
            write_json(output / 'cuda_smoke_pass.json', report)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline_roberta_four_task.json'))
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--cpu-smoke', action='store_true', help='Explicit CPU FP32 debug mode only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-certificate', type=Path, default=Path('results/kaggle-t4x2-smoke/cuda_smoke_pass.json'))
    args = parser.parse_args()
    if args.cpu_smoke and args.mode != 'smoke':
        parser.error('--cpu-smoke cannot be used with --mode full')
    run(args)


if __name__ == '__main__':
    main()
