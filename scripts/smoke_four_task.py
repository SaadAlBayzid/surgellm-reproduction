"""Verify frozen four-task data; optional tiny CUDA diagnostic using existing debug head.

Does not implement the four-head baseline or any SURGELLM component.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.task_data import read_frozen, sha256
from src.roberta_baseline import EncodedSST2, seed_everything, evaluate
from src.device_runtime import resolve_runtime, autocast_context, grad_scaler, optimizer_update


def verify_bundle(root):
    lock=json.loads((root/'experiment.lock.json').read_text())
    if set(lock['tasks']) != {'D1','D2','D3','D4'}:
        raise ValueError('Require all four frozen tasks')
    tasks={}
    for task, info in lock['tasks'].items():
        path=root/info['manifest']
        if root.resolve() not in path.resolve().parents or sha256(path)!=info['sha256']:
            raise ValueError('Task manifest differs from experiment lock')
        tasks[task]=read_frozen(path.parent)
        if any(r.task_id!=task for rows in tasks[task].values() for r in rows):
            raise ValueError('Task ID mismatch')
    return tasks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,default=Path('data/frozen/v1'))
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda-smoke',action='store_true')
    a=p.parse_args()
    if Path('results').resolve() not in a.output.resolve().parents or a.output.exists():
        raise ValueError('Use a new report path under results/')
    seed_everything(0,4)
    tasks=verify_bundle(a.frozen)
    runtime=resolve_runtime()
    tokenizer=AutoTokenizer.from_pretrained(a.assets/'model',local_files_only=True,use_fast=True)
    report={'full_training_started':False,'runtime':runtime,'tasks':{},'data_frozen':True,
            'model_scope':'existing single-head debugging classifier; not four-head reproduction'}
    for task, splits in tasks.items():
        tiny=[next(r for r in splits['train'] if r.label==label) for label in (0,1)]
        data=EncodedSST2(tiny,tokenizer,128)
        report['tasks'][task]={'counts':{k:len(v) for k,v in splits.items()},'token_shape':list(data.encoding['input_ids'].shape)}
    if a.cuda_smoke and runtime['device']=='cuda':
        model=AutoModelForSequenceClassification.from_pretrained(a.assets/'model',num_labels=2,local_files_only=True,use_safetensors=True).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        scaler=grad_scaler(runtime)
        initial=model.classifier.out_proj.weight.detach().clone()
        for task,splits in tasks.items():
            data=EncodedSST2([next(r for r in splits['train'] if r.label==k) for k in (0,1)],tokenizer,128)
            model.train()
            batch=next(iter(DataLoader(data,batch_size=1)))
            with autocast_context(runtime): out=model(**{k:v.cuda() for k,v in batch.items()})
            if not torch.isfinite(out.loss): raise ValueError('Nonfinite CUDA loss')
            scaler.scale(out.loss).backward()
            norm,skipped=optimizer_update(model,optimizer,scaler,1.0)
            report['tasks'][task]['cuda']={'loss':out.loss.item(),'skipped':skipped,'metrics':evaluate(model,DataLoader(data,batch_size=1),'cuda',runtime)}
        if torch.equal(initial,model.classifier.out_proj.weight.detach()): raise ValueError('No optimizer update')
        report['cuda_smoke']='PASSED_DEBUG_ONLY'
    else:
        report['cuda_smoke']='SKIPPED_CUDA_UNAVAILABLE' if a.cuda_smoke else 'NOT_REQUESTED'
    report['data_checks']='PASSED'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
