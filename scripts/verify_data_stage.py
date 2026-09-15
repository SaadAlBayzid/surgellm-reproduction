"""Repeat bounded data/unit/CUDA-readiness checks; never launches full training."""
import json,subprocess,sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    commands=[
      ('unit_tests',['-m','unittest','discover','-s','tests','-v'],0),
      ('freeze_replay',['-m','scripts.freeze_data'],0),
      ('preflight',['-m','scripts.preflight_baseline','--config','configs/baseline_roberta_four_task.json','--seed','0','--output','results/baseline-preflight.json'],2),
    ]
    report={}
    for name,args,expected in commands:
        run=subprocess.run([sys.executable]+args,cwd=root,capture_output=True,text=True)
        (root/'results'/f'{name}.log').write_text(run.stdout+run.stderr,encoding='utf-8')
        report[name]={'exit_code':run.returncode,'expected_exit_code':expected,'passed':run.returncode==expected}
        print(name,report[name],flush=True)
    (root/'results/data_stage_verification.json').write_text(json.dumps(report,indent=2))
    if not all(x['passed'] for x in report.values()): raise SystemExit(1)


if __name__=='__main__': main()
