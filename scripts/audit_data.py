"""Inventory local raw artifacts and freeze counts without training."""
import csv,json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from src.task_data import sha256


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/data_freeze.json').read_text())
    stats={}
    csv.field_size_limit(10_000_000)
    for task,paths in cfg['raw'].items():
        stats[task]={'raw_artifacts':[]}
        for relative in paths:
            path=root/relative
            if path.suffix=='.parquet':
                raw=pq.read_table(path,columns=['level']).to_pylist()
                counts=dict(Counter(r['level'] for r in raw)); columns=pq.ParquetFile(path).schema.names
            else:
                with path.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ',')
                    raw=list(reader); columns=reader.fieldnames
                counts=dict(Counter(r['label'] for r in raw))
            stats[task]['raw_artifacts'].append({'path':relative,'sha256':sha256(path),'bytes':path.stat().st_size,
                  'rows':len(raw),'raw_class_counts':counts,'columns':columns})
        stats[task]['frozen']=json.loads((root/cfg['output']/task/'manifest.json').read_text())
        for v in stats[task]['frozen']['splits'].values():
            v.pop('source_ids'); v.pop('row_indices')
    other=['../../work/data-downloads/d3_v1.zip','../../work/roberta-assets/data/train-00000-of-00001.parquet','../../work/roberta-assets/data/validation-00000-of-00001.parquet']
    stats['additional_raw_artifacts']=[{'path':p,'sha256':sha256(root/p),'bytes':(root/p).stat().st_size} for p in other]
    (root/'results/data_audit.json').write_text(json.dumps(stats,indent=2))
    for task in ['D1','D2','D3','D4']:
        print(task,[(x['rows'],x['raw_class_counts']) for x in stats[task]['raw_artifacts']],
              {k:v['class_counts'] for k,v in stats[task]['frozen']['splits'].items()},stats[task]['frozen']['cross_split_exact_text_overlap'])


if __name__=='__main__': main()
