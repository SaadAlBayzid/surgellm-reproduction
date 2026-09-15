import csv
import json
import unittest
import uuid
from pathlib import Path
from src.task_data import TaskExample, validate, load_d3, load_d4, hotpot_example, select, partition, freeze_task, read_frozen


class DataTests(unittest.TestCase):
    def fixture(self, n=100, task='D4'):
        return [TaskExample(f'Raw text {i}!  Keep punctuation.', i % 2, task, f'{task}:{i}', i) for i in range(n)]

    def csvfile(self, human, ai):
        p = Path(__file__).parent / ('fixture-' + uuid.uuid4().hex + '.csv')
        with p.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow(['text','label'])
            for label, n in [(0,human),(1,ai)]:
                for i in range(n): w.writerow([f'  Exact punctuation?! {label}:{i}  ',label])
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        return p

    def test_d3_rejects_wrong_version_and_counts(self):
        with self.assertRaisesRegex(ValueError, 'version/count'):
            load_d3(self.csvfile(13712,1166))
        with self.assertRaisesRegex(ValueError, 'class counts'):
            load_d3(self.csvfile(13711,1166))
        rows = load_d3(self.csvfile(13712,1165))
        self.assertTrue(rows[0].text.startswith('  '))
        self.assertEqual(rows[-1].label,1)

    def test_d4_exact_quota_and_reorder_invariance(self):
        rows = load_d4(self.csvfile(4452,5548))
        a = select(rows,{0:2500,1:2500})
        self.assertEqual(a,select(list(reversed(rows)),{0:2500,1:2500}))
        self.assertEqual(len(a),5000)
        self.assertEqual(sum(r.label for r in a),2500)
        with self.assertRaises(ValueError): select(rows,{0:5000,1:0})

    def test_hotpot_mapping_and_context(self):
        raw={'id':'abc','question':'Why?', 'level':'easy',
             'context': {'title':['distractor','support'], 'sentences':[['OMIT'], ['Keep!'] + ['word']*400]},
             'supporting_facts': {'title':['support'],'sent_id':[0]}}
        for difficulty,label in [('easy',0),('medium',1),('hard',1)]:
            raw['level']=difficulty; row=hotpot_example(raw,0)
            self.assertEqual(row.label,label)
            self.assertTrue(row.text.startswith('[Q]Why?[CTX]Keep!'))
            self.assertNotIn('OMIT',row.text)
            self.assertEqual(len(row.text.split('[CTX]')[1].split()),300)
        raw['level']='unknown'
        with self.assertRaises(ValueError): hotpot_example(raw,0)

    def test_invalid_labels_empty_text(self):
        for r in [TaskExample(' ',0,'D1','x',0), TaskExample('text',2,'D1','x',0)]:
            with self.assertRaises(ValueError): validate([r])

    def test_partition_no_id_overlap(self):
        rows=self.fixture()
        s=partition(rows)
        self.assertEqual([len(s[k]) for k in ('train','validation','test')],[70,15,15])
        self.assertEqual(s,partition(rows))
        self.assertEqual(len({r.source_id for v in s.values() for r in v}),100)

    def test_freeze_reuse_tamper_rejected(self):
        p=Path(__file__).parent / ('bundle-'+uuid.uuid4().hex)
        p_raw=self.csvfile(50,50)
        s=partition(self.fixture())
        try:
            freeze_task('D4',s,p,[p_raw],'test')
            freeze_task('D4',s,p,[p_raw],'test')
            self.assertEqual(s,read_frozen(p))
            (p/'train.jsonl').write_text('tamper')
            with self.assertRaises(ValueError): read_frozen(p)
            with self.assertRaises(ValueError): freeze_task('D4',s,p,[p_raw],'test')
        finally:
            if p.exists():
                for f in p.iterdir(): f.unlink()
                p.rmdir()


if __name__=='__main__': unittest.main()
