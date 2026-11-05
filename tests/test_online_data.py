"""Atomic publication, held-out isolation, deduplication and worker cleanup."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from online_data import OnlinePool, draw_problems


class OnlineDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"ready").mkdir()
        self.parent=dict(snapshot_id="snapshot",training_groups=["train"])
        self.pool=OnlinePool(self.root,self.parent,"parent")

    def publish(self,name="batch",group="train",program=None,certificate=None):
        record=dict(schema_version=1,group=group,program=program or {"domain":"d","problem":"p"},certificate=certificate or {"actions":[]})
        raw=json.dumps(record).encode();identity=hashlib.sha256(raw).hexdigest()
        file=f"objects/{identity[:2]}/{identity}.json"
        path=self.root/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        reference=dict(id=identity,group=group,file=file)
        batch=dict(schema_version=1,parent_sha256="parent",snapshot_id="snapshot",samples=[reference])
        (self.root/"ready"/(name+".json")).write_text(json.dumps(batch))
        return reference

    def test_uncommitted_objects_are_invisible(self):
        reference=self.publish()
        (self.root/"ready/batch.json").unlink()
        self.assertEqual(self.pool.refresh(),[])
        self.assertTrue((self.root/reference["file"]).exists())

    def test_new_commits_are_additive_but_old_manifests_are_pinned(self):
        self.publish()
        self.assertEqual(len(self.pool.refresh()),1)
        pinned=self.pool.manifests.copy()
        self.publish("second",program={"problem":"second"})
        resumed=OnlinePool(self.root,self.parent,"parent")
        self.assertEqual(len(resumed.refresh(pinned)),2)
        (self.root/"ready/batch.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"manifest changed"):resumed.refresh()

    def test_changed_object_cannot_be_replayed(self):
        reference=self.publish();self.pool.refresh()
        (self.root/reference["file"]).write_text("{}");
        with self.assertRaisesRegex(ValueError,"checksum"):self.pool.read(reference)

    def test_foreign_parent_batch_is_rejected(self):
        self.publish()
        other=OnlinePool(self.root,self.parent,"different")
        with self.assertRaisesRegex(ValueError,"parent/snapshot"):other.refresh()

    def test_held_out_domain_cannot_enter_training(self):
        self.publish(group="test")
        with self.assertRaisesRegex(ValueError,"held-out"):self.pool.refresh()

    def test_certificates_for_the_same_program_do_not_inflate_instance_counts(self):
        self.publish(certificate={"actions":[1]})
        self.publish("second",certificate={"actions":[2]})
        self.assertEqual(len(self.pool.refresh()),1)

    def test_sampling_mixes_fresh_instances_without_changing_domain_weights(self):
        base={"a":[0],"b":[1]};fresh=[dict(id="x",group="a"),dict(id="y",group="b")]
        rows=draw_problems(random.Random(2),base,fresh,1000,1.0)
        self.assertTrue(all(isinstance(row,dict) for row in rows))
        self.assertTrue(430<sum(r["group"]=="a" for r in rows)<570)
        self.assertTrue(all(isinstance(row,int) for row in draw_problems(random.Random(2),base,fresh,20,0.0)))
        fresh.append(dict(id="z",group="c"))
        self.assertTrue(all(isinstance(row,int) for row in draw_problems(random.Random(2),base,fresh,20,0.0)))

    def test_collector_deadline_reaps_active_workers(self):
        run=self.root/"run";(run/"models").mkdir(parents=True);(run/"data/objects").mkdir(parents=True)
        (run/"models/merge-policy.json").write_text(json.dumps(self.parent))
        problem=dict(objects=[],init_tn=dict(tn=dict(subtasks=[{},{}])))
        record=dict(payload=dict(program=dict(problem=problem)))
        raw=json.dumps(record).encode();identity=hashlib.sha256(raw).hexdigest()
        path=run/"data/objects"/identity[:2]/(identity+".json");path.parent.mkdir();path.write_bytes(raw)
        index=dict(entries=[dict(id=identity,metadata=dict(kind="Problem",group_id="train"))])
        (run/"data/index.json").write_text(json.dumps(index))
        worker=self.root/"worker"
        worker.write_text("#!/usr/bin/env python3\nimport os,time\nprint(os.getpid(),flush=True)\ntime.sleep(30)\n")
        worker.chmod(0o755)
        output=self.root/"collection"
        script=Path(__file__).resolve().parents[1]/"scripts/train_online.py"
        result=subprocess.run([sys.executable,str(script),"--collect-only","--run-dir",str(run),
            "--online-dir",str(output),"--native",str(worker),"--rounds","2","--workers","2",
            "--generation-seconds","1","--job-seconds","20"],capture_output=True,text=True,timeout=15,start_new_session=True)
        self.assertEqual(result.returncode,0,result.stderr)
        state=json.loads((output/"generation-state.json").read_text())
        self.assertEqual(state["stopped"],"time_limit")
        logs=list((output/"logs").glob("*.log"));self.assertEqual(len(logs),2)
        for log in logs:
            pid=int(log.read_text().strip())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=="__main__":unittest.main()
