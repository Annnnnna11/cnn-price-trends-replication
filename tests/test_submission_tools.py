"""Checks for new-run identity binding and data-integrity failures; no GPU imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduction_launcher',ROOT/'scripts/reproduce.py')
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)


class SubmissionTools(unittest.TestCase):
    def baseline(self, root):
        directory = root/'baseline'; directory.mkdir()
        launcher.write(directory/'manifest.json', {'run_id':'new-test-run'})
        for seed in range(101,106):
            folder = directory/f'seed_{seed}'; folder.mkdir()
            (folder/'best.pt').write_bytes(f'fake fixture, not a model, {seed}'.encode())
            launcher.write(folder/'completed.json',{'seed':seed,'run_id':'new-test-run',
                'checkpoint_hashes':{'best.pt':launcher.sha(folder/'best.pt')}})

    def test_new_run_binding_keeps_training_python_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary); self.baseline(run)
            destination=launcher.runtime_packages(run)
            for package in launcher.PACKAGES:
                for source in (ROOT/'code'/package).glob('*.py'):
                    self.assertEqual(source.read_bytes(),(destination/package/source.name).read_bytes())
                plan=destination/package/'study_plan.json'
                if plan.exists():self.assertEqual(launcher.read(plan)['baseline_run_id'],'new-test-run')
            inherited=launcher.read(destination/'i20r20_followup_v1/inherited_hashes.json')
            for key, package in [('ablation','i20r20_ablation_v1'),('baseline_eval','i20r20_eval_v1')]:
                for name,digest in inherited[key].items():self.assertEqual(launcher.sha(destination/package/name),digest)
            self.assertEqual(launcher.runtime_packages(run),destination)
            changed=destination/'i20r20_ablation_v1/models.py'
            changed.write_bytes(changed.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'Changed runtime source'):launcher.runtime_packages(run)

    def test_checkpoint_corruption_stops_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary); self.baseline(run)
            (run/'baseline/seed_101/best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):launcher.runtime_packages(run)
            self.assertFalse((run/'code').exists())

    def test_plan_does_not_start_or_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'not-created'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'plan','--run-root',str(run)],capture_output=True,text=True,check=True)
            plan=json.loads(result.stdout)
            self.assertFalse(plan['training_started']); self.assertFalse(run.exists())
            self.assertEqual(plan['training_members'],15)
            self.assertLess(plan['stages'].index('configure-extensions'),plan['stages'].index('evaluate-baseline'))

    def test_recorded_results_cannot_be_run_destination(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'plan','--run-root',str(ROOT/'results')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('separate run directory',result.stderr)

    def test_data_checker_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/'data'; data.mkdir(); payload=b'original'
            (data/'fixture.dat').write_bytes(payload)
            manifest=root/'files.json'
            manifest.write_text(json.dumps({'files':[{'name':'fixture.dat','bytes':len(payload),
                'sha256':hashlib.sha256(payload).hexdigest()}]}),encoding='utf-8')
            command=[sys.executable,str(ROOT/'scripts/check_data.py'),'--data-dir',str(data),'--manifest',str(manifest),'--sha256']
            good=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            (data/'fixture.dat').write_bytes(b'corrupt!')
            bad=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)


if __name__ == '__main__':unittest.main()
