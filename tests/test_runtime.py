"""Exercise dispatch and proof-input checks without a GPU or BOINC client."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bundle = Path(self.directory.name)
        (self.bundle / "bin").mkdir()
        shutil.copy2(ROOT / "runtime/dispatch.py", self.bundle / "genefer-boinc")
        for name in ("genefer-boinc-checked", "genefer-boinc-stock", "genefer-boinc-two-prime"):
            path = self.bundle / "bin" / name
            path.write_text(
                '#!/usr/bin/python3\nimport json,os,sys\nprint(json.dumps(dict(backend=os.path.basename(__file__),pid=os.getpid(),argv=sys.argv[1:],cuda=os.getenv("CUDA_TEST"))))\n'
            )
            path.chmod(0o755)
        self.manifest()

    def manifest(self):
        files = {
            str(path.relative_to(self.bundle)): sha(path)
            for path in self.bundle.rglob("*")
            if path.is_file() and path.name != "deployment-manifest.json"
        }
        (self.bundle / "deployment-manifest.json").write_text(
            json.dumps(dict(version=1, gpu="3090", files=files))
        )

    def launch(self, arguments):
        return subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                str(self.bundle / "genefer-boinc"),
                "--guard-bundle",
                str(self.bundle),
            ]
            + arguments,
            cwd=self.bundle,
            env=dict(os.environ, CUDA_TEST="inherited"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_routes_and_exec(self):
        for arguments, optimized in [
            (["-boinc", "-n19", "-b28000000", "-p"], True),
            (["-boinc", "-n19", "-b30000002", "-p"], True),
            (["-boinc", "-n19", "-b528220156", "-p"], True),
            (["-boinc", "-n19", "-b528220158", "-p"], False),
            (["-boinc", "-n19", "-b30000002", "-c"], True),
            (["-boinc", "-n19", "-b1024", "-p"], False),
            (["-boinc", "-n19", "-b1027", "-p"], False),
            (["-boinc", "-n20", "-b28000000", "-p"], False),
            (["-boinc", "-n19", "-b28000000", "-p", "-d1"], False),
            (["-boinc", "-n19", "-b28000000", "-p", "--upstream"], False),
            (["-n19", "-b28000000", "-p"], False),
        ]:
            with self.subTest(arguments=arguments):
                result = self.launch(arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads(result.stdout)
                diagnostic = json.loads(
                    (self.bundle / ".gfn19-dispatch.json").read_text()
                )
                self.assertEqual(actual["pid"], diagnostic["pid"])
                self.assertEqual(
                    actual["backend"],
                    "genefer-boinc-checked" if optimized else "genefer-boinc-stock",
                )
                if not optimized:
                    self.assertIsNone(actual["cuda"])
                    self.assertNotIn("--guard-bundle", actual["argv"])

    def test_explicit_two_prime_routes(self):
        for base, expected in [(28000000, "two"), (30000000, "two"),
                               (30000002, "three"), (528220156, "three")]:
            with self.subTest(base=base):
                result=self.launch(["-boinc", "-n19", "-b"+str(base), "-p", "--two-prime"])
                self.assertEqual(result.returncode,0,result.stderr)
                actual=json.loads(result.stdout)
                self.assertNotIn("--two-prime",actual["argv"])
                expected_binary="genefer-boinc-two-prime" if expected=="two" else "genefer-boinc-checked"
                self.assertEqual(actual["backend"],expected_binary)
                diagnostic=json.loads((self.bundle/".gfn19-dispatch.json").read_text())
                self.assertEqual(diagnostic["arithmetic"],expected)

    def test_changed_bundle_rejected(self):
        (self.bundle / "bin/genefer-boinc-checked").write_text("changed")
        result = self.launch(["-boinc", "-n19", "-b28000000", "-p"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("deployment file changed", result.stderr)
        self.assertFalse((self.bundle / ".gfn19-dispatch.json").exists())

    def test_unsafe_manifest_path_rejected(self):
        manifest = dict(version=1, gpu="3090", files={"../outside": "unused"})
        (self.bundle / "deployment-manifest.json").write_text(json.dumps(manifest))
        result = self.launch(["-boinc", "-n19", "-b28000000", "-p"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe manifest path", result.stderr)

    def test_checkpoint_mode_mismatch_rejected(self):
        spec=importlib.util.spec_from_file_location("guard_verify", ROOT/"runtime/boinc_guard_verify.py")
        guard=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        prefix=self.bundle/"candidate"
        identity={"base":28000000,"arithmetic":"three"}
        guard.check_owner(prefix,identity,True)
        guard.check_owner(prefix,identity,False)
        with self.assertRaisesRegex(guard.GuardError,"checkpoint owner/backend changed"):
            guard.check_owner(prefix,dict(identity,arithmetic="two"),False)
        self.assertEqual(json.loads((self.bundle/"candidate.backend.json").read_text()),identity)

    def test_truncated_proof_rejected(self):
        spec = importlib.util.spec_from_file_location(
            "checked", ROOT / "runtime/run_checked.py"
        )
        checked = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checked)
        proof = self.bundle / "truncated.proof"
        proof.write_bytes(b"bad")
        with self.assertRaises(checked.ValidationError):
            checked.proof_header(proof, 28000000)
        self.assertFalse(
            any(name.startswith("CUDA_") for name in checked.clean_environment())
        )


if __name__ == "__main__":
    unittest.main()
