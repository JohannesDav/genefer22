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
        for name in (
            "genefer-boinc-checked",
            "genefer-boinc-stock",
            "genefer-boinc-two",
            "genefer-boinc-upstream",
        ):
            path = self.bundle / "bin" / name
            path.write_text(
                '#!/usr/bin/python3\nimport json,os,sys\nprint(json.dumps(dict(backend=os.path.basename(__file__),pid=os.getpid(),argv=sys.argv[1:],cuda=os.getenv("CUDA_TEST"))))\n'
            )
            path.chmod(0o755)
        shutil.copy2(ROOT / "runtime/routing.py", self.bundle / "routing.py")
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
            (["-boinc", "-n19", "-b528220158", "-p"], True),
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
                    (
                        "genefer-boinc-upstream"
                        if "-b528220158" in arguments
                        else "genefer-boinc-checked"
                    )
                    if optimized
                    else "genefer-boinc-stock",
                )
                if not optimized:
                    self.assertIsNone(actual["cuda"])
                    self.assertNotIn("--guard-bundle", actual["argv"])

    def test_automatic_exact_ranges(self):
        for base, expected in [
            (1026, "two"),
            (2072558, "two"),
            (2072560, "two"),
            (4038248, "two"),
            (4038250, "three"),
            (4194302, "three"),
            (4194306, "three"),
            (528220156, "three"),
            (528220158, "upstream"),
            (2000000000, "upstream"),
        ]:
            with self.subTest(base=base):
                result = self.launch(
                    ["-boinc", "-n19", "-b" + str(base), "-p", "--check-tail"]
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads(result.stdout)
                self.assertIn("--check-tail", actual["argv"])
                binary = {
                    "two": "genefer-boinc-two",
                    "three": "genefer-boinc-checked",
                    "upstream": "genefer-boinc-upstream",
                }[expected]
                self.assertEqual(actual["backend"], binary)
                diagnostic = json.loads(
                    (self.bundle / ".gfn19-dispatch.json").read_text()
                )
                self.assertEqual(diagnostic["arithmetic"], expected)

    def test_explicit_upstream_preserves_requested_checks(self):
        result = self.launch(
            ["-boinc", "-n19", "-b28000000", "-p", "--upstream", "--check-certificate"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual["backend"], "genefer-boinc-upstream")
        self.assertIn("--check-certificate", actual["argv"])

    def test_unsupported_checks_not_silently_ignored(self):
        result = self.launch(["-boinc", "-n20", "-b28000000", "-p", "--check-tail"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("optional checks require", result.stderr)

    def test_removed_experimental_option_rejected(self):
        result = self.launch(["-boinc", "-n19", "-b28000000", "-p", "--two-prime"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was removed", result.stderr)

    def test_upstream_threshold_inequalities(self):
        spec = importlib.util.spec_from_file_location(
            "routing", ROOT / "runtime/routing.py"
        )
        routes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routes)
        for bound, width, primes in [
            (2072558, 31, routes.SMALL),
            (4038248, 32, routes.WIDE),
        ]:
            limit = (primes[0] * primes[1] // 2) // (1 << 19)
            self.assertLess(bound * bound, limit)
            self.assertGreaterEqual((bound + 2) ** 2, limit)
            self.assertEqual(routes.two_prime_width(bound), width)
        self.assertIsNone(routes.two_prime_width(4038250))

    def test_native_result_comes_from_result_file(self):
        spec = importlib.util.spec_from_file_location(
            "checked_result", ROOT / "runtime/run_checked.py"
        )
        checked = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checked)
        binary = self.bundle / "native-prover"
        binary.write_text(
            "#!/usr/bin/python3\n"
            'print("proof file is generated")\n'
            'with open("results.txt", "a") as f: f.write("new native result\\n")\n'
        )
        binary.chmod(0o755)
        (self.bundle / "results.txt").write_text("stale native result\n")
        text, _ = checked.command(
            binary,
            "-p",
            1026,
            0,
            self.bundle / "gproof",
            self.bundle,
            self.bundle / "proof.log",
            checked.clean_environment(),
        )
        self.assertIn("proof file is generated", text)
        self.assertIn("new native result", text)
        self.assertNotIn("stale native result", text)

    def test_optional_checks_are_independent(self):
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location(
            "checked_flags", ROOT / "runtime/run_checked.py"
        )
        checked = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checked)
        for name in ["genefer-conservative", "genefer-gfn19", "genefer-gfn19-upstream"]:
            (self.bundle / "bin" / name).write_text("test binary")
        kernel = self.bundle / "kernel.cl"
        kernel.write_text("test kernel")
        certificate = self.bundle / "certificate"
        certificate.write_text("test certificate")
        manifest = dict(
            gpu="3090",
            arithmetic="three",
            backend="opencl",
            kernel="kernel.cl",
            prover="bin/genefer-gfn19",
        )
        key = "0123456789ABCDEF"
        residue = "ABCDEF0123456789"

        def prover(binary, mode, base, device, prefix, cwd, log, env):
            self.assertEqual(mode, "-p")
            prefix.with_suffix(".proof").write_bytes(b"test proof")
            if env.get("GENEFER_CHECK_TAIL") == "1":
                prefix.with_suffix(".tail").write_bytes(b"test checkpoint")
            return (
                f"{base}^{{2^19}} + 1 is composite, pkey = {key}, res64 = {residue}\nproof file is generated",
                1.0,
            )

        for cert, tail in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(certificate=cert, tail=tail):
                self.manifest()
                prefix = self.bundle / f"candidate-{int(cert)}-{int(tail)}"
                result = dict(
                    probable_prime=False,
                    proof_key=key,
                    residue64=residue,
                    certificate_path=str(certificate),
                )
                with patch.object(
                    checked, "arithmetic_manifest", return_value=manifest
                ), patch.object(checked, "configure_runtime"), patch.object(
                    checked, "command", side_effect=prover
                ), patch.object(
                    checked, "proof_header"
                ), patch.object(
                    checked, "verify", return_value=result
                ) as certificate_call, patch.object(
                    checked, "verify_tail", return_value=dict(tail_verified=True)
                ) as tail_call:
                    report = checked.execute(
                        1026,
                        prefix,
                        bundle=self.bundle,
                        check_certificate=cert,
                        check_tail=tail,
                    )
                self.assertEqual(certificate_call.call_count, int(cert))
                self.assertEqual(tail_call.call_count, int(tail))
                self.assertEqual(report["certificate_verified"], cert)
                self.assertEqual(report["tail_verified"], tail)
                self.assertEqual(report["validated"], cert and tail)
                self.assertEqual(prefix.with_suffix(".cert").exists(), cert)
                self.assertEqual(
                    (Path(str(prefix) + ".work") / "fast/gproof.tail").exists(), tail
                )

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
        spec = importlib.util.spec_from_file_location(
            "guard_verify", ROOT / "runtime/boinc_guard_verify.py"
        )
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        prefix = self.bundle / "candidate"
        identity = {"base": 28000000, "arithmetic": "three"}
        guard.check_owner(prefix, identity, True)
        guard.check_owner(prefix, identity, False)
        with self.assertRaisesRegex(
            guard.GuardError, "checkpoint owner/backend changed"
        ):
            guard.check_owner(prefix, dict(identity, arithmetic="two"), False)
        self.assertEqual(
            json.loads((self.bundle / "candidate.backend.json").read_text()), identity
        )

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
