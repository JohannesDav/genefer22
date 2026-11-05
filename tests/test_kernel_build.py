"""Check that PTX packaging preserves kernels and refuses an unexpected ABI."""
import importlib.util
from pathlib import Path
import re
import unittest

spec = importlib.util.spec_from_file_location(
    "kernel_builder", Path(__file__).resolve().parents[1] / "kernels/build.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def module():
    entries = []
    for name in sorted(builder.KERNELS):
        body = "ret;"
        if name == "boundary_bridge_signed":
            body = "ld.param.u32 %r7, [boundary_bridge_signed_param_7];\nret;"
        entries.append(".entry " + name + "()\n{\n" + body + "\n}")
    return "\n\n".join(entries) + "\n"


class KernelBuildTests(unittest.TestCase):
    def test_specialization_preserves_original_entries(self):
        original = module()
        compiled = builder.assemble(original, "3090")
        before, after = builder.entry_regions(original), builder.entry_regions(compiled)
        for name, (start, _, end) in before.items():
            a, _, b = after[name]
            self.assertEqual(original[start:end], compiled[a:b])
        for bit in (0, 1):
            name = "boundary_bridge_signed_dup" + str(bit)
            a, _, b = after[name]
            self.assertIn("mov.u32 %r7, " + str(bit) + ";", compiled[a:b])
            self.assertNotIn("ld.param.u32", compiled[a:b])

    def test_caps_apply_only_to_selected_functions(self):
        compiled = builder.assemble(module(), "4060")
        caps = {}
        for name, (a, opening, _) in builder.entry_regions(compiled).items():
            match = re.search(r"\.maxnreg (\d+)", compiled[a:opening])
            if match:
                caps[name] = int(match[1])
        self.assertEqual(
            caps,
            {
                "square2048": 56,
                "boundary_bridge_signed": 64,
                "boundary_bridge_signed_dup0": 64,
                "boundary_bridge_signed_dup1": 64,
            },
        )

    def test_missing_or_duplicate_flag_load_rejected(self):
        load = "ld.param.u32 %r7, [boundary_bridge_signed_param_7];"
        for replacement in ("", load + "\n" + load):
            with self.subTest(replacement=replacement):
                with self.assertRaisesRegex(RuntimeError, "one bridge"):
                    builder.assemble(module().replace(load, replacement), "3090")

    def test_source_dimensions_match_runtime(self):
        for gpu in ("3090", "4060"):
            folder = builder.ROOT / "kernels/src"
            source = (
                (folder / ("gfn19-" + gpu + ".cl"))
                .read_text()
                .replace('#include "common.clh"', (folder / "common.clh").read_text())
            )
            identifier = builder.configuration(gpu)
            builder.validate_configuration(source, identifier)
            with self.assertRaisesRegex(RuntimeError, "N_SZ"):
                builder.validate_configuration(
                    source.replace("#define N_SZ 524288u", "#define N_SZ 1048576u"),
                    identifier,
                )

    def test_unexpected_kernel_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "kernel set"):
            builder.assemble(module() + "\n.entry unexpected() { ret; }", "3090")


if __name__ == "__main__":
    unittest.main()
