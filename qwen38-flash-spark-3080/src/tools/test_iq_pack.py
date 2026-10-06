"""Focused pack compatibility tests; run .venv/bin/python -m unittest discover -s tools -p test_iq_pack.py."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
import iq_pack


def write_gguf(path, tensors):
    writer = GGUFWriter(path, "qwen4exp")
    for name, values, kind in tensors:
        writer.add_tensor(name, quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class CompatibilityTests(unittest.TestCase):
    def test_bf16_halfway_rounds_to_even(self):
        values = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32)
        got = np.frombuffer(iq_pack.bf16_bytes(values.view(np.uint8), "F32"), dtype=np.uint16)
        np.testing.assert_array_equal(got, [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_nonfinite_conversion_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            iq_pack.bf16_bytes(np.array([np.nan], dtype=np.float32).view(np.uint8), "F32")

    def test_projection_conversion_and_native_bytes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
            router = np.full((2, 128), 0.10001, dtype=np.float32)
            names = ["blk.0.hc_attn_down.weight", "output_hc_up.weight", "blk.1.ple_key.weight",
                     "blk.0.ssm_alpha.weight", "blk.3.indexer.q_proj.weight", "blk.1.ple_value.weight"]
            write_gguf(source, [(n, values, Q.Q8_0) for n in names] + [
                ("blk.0.ffn_gate_inp.weight", router, Q.F32),
                ("blk.0.attn_qkv.weight", values, Q.Q8_0),
                ("per_layer_token_embd.weight", values, Q.Q8_0),
            ])
            before = source.read_bytes()
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model, True), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            # A scalar oracle independent of the BF16 encoder: choose the nearest BF16 value,
            # using the even representation at exact ties.
            decoded = quants.dequantize(quants.quantize(values, Q.Q8_0), Q.Q8_0).ravel()
            def rounded(x):
                bits = int(np.float32(x).view(np.uint32))
                return (bits >> 16) + ((bits & 65535) > 32768 or
                                      ((bits & 65535) == 32768 and (bits >> 16) & 1))
            expected = np.array([rounded(x) for x in decoded], dtype=np.uint16)
            for name in names:
                row = rows[name]
                self.assertEqual(row[2], "4")
                offset, size = int(row[3]), int(row[4])
                np.testing.assert_array_equal(np.frombuffer(dense[offset:offset + size], dtype=np.uint16), expected)
            native = rows["blk.0.attn_qkv.weight"]
            self.assertEqual(native[4], "0")
            self.assertEqual(native[9], "8")
            self.assertNotIn("per_layer_token_embd.weight", rows)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(json.loads((root / "compat-bf16.json").read_text())["tensors"]), 7)

    def test_default_still_refuses_inexact_f32_router(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.ffn_gate_inp.weight", np.full((2, 32), 0.10001, np.float32), Q.F32)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)

    def test_existing_bf16_pack_remains_byte_identical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-2, 2, 256, dtype=np.float32).reshape(2, 128)
            write_gguf(source, [("blk.0.hc_attn_down.weight", values, Q.BF16)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 0)
            self.assertEqual((root / "dense.bin").read_bytes(), quants.quantize(values, Q.BF16).tobytes())

    def test_quantized_control_weights_need_explicit_conversion(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0)])
            (root / "dense.bin").write_bytes(b"previous pack")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)
            self.assertEqual((root / "dense.bin").read_bytes(), b"previous pack")

    def test_q2_ple_and_large_tensors_stay_native(self):
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q2_0"))
        for name in ["blk.0.ffn_gate_exps.weight", "token_embd.weight", "output.weight",
                     "per_layer_token_embd.weight", "blk.0.attn_qkv.weight"]:
            self.assertFalse(iq_pack.needs_bf16(name, "IQ3_XXS"))

    def test_snapshot_symlinks_keep_split_discovery(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snapshot"
            snap.mkdir()
            first = snap / "model-00001-of-00002.gguf"
            second = snap / "model-00002-of-00002.gguf"
            write_gguf(root / "blob1", [
                ("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", np.ones((512, 32), np.float32), Q.BF16),
            ])
            write_gguf(root / "blob2", [(f"blk.0.ffn_{r}_exps.weight", np.ones((512, 2, 32), np.float32), Q.Q8_0)
                                       for r in ("gate", "up", "down")])
            try:
                first.symlink_to(root / "blob1")
            except OSError:                              # Windows without Developer Mode or admin rights
                self.skipTest("symlinks are not available here")
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards"):
                iq_pack.Model(first)
            second.symlink_to(root / "blob2")
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--compat-bf16"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(iq_pack.main(), 0)
            expert_index = (out / "native_experts.txt").read_text()
            self.assertIn(second.name, expert_index)
            self.assertIn("n_expert 512,", expert_index)
            (root / "blob2").write_bytes((root / "blob2").read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(first)


if __name__ == "__main__":
    unittest.main()
