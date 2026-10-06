"""A missing or short model shard is named with its numbers, and a pack's verify reads back the source hash
its manifest recorded - over a minimal GGUF written here (no download, no model, no GPU).

    python -m unittest tools.test_shards
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
import strata_pack  # noqa: E402


class Stop(Exception):
    """setup.fail(), caught instead of exiting: the message is what the test reads."""


def write_gguf(path: Path, names=("blk.0.attn_q.weight", "blk.0.attn_k.weight")) -> int:
    """A GGUF v3 of F32[8] tensors 32 bytes apart from a 32-byte-aligned data start; returns its whole length."""
    b = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(names), 0))
    for i, n in enumerate(names):
        b += struct.pack("<Q", len(n)) + n.encode() + struct.pack("<IQIQ", 1, 8, 0, 32 * i)
    b += bytes(-len(b) % 32 + 32 * len(names))
    path.write_bytes(b)
    return len(b)


class ShardCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved = S.fail
        S.fail = lambda msg, hint=None: (_ for _ in ()).throw(Stop(msg))

    def tearDown(self):
        S.fail = self.saved
        self.tmp.cleanup()

    def test_whole_shards_pass(self):
        shards = [self.dir / f"m-0000{i}-of-00002.gguf" for i in (1, 2)]
        for s in shards:
            write_gguf(s)
        S.check_shards(shards)

    def test_missing_shard_is_named(self):
        shards = [self.dir / "m-00001-of-00002.gguf", self.dir / "m-00002-of-00002.gguf"]
        write_gguf(shards[0])
        with self.assertRaises(Stop) as cm:
            S.check_shards(shards)
        self.assertIn("m-00002-of-00002.gguf", str(cm.exception))

    def test_short_shard_names_file_and_sizes(self):
        s = self.dir / "m-00001-of-00002.gguf"
        whole = write_gguf(s)
        s.write_bytes(s.read_bytes()[:-40])
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        msg = str(cm.exception)
        self.assertIn("m-00001-of-00002.gguf is short", msg)
        self.assertIn(f"{whole - 40:,} of {whole:,} bytes", msg)
        self.assertIn("40 missing", msg)

    def test_truncated_header_is_refused(self):
        s = self.dir / "m-00001-of-00002.gguf"
        write_gguf(s)
        s.write_bytes(s.read_bytes()[:12])
        with self.assertRaises(Stop) as cm:
            S.check_shards([s])
        self.assertIn("m-00001-of-00002.gguf is not a whole GGUF shard", str(cm.exception))


class PackVerifyHash(unittest.TestCase):
    """verify over a manifest with no tensor entries: the only check left is the source hash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gguf = self.dir / "m-00001-of-00002.gguf"
        write_gguf(self.gguf)
        self.digest = hashlib.sha256(self.gguf.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def run_verify(self, source, limit=None):
        (self.dir / "manifest.json").write_text(json.dumps({"tensors": {}, "source": source}), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = strata_pack.verify(self.gguf, self.dir, limit, 0)
        return rc, out.getvalue()

    def test_matching_hash_passes(self):
        rc, _ = self.run_verify({"shard1_sha256": self.digest})
        self.assertEqual(rc, 0)

    def test_wrong_hash_fails_naming_both(self):
        rc, out = self.run_verify({"shard1_sha256": "0" * 64})
        self.assertEqual(rc, 1)
        self.assertIn("m-00001-of-00002.gguf: sha256 " + self.digest, out)
        self.assertIn("built from " + "0" * 64, out)

    def test_no_hash_or_limit_skips(self):
        self.assertEqual(self.run_verify({})[0], 0)                                  # setup builds --skip-hash
        self.assertEqual(self.run_verify({"shard1_sha256": "0" * 64}, limit=1)[0], 0)  # --limit stays quick


if __name__ == "__main__":
    unittest.main()
