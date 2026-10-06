"""The streaming detokenizer (perf-review F-1): the incremental one emits exactly what the old re-decode did, token
by token, and its cost per token does not grow with the reply.

    python -m unittest serve.test_detok      (needs a pack's tokenizer; skipped without one)
"""
import os
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from serve.server import Detokenizer  # noqa: E402


def find_tokenizer():
    cands = [os.environ.get("STRATA_TOKENIZER", "")]
    cands += [str(p) for p in ROOT.glob("packs/*/tokenizer")] + [str(p) for p in ROOT.glob("pack/*/tokenizer")]
    cands += [r"/path/to/strata\Public\Engine\pack\full\tokenizer"]
    for c in cands:
        if c and (Path(c) / "vocab.json").exists():
            return Path(c)
    return None


class OldDetokenizer:
    """The pre-0.1.22 algorithm: re-decode every generated id, hold a trailing U+FFFD."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, t):
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


@unittest.skipIf(find_tokenizer() is None, "no pack tokenizer here")
class Detok(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        import strata_tokenizer as ST
        t = find_tokenizer()
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        toks = [None] * len(vocab)
        for s, i in vocab.items():
            toks[i] = s
        cls.tok = ST.Tokenizer(toks, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                               json.loads((t / "token_type.json").read_text()))

    def same_stream(self, ids):
        """The same text as the old re-decode, never behind it (the old one held complete characters too while
        the tail was an incomplete one; the new one holds only the incomplete bytes), and never ahead of what the
        ids so far decode to."""
        a, b = OldDetokenizer(self.tok), Detokenizer(self.tok)
        self.assertIsNotNone(b.inc)
        old = new = ""
        for i, t in enumerate(ids):
            old += a.push(t)
            new += b.push(t)
            self.assertTrue(new.startswith(old), f"token {i} ({t}): behind the old stream")
            self.assertTrue(self.tok.decode(ids[:i + 1]).startswith(new), f"token {i} ({t}): ahead of the ids")
        if not self.tok.decode(ids).endswith("�"):
            self.assertEqual(old, new)

    def test_text_with_split_characters(self):
        text = ("Grüße aus Zürich! 你好，世界。日本語のテキスト 🌊🦊👩‍💻 and ASCII code: def f(x): return x**2\n" * 40)
        self.same_stream(self.tok.encode(text))

    def test_random_ids_and_broken_bytes(self):
        rng = random.Random(5)
        n = len(self.tok.tokens)
        for _ in range(20):   # random ids: split and invalid UTF-8 sequences everywhere (no U+FFFD token itself)
            ids = [rng.randrange(n) for _ in range(300)]
            ids = [t for t in ids if "\ufffd" not in self.tok.decode([t], errors="strict" if False else "replace")
                   or len(self.tok.token_bytes(t)) < 3]
            self.same_stream(ids)

    def test_constant_cost(self):
        ids = self.tok.encode("The quick brown fox jumps over the lazy dog. " * 2000)[:16384]
        d = Detokenizer(self.tok)
        for t in ids[:1024]:
            d.push(t)
        t0 = time.perf_counter()
        for t in ids[1024:1124]:
            d.push(t)
        early = time.perf_counter() - t0
        for t in ids[1124:16000]:
            d.push(t)
        t0 = time.perf_counter()
        for t in ids[16000:16100]:
            d.push(t)
        late = time.perf_counter() - t0
        self.assertLess(late / 100, 0.0002, "a token after 16K tokens costs more than 0.2 ms")
        self.assertLess(late, early * 5 + 0.002)


if __name__ == "__main__":
    unittest.main()
