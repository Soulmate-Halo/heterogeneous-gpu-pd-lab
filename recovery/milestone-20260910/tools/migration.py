# -*- coding: utf-8 -*-
"""Heterogeneous GPU PD lab: backup / recatalog / restore via GitHub CLI.

Stages: prepare, backup, publish, verify, restore-plan, restore.
Remote writes happen only with --execute. JSON is UTF-8 without BOM.
Requires authenticated `gh` on PATH. Does not invoke git.exe and does not
print tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = "Soulmate-Halo/heterogeneous-gpu-pd-lab"
EXPECTED_OLD_MAIN = "a608d07e4d70094b9dd5949abb889095a93c26e6"
EXPECTED_OLD_TREE = "d73412b26bdd9204444a77e6ddd7944de9583564"
ARCHIVE_BRANCH = "archive/pre-milestones-20260910"
ARCHIVE_REF = "refs/heads/archive/pre-milestones-20260910"
RECOVERY_ROOT = "recovery/milestone-20260910"
ALLOWED_DOC_FILES = (
    "README.md",
    "README_ZH.md",
    "CHANGELOG.md",
    "CHANGELOG_ZH.md",
    "VERSION",
    "VERSION_HISTORY.md",
)
PUBLIC_MILESTONES = ("v1.0", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6")
CREATED_TAGS = ("v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6")
FROZEN_MILESTONE_COMMITS = {
    "v1.0": "b36c383b68b349937a568846f969cdb7534160c3",
    "v1.1": "0814893d8778af874c7b704444087a07b6c4fa6b",
    "v1.2": "5472a50bb98dcb730c08aa4967d7a699875410d6",
    "v1.3": "a22229a47c40af720046ac7aab63553b32da1e28",
    "v1.4": "898b81898e57f63dd722f30f1ffae7e78a453fe7",
    "v1.5": "ad27c08e92e745009613c1ca3d829e3bb5c8ed1c",
}
UNTAGGED_LOCATE = {
    "v2.5": {
        "public": "v1.2",
        "prefix": "v2.5:",
        "version_text": "2.5",
        "frozen_tip": "5472a50bb98dcb730c08aa4967d7a699875410d6",
    },
    "v2.6": {
        "public": "v1.3",
        "prefix": "v2.6:",
        "version_text": "2.6",
        "frozen_tip": "a22229a47c40af720046ac7aab63553b32da1e28",
    },
    "v2.11": {
        "public": "v1.4",
        "prefix": "v2.11:",
        "version_text": "2.11",
        "frozen_tip": "898b81898e57f63dd722f30f1ffae7e78a453fe7",
    },
}
IRREVERSIBLE_NOTE = (
    "GitHub numeric release id, original published_at/created_at, and asset "
    "download_count cannot be restored after delete. Restore recreates a new "
    "release with the same tag_name, name, body, draft, prerelease and "
    "target_commitish only."
)


class Fail(Exception):
    """Hard failure; journaled by the caller."""


class Block(Exception):
    """Cannot determine a required historical mapping; do not guess."""


class GhError(Exception):
    def __init__(self, rc, method, endpoint, stderr):
        super().__init__(f"gh {method} {endpoint} rc={rc}")
        self.rc = rc
        self.method = method
        self.endpoint = endpoint
        self.stderr = stderr


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(text.encode("utf-8"))  # no BOM


def read_json(path: Path):
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8")
    return json.loads(text)


def append_journal(journal_path: Path, event: str, **fields) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utc_now(), "event": event}
    rec.update(fields)
    line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
    with journal_path.open("ab") as fh:
        fh.write(line.encode("utf-8"))


def default_gh_runner(cmd, input_bytes=None):
    return subprocess.run(cmd, input=input_bytes, capture_output=True)


class GhClient:
    def __init__(self, exe="gh", runner=None, journal_path=None):
        self.exe = exe
        self.runner = runner or default_gh_runner
        self.journal_path = journal_path

    def _journal(self, event, **fields):
        if self.journal_path:
            append_journal(self.journal_path, event, **fields)

    def raw(self, args, input_bytes=None, binary=False):
        cmd = [self.exe, *args]
        proc = self.runner(cmd, input_bytes=input_bytes)
        rc = proc.returncode
        stdout = proc.stdout or b""
        stderr = proc.stderr or b""
        if rc != 0:
            err = stderr.decode("utf-8", errors="replace")
            self._journal(
                "subprocess_fail",
                args=args,
                rc=rc,
                stderr=err[:4000],
            )
            raise GhError(rc, args[0] if args else "", " ".join(args[1:4]), err)
        if binary:
            return stdout
        text = stdout.decode("utf-8")
        if text.startswith("\ufeff"):
            text = text.lstrip("\ufeff")
        return text

    def api(self, method, endpoint, body=None, paginate=False, binary=False):
        args = ["api", "--method", method, endpoint]
        if paginate:
            args.append("--paginate")
        input_bytes = None
        tmp = None
        try:
            if body is not None:
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                tmp = tempfile.NamedTemporaryFile(prefix="gh-body-", suffix=".json", delete=False)
                tmp.write(payload)
                tmp.close()
                args.extend(["--input", tmp.name])
            text_or_bytes = self.raw(args, input_bytes=input_bytes, binary=binary)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        if binary:
            return text_or_bytes
        text = text_or_bytes
        if not str(text).strip():
            return None
        return json.loads(text)

    def api_ok(self, method, endpoint):
        try:
            self.api(method, endpoint)
            return True
        except GhError as exc:
            if exc.rc in (1, 44) or "404" in (exc.stderr or ""):
                return False
            raise


def first_line(msg):
    return (msg or "").split("\n", 1)[0]


def tag_name_from_ref(ref):
    prefix = "refs/tags/"
    if ref.startswith(prefix):
        return ref[len(prefix) :]
    return ref


def head_name_from_ref(ref):
    prefix = "refs/heads/"
    if ref.startswith(prefix):
        return ref[len(prefix) :]
    return ref


def decode_contents(payload):
    raw = "".join((payload.get("content") or "").split())
    import base64

    return base64.b64decode(raw)


def load_frozen_evidence(script_dir: Path):
    path = script_dir / "evidence" / "historical-sha.json"
    if path.is_file():
        return read_json(path)
    return {}


class Migration:
    def __init__(
        self,
        repo=REPO_DEFAULT,
        work_dir=None,
        docs_dir=None,
        local_snapshot=None,
        execute=False,
        gh=None,
        script_path=None,
        expected_old_main=EXPECTED_OLD_MAIN,
        expected_old_tree=EXPECTED_OLD_TREE,
    ):
        self.repo = repo
        self.script_path = Path(script_path or __file__).resolve()
        self.script_dir = self.script_path.parent
        self.work_dir = Path(work_dir or (self.script_dir / "work")).resolve()
        self.docs_dir = Path(docs_dir).resolve() if docs_dir else None
        self.local_snapshot = Path(local_snapshot).resolve() if local_snapshot else None
        self.execute = bool(execute)
        self.expected_old_main = expected_old_main
        self.expected_old_tree = expected_old_tree
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.work_dir / "journal.jsonl"
        self.gh = gh or GhClient(journal_path=self.journal_path)
        self.gh.journal_path = self.journal_path
        self.frozen = load_frozen_evidence(self.script_dir)

    def journal(self, event, **fields):
        append_journal(self.journal_path, event, **fields)

    def fail(self, msg, **fields):
        self.journal("fail", message=msg, **fields)
        raise Fail(msg)

    def block(self, msg, **fields):
        self.journal("block", message=msg, **fields)
        raise Block(msg)

    def require_no_execute_for_readonly(self, stage):
        if self.execute and stage in ("prepare", "verify", "restore-plan"):
            self.journal("note", message=f"{stage} ignores --execute (read-only)")

    def snapshot_refs(self):
        tags = self.gh.api("GET", f"repos/{self.repo}/git/matching-refs/tags", paginate=True) or []
        heads = self.gh.api("GET", f"repos/{self.repo}/git/matching-refs/heads", paginate=True) or []
        return {"tags": tags, "heads": heads}

    def resolve_tag_object(self, ref_entry):
        sha = ref_entry["object"]["sha"]
        typ = ref_entry["object"]["type"]
        chain = []
        seen = set()
        while typ == "tag":
            if sha in seen:
                self.fail("annotated tag cycle", sha=sha)
            seen.add(sha)
            obj = self.gh.api("GET", f"repos/{self.repo}/git/tags/{sha}")
            chain.append(
                {
                    "sha": obj.get("sha") or sha,
                    "tag": obj.get("tag"),
                    "message": obj.get("message"),
                    "object": obj.get("object"),
                    "tagger": obj.get("tagger"),
                }
            )
            sha = obj["object"]["sha"]
            typ = obj["object"]["type"]
        if typ != "commit":
            self.fail("tag did not resolve to commit", type=typ, sha=sha)
        return {"resolved_commit": sha, "resolved_type": typ, "annotated_chain": chain}

    def get_commit(self, sha):
        return self.gh.api("GET", f"repos/{self.repo}/git/commits/{sha}")

    def get_tree_map(self, tree_sha):
        data = self.gh.api("GET", f"repos/{self.repo}/git/trees/{tree_sha}?recursive=1")
        blobs = {}
        for entry in data.get("tree") or []:
            if entry.get("type") == "blob":
                blobs[entry["path"]] = {"sha": entry["sha"], "size": entry.get("size"), "mode": entry.get("mode")}
        return {
            "sha": data.get("sha") or tree_sha,
            "truncated": bool(data.get("truncated")),
            "blobs": blobs,
        }

    def snapshot_releases(self):
        rels = self.gh.api("GET", f"repos/{self.repo}/releases?per_page=100", paginate=True) or []
        out = []
        for r in rels:
            assets = r.get("assets") or []
            out.append(
                {
                    "id": r.get("id"),
                    "tag_name": r.get("tag_name"),
                    "name": r.get("name"),
                    "body": r.get("body") or "",
                    "draft": bool(r.get("draft")),
                    "prerelease": bool(r.get("prerelease")),
                    "target_commitish": r.get("target_commitish"),
                    "published_at": r.get("published_at"),
                    "created_at": r.get("created_at"),
                    "html_url": r.get("html_url"),
                    "download_count_sum": sum(int(a.get("download_count") or 0) for a in assets),
                    "assets": [
                        {
                            "id": a.get("id"),
                            "name": a.get("name"),
                            "download_count": a.get("download_count"),
                        }
                        for a in assets
                    ],
                }
            )
        return out

    def file_text_at(self, path, ref):
        try:
            payload = self.gh.api("GET", f"repos/{self.repo}/contents/{path}?ref={ref}")
        except GhError as exc:
            if "404" in (exc.stderr or ""):
                return None
            raise
        return decode_contents(payload).decode("utf-8")

    def locate_untagged_series(self, spec, commits):
        prefix = spec["prefix"]
        matched = []
        for c in commits:
            msg = first_line(c.get("commit", {}).get("message"))
            if msg.startswith(prefix) or msg.startswith(prefix.rstrip(":")):
                matched.append(c)
        if not matched:
            return None
        tip = matched[0]
        sha = tip["sha"]
        version_text = self.file_text_at("VERSION", sha)
        if version_text is None or version_text.strip() != spec["version_text"]:
            self.block(
                f"cannot confirm VERSION for {spec['prefix']}",
                commit=sha,
                version_text=version_text,
            )
        if sha != spec["frozen_tip"]:
            self.block(
                f"live tip for {spec['prefix']} disagrees with frozen evidence; refusing to guess",
                live=sha,
                frozen=spec["frozen_tip"],
            )
        return {
            "tip_commit": sha,
            "committer_date": tip.get("commit", {}).get("committer", {}).get("date"),
            "message": first_line(tip.get("commit", {}).get("message")),
            "series": [c["sha"] for c in matched],
            "VERSION": spec["version_text"],
        }

    def list_commits_from(self, sha):
        return self.gh.api("GET", f"repos/{self.repo}/commits?sha={sha}&per_page=100", paginate=True) or []

    def download_zipball(self, sha, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.gh.api("GET", f"repos/{self.repo}/zipball/{sha}", binary=True)
        dest.write_bytes(data)
        digest = sha256_bytes(data)
        try:
            zf = zipfile.ZipFile(dest)
            names = zf.namelist()
            zf.close()
        except zipfile.BadZipFile:
            self.fail("zipball is not a valid zip", sha=sha)
        if not names:
            self.fail("zipball empty", sha=sha)
        return {"path": str(dest), "sha256": digest, "nbytes": len(data), "entries": len(names)}

    def snapshot_local_docs(self, dest: Path, source: Path):
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ALLOWED_DOC_FILES:
            src = source / name
            if not src.is_file():
                files[name] = {"present": False}
                continue
            data = src.read_bytes()
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[name] = {
                "present": True,
                "sha256": sha256_bytes(data),
                "git_blob_sha": git_blob_sha(data),
                "nbytes": len(data),
            }
        write_json(dest / "manifest.json", {"source": str(source), "files": files})
        return files

    def build_tag_snapshot(self, tag_refs):
        out = []
        for entry in tag_refs:
            name = tag_name_from_ref(entry["ref"])
            resolved = self.resolve_tag_object(entry)
            commit = self.get_commit(resolved["resolved_commit"])
            out.append(
                {
                    "name": name,
                    "ref": entry["ref"],
                    "ref_object_type": entry["object"]["type"],
                    "ref_object_sha": entry["object"]["sha"],
                    "resolved_commit": resolved["resolved_commit"],
                    "annotated_chain": resolved["annotated_chain"],
                    "commit_tree": commit["tree"]["sha"],
                    "commit_parents": [p["sha"] for p in commit.get("parents") or []],
                    "commit_message": commit.get("message"),
                }
            )
        out.sort(key=lambda x: x["name"])
        return out

    def prepare(self):
        self.require_no_execute_for_readonly("prepare")
        self.journal("stage_start", stage="prepare")
        refs = self.snapshot_refs()
        main_ref = None
        for h in refs["heads"]:
            if h["ref"] == "refs/heads/main":
                main_ref = h
                break
        if main_ref is None:
            self.fail("main branch missing")
        main_sha = main_ref["object"]["sha"]
        if main_sha != self.expected_old_main:
            self.fail("main is not the expected pre-milestone SHA", live=main_sha, expected=self.expected_old_main)
        commit = self.get_commit(main_sha)
        tree_sha = commit["tree"]["sha"]
        if tree_sha != self.expected_old_tree:
            self.fail("main tree is not the expected pre-milestone tree", live=tree_sha, expected=self.expected_old_tree)
        tree = self.get_tree_map(tree_sha)
        if tree["truncated"]:
            self.fail("main tree listing truncated")
        tags = self.build_tag_snapshot(refs["tags"])
        if len(tags) != 21:
            self.fail("expected 21 tags", count=len(tags))
        releases = self.snapshot_releases()
        if len(releases) != 21:
            self.fail("expected 21 releases", count=len(releases))
        locate_anchor = FROZEN_MILESTONE_COMMITS["v1.5"]
        commits = self.list_commits_from(locate_anchor)
        located = {}
        for old_ver, spec in UNTAGGED_LOCATE.items():
            found = self.locate_untagged_series(spec, commits)
            if found is None:
                self.block(f"missing tag {old_ver} cannot be located from commit/VERSION history")
            located[old_ver] = found
        mapping = dict(FROZEN_MILESTONE_COMMITS)
        mapping["v1.6"] = "NEW_MAIN_AFTER_SIX_DOC_FILES"
        state = {
            "stage": "prepare",
            "ts": utc_now(),
            "repo": self.repo,
            "main": {"sha": main_sha, "tree": tree_sha, "parents": [p["sha"] for p in commit.get("parents") or []]},
            "heads": [{"ref": h["ref"], "sha": h["object"]["sha"], "type": h["object"]["type"]} for h in refs["heads"]],
            "tags": tags,
            "releases": releases,
            "untagged_located": located,
            "milestone_tag_targets": mapping,
            "reachability_parents": sorted(
                {t["resolved_commit"] for t in tags} | {main_sha}
            ),
        }
        write_json(self.work_dir / "prepare-state.json", state)
        self.journal("stage_done", stage="prepare", tags=len(tags), releases=len(releases))
        return state

    def load_prepare(self):
        path = self.work_dir / "prepare-state.json"
        if path.is_file():
            return read_json(path)
        return self.prepare()

    def recovery_files(self, prepare, zip_meta, extra_state=None):
        manifest = {
            "repo": self.repo,
            "archive_branch": ARCHIVE_BRANCH,
            "old_main": prepare["main"]["sha"],
            "old_tree": prepare["main"]["tree"],
            "tags": [
                {
                    "name": t["name"],
                    "ref_object_type": t["ref_object_type"],
                    "ref_object_sha": t["ref_object_sha"],
                    "resolved_commit": t["resolved_commit"],
                    "annotated_chain": t["annotated_chain"],
                }
                for t in prepare["tags"]
            ],
            "releases": prepare["releases"],
            "untagged_located": prepare["untagged_located"],
            "milestone_tag_targets": prepare["milestone_tag_targets"],
            "zip": zip_meta,
            "irreversible_on_delete": IRREVERSIBLE_NOTE,
            "generated_at": utc_now(),
        }
        if extra_state:
            manifest["extra"] = extra_state
        readme = self.recovery_readme_text()
        script_bytes = self.script_path.read_bytes()
        files = {
            f"{RECOVERY_ROOT}/manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
            f"{RECOVERY_ROOT}/README.md": readme.encode("utf-8"),
            f"{RECOVERY_ROOT}/tools/migration.py": script_bytes,
            f"{RECOVERY_ROOT}/evidence/historical-sha.json": (
                self.script_dir / "evidence" / "historical-sha.json"
            ).read_bytes()
            if (self.script_dir / "evidence" / "historical-sha.json").is_file()
            else json.dumps(self.frozen, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
            f"{RECOVERY_ROOT}/release-plan.json": (
                self.script_dir / "release-plan.json"
            ).read_bytes()
            if (self.script_dir / "release-plan.json").is_file()
            else b"{}\n",
        }
        return files, manifest

    def recovery_readme_text(self):
        return f"""# 回退包（异机可执行）

本目录在 `{ARCHIVE_BRANCH}` 分支的 `{RECOVERY_ROOT}` 下。不要依赖本机绝对路径。

## 认证

需要已登录的 GitHub CLI（`gh auth status` 成功），权限覆盖该仓库的 contents 与 releases。不要把 token 写进文件或贴到日志。

```
gh auth status
```

## 工作目录

在任意机器：

```
gh api repos/{self.repo}/contents/{RECOVERY_ROOT}/tools/migration.py?ref={ARCHIVE_BRANCH}
```

更直接：检出备份分支后进入本目录。

```
python tools/migration.py --repo {self.repo} --work-dir ./work prepare
python tools/migration.py --repo {self.repo} --work-dir ./work backup --dry-run
python tools/migration.py --repo {self.repo} --work-dir ./work restore-plan
python tools/migration.py --repo {self.repo} --work-dir ./work restore --dry-run
python tools/migration.py --repo {self.repo} --work-dir ./work restore --execute
```

远端写必须显式 `--execute`。先 `restore-plan` 与 `restore --dry-run`，核对后再 `restore --execute`。不要 force-push，不要改写 commit 历史。

## 风险

- 并发有人改 main 或任何 tag：默认拒绝，不 force。
- 删除 Release 后，原 id / published_at / 下载计数不可恢复。
- zip 不上传为公开 Release 资产；远端基准是旧 tree 对象。
"""

    def create_blob(self, data: bytes):
        import base64

        body = {"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"}
        created = self.gh.api("POST", f"repos/{self.repo}/git/blobs", body=body)
        sha = created["sha"]
        expected = git_blob_sha(data)
        if sha != expected:
            self.fail("blob SHA mismatch", live=sha, expected=expected)
        return sha

    def create_tree(self, base_tree, files: dict):
        entries = []
        expected = {}
        for path in sorted(files):
            data = files[path]
            blob_sha = self.create_blob(data)
            expected[path] = blob_sha
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob_sha})
        body = {"base_tree": base_tree, "tree": entries}
        created = self.gh.api("POST", f"repos/{self.repo}/git/trees", body=body)
        return created["sha"], expected

    def create_commit(self, message, tree_sha, parents):
        body = {"message": message, "tree": tree_sha, "parents": list(parents)}
        created = self.gh.api("POST", f"repos/{self.repo}/git/commits", body=body)
        return created["sha"]

    def get_ref(self, ref):
        # git/ref/heads/main or git/ref/tags/v1.0
        if ref.startswith("refs/"):
            short = ref[len("refs/") :]
        else:
            short = ref
        try:
            return self.gh.api("GET", f"repos/{self.repo}/git/ref/{short}")
        except GhError as exc:
            if "404" in (exc.stderr or ""):
                return None
            raise

    def create_ref(self, ref, sha):
        return self.gh.api("POST", f"repos/{self.repo}/git/refs", body={"ref": ref, "sha": sha})

    def update_ref(self, ref, sha, expected_sha, force=False):
        live = self.get_ref(ref)
        if live is None:
            self.fail("ref missing before update", ref=ref)
        live_sha = live["object"]["sha"]
        if live_sha != expected_sha:
            self.fail("concurrent ref change", ref=ref, live=live_sha, expected=expected_sha)
        short = ref[len("refs/") :] if ref.startswith("refs/") else ref
        return self.gh.api(
            "PATCH",
            f"repos/{self.repo}/git/refs/{short}",
            body={"sha": sha, "force": bool(force)},
        )

    def ensure_annotated_object(self, chain_item):
        sha = chain_item["sha"]
        try:
            self.gh.api("GET", f"repos/{self.repo}/git/tags/{sha}")
            return sha
        except GhError as exc:
            if "404" not in (exc.stderr or ""):
                raise
        obj = chain_item.get("object") or {}
        body = {
            "tag": chain_item["tag"],
            "message": chain_item.get("message") or "",
            "object": obj.get("sha"),
            "type": obj.get("type") or "commit",
        }
        if chain_item.get("tagger"):
            body["tagger"] = chain_item["tagger"]
        created = self.gh.api("POST", f"repos/{self.repo}/git/tags", body=body)
        if created["sha"] != sha:
            self.fail("recreated annotated tag SHA mismatch", live=created["sha"], expected=sha)
        return sha

    def create_tag_ref(self, name, object_sha):
        ref = f"refs/tags/{name}"
        live = self.get_ref(f"tags/{name}")
        if live is not None:
            live_sha = live["object"]["sha"]
            if live_sha == object_sha:
                self.journal("tag_exists_same", tag=name, sha=object_sha)
                return "exists"
            self.fail("tag exists with different object", tag=name, live=live_sha, expected=object_sha)
        try:
            self.create_ref(ref, object_sha)
        except GhError as exc:
            live = self.get_ref(f"tags/{name}")
            if live and live["object"]["sha"] == object_sha:
                self.journal("tag_create_race_same", tag=name)
                return "exists"
            self.fail("tag create failed", tag=name, stderr=exc.stderr)
        return "created"

    def delete_ref(self, ref):
        short = ref[len("refs/") :] if ref.startswith("refs/") else ref
        try:
            self.gh.api("DELETE", f"repos/{self.repo}/git/refs/{short}")
        except GhError as exc:
            if "404" in (exc.stderr or ""):
                self.journal("ref_already_absent", ref=ref)
                return
            raise

    def get_release(self, tag):
        try:
            return self.gh.api("GET", f"repos/{self.repo}/releases/tags/{tag}")
        except GhError as exc:
            if "404" in (exc.stderr or ""):
                return None
            raise

    def create_release(self, spec):
        body = {
            "tag_name": spec["tag_name"],
            "name": spec.get("name") or spec["tag_name"],
            "body": spec.get("body") or "",
            "draft": bool(spec.get("draft")),
            "prerelease": bool(spec.get("prerelease")),
            "target_commitish": spec.get("target_commitish"),
            "generate_release_notes": False,
        }
        if spec.get("make_latest"):
            body["make_latest"] = "true"
        return self.gh.api("POST", f"repos/{self.repo}/releases", body=body)

    def patch_release(self, release_id, spec):
        body = {
            "name": spec.get("name"),
            "body": spec.get("body") or "",
            "draft": bool(spec.get("draft")),
            "prerelease": bool(spec.get("prerelease")),
        }
        if spec.get("target_commitish"):
            body["target_commitish"] = spec["target_commitish"]
        if spec.get("make_latest"):
            body["make_latest"] = "true"
        return self.gh.api("PATCH", f"repos/{self.repo}/releases/{release_id}", body=body)

    def delete_release(self, release_id):
        try:
            self.gh.api("DELETE", f"repos/{self.repo}/releases/{release_id}")
        except GhError as exc:
            if "404" in (exc.stderr or ""):
                self.journal("release_already_absent", id=release_id)
                return
            raise

    def verify_backup_gate(self, old_tree_sha, archive_tree_sha, tag_snapshot):
        old_map = self.get_tree_map(old_tree_sha)
        new_map = self.get_tree_map(archive_tree_sha)
        if old_map["truncated"] or new_map["truncated"]:
            self.fail("tree listing truncated during backup gate")
        extras = []
        for path, meta in new_map["blobs"].items():
            if path.startswith(RECOVERY_ROOT + "/"):
                extras.append(path)
                continue
            if path not in old_map["blobs"]:
                self.fail("backup tree added non-recovery path", path=path)
            if meta["sha"] != old_map["blobs"][path]["sha"]:
                self.fail("backup mutated old blob", path=path, live=meta["sha"], expected=old_map["blobs"][path]["sha"])
        for path, meta in old_map["blobs"].items():
            if path not in new_map["blobs"]:
                self.fail("backup dropped old path", path=path)
            elif not path.startswith(RECOVERY_ROOT + "/") and new_map["blobs"][path]["sha"] != meta["sha"]:
                self.fail("backup mutated old blob", path=path)
        if not extras:
            self.fail("backup missing recovery files")
        for item in tag_snapshot:
            live = self.get_ref(f"tags/{item['name']}")
            if live is None:
                self.fail("tag missing during backup gate", tag=item["name"])
            if live["object"]["sha"] != item["ref_object_sha"]:
                self.fail(
                    "tag object changed during backup gate",
                    tag=item["name"],
                    live=live["object"]["sha"],
                    expected=item["ref_object_sha"],
                )
        return {"recovery_paths": sorted(extras), "old_blob_count": len(old_map["blobs"])}

    def backup(self):
        prepare = self.load_prepare()
        self.journal("stage_start", stage="backup", execute=self.execute)
        zip_meta = self.download_zipball(
            prepare["main"]["sha"],
            self.work_dir / "remote-head.zip",
        )
        write_json(self.work_dir / "remote-head-zip.json", zip_meta)
        local_meta = None
        if self.local_snapshot:
            local_meta = self.snapshot_local_docs(self.work_dir / "local-snapshot", self.local_snapshot)
        files, manifest = self.recovery_files(prepare, zip_meta)
        parents = list(prepare["reachability_parents"])
        plan = {
            "archive_branch": ARCHIVE_BRANCH,
            "base_tree": prepare["main"]["tree"],
            "parents": parents,
            "recovery_files": sorted(files),
            "zip": zip_meta,
            "local_snapshot": local_meta,
            "execute": self.execute,
        }
        write_json(self.work_dir / "backup-plan.json", plan)
        if not self.execute:
            write_json(self.work_dir / "backup-state.json", {"stage": "backup", "mode": "dry-run", "plan": plan, "manifest": manifest})
            self.journal("stage_done", stage="backup", mode="dry-run")
            return plan
        live_archive = self.get_ref("heads/" + ARCHIVE_BRANCH)
        if live_archive:
            commit = self.get_commit(live_archive["object"]["sha"])
            try:
                self.verify_backup_gate(prepare["main"]["tree"], commit["tree"]["sha"], prepare["tags"])
                self.journal("backup_already_present", sha=live_archive["object"]["sha"])
                write_json(
                    self.work_dir / "backup-state.json",
                    {
                        "stage": "backup",
                        "mode": "execute",
                        "archive_commit": live_archive["object"]["sha"],
                        "reused": True,
                        "manifest": manifest,
                    },
                )
                return plan
            except Fail:
                self.fail(
                    "archive branch exists but fails backup gate; refusing to force",
                    sha=live_archive["object"]["sha"],
                )
        tree_sha, expected_blobs = self.create_tree(prepare["main"]["tree"], files)
        commit_sha = self.create_commit(
            "archive: pre-milestones-20260910 backup with recovery pack",
            tree_sha,
            parents,
        )
        self.create_ref(ARCHIVE_REF, commit_sha)
        live = self.get_ref("heads/" + ARCHIVE_BRANCH)
        if live is None or live["object"]["sha"] != commit_sha:
            self.fail("archive branch not at created commit")
        gate = self.verify_backup_gate(prepare["main"]["tree"], tree_sha, prepare["tags"])
        state = {
            "stage": "backup",
            "mode": "execute",
            "archive_commit": commit_sha,
            "archive_tree": tree_sha,
            "recovery_blobs": expected_blobs,
            "gate": gate,
            "manifest": manifest,
            "zip": zip_meta,
        }
        write_json(self.work_dir / "backup-state.json", state)
        self.journal("stage_done", stage="backup", mode="execute", commit=commit_sha)
        return state

    def release_spec_for_milestone(self, name, commit_sha, make_latest=False):
        plan = read_json(self.script_dir / "release-plan.json")
        item = None
        for m in plan.get("milestones") or []:
            if m["public"] == name:
                item = m
                break
        en = (item or {}).get("en") or {}
        zh = (item or {}).get("zh") or {}
        body = (
            f"{name} recataloged {plan.get('catalog_date')} (catalog date, not original publish time).\n\n"
            f"Measured: {en.get('measured')}\n\n"
            f"Result: {en.get('result')}\n\n"
            f"Conclusion: {en.get('conclusion')}\n\n"
            f"Limit: {en.get('limit')}\n\n"
            f"测了：{zh.get('measured')}\n\n"
            f"结果：{zh.get('result')}\n\n"
            f"结论：{zh.get('conclusion')}\n\n"
            f"限制：{zh.get('limit')}\n"
        )
        return {
            "tag_name": name,
            "name": f"{name} — recataloged experiment milestone",
            "body": body,
            "draft": False,
            "prerelease": False,
            "target_commitish": commit_sha,
            "make_latest": make_latest,
        }

    def collect_docs(self):
        if self.docs_dir is None or not self.docs_dir.is_dir():
            self.fail("--docs-dir is required for publish")
        files = {}
        for name in ALLOWED_DOC_FILES:
            src = self.docs_dir / name
            if not src.is_file():
                self.fail("required docs file missing", path=name)
            files[name] = src.read_bytes()
        return files

    def guard_main(self, expected_sha):
        live = self.get_ref("heads/main")
        if live is None:
            self.fail("main missing")
        sha = live["object"]["sha"]
        if sha != expected_sha:
            self.fail("main changed (concurrent or unexpected)", live=sha, expected=expected_sha)
        return sha

    def guard_tags(self, expected: dict):
        for name, sha in expected.items():
            live = self.get_ref(f"tags/{name}")
            if live is None:
                self.fail("expected tag missing", tag=name)
            if live["object"]["sha"] != sha:
                self.fail("tag changed", tag=name, live=live["object"]["sha"], expected=sha)

    def _assert_old_tags_intact(self, prepare):
        for t in prepare["tags"]:
            live = self.get_ref(f"tags/{t['name']}")
            if live is None or live["object"]["sha"] != t["ref_object_sha"]:
                self.fail("pre-publish tag drift", tag=t["name"])

    def publish(self):
        prepare = self.load_prepare()
        self.journal("stage_start", stage="publish", execute=self.execute)
        backup_state_path = self.work_dir / "backup-state.json"
        if not backup_state_path.is_file():
            self.fail("backup-state.json missing; run backup first")
        backup_state = read_json(backup_state_path)
        if self.execute and backup_state.get("mode") != "execute":
            self.fail("publish --execute requires backup --execute to have created the archive branch")
        docs = self.collect_docs()
        new_tree_plan = {
            "base_tree": prepare["main"]["tree"],
            "files": sorted(docs),
            "parent": prepare["main"]["sha"],
        }
        write_json(self.work_dir / "publish-plan.json", new_tree_plan)
        if not self.execute:
            live_main = self.get_ref("heads/main")
            if live_main is None or live_main["object"]["sha"] != prepare["main"]["sha"]:
                self.fail("dry-run: main is not the expected pre-milestone SHA")
            self._assert_old_tags_intact(prepare)
            write_json(
                self.work_dir / "publish-state.json",
                {"stage": "publish", "mode": "dry-run", "plan": new_tree_plan},
            )
            self.journal("stage_done", stage="publish", mode="dry-run")
            return new_tree_plan
        archive = self.get_ref("heads/" + ARCHIVE_BRANCH)
        if archive is None:
            self.fail("archive branch missing; cannot publish")
        arch_commit = self.get_commit(archive["object"]["sha"])
        self.verify_backup_gate(prepare["main"]["tree"], arch_commit["tree"]["sha"], prepare["tags"])
        progress_path = self.work_dir / "publish-progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else {"phase": "docs"}
        live_main = self.get_ref("heads/main")
        if live_main is None:
            self.fail("main missing")
        live_sha = live_main["object"]["sha"]
        if progress.get("new_main"):
            if live_sha != progress["new_main"]:
                self.fail("main changed after docs commit", live=live_sha, expected=progress["new_main"])
        else:
            if live_sha != prepare["main"]["sha"]:
                self.fail("main changed (concurrent or unexpected)", live=live_sha, expected=prepare["main"]["sha"])
            self._assert_old_tags_intact(prepare)
            new_tree, blob_map = self.create_tree(prepare["main"]["tree"], docs)
            new_commit = self.create_commit(
                "docs: recatalog seven public experiment milestones (2026-09-10 catalog date)",
                new_tree,
                [prepare["main"]["sha"]],
            )
            self.update_ref("refs/heads/main", new_commit, prepare["main"]["sha"], force=False)
            progress["new_main"] = new_commit
            progress["new_tree"] = new_tree
            progress["doc_blobs"] = blob_map
            progress["phase"] = "tags"
            write_json(progress_path, progress)
        if not progress.get("deleted_v2"):
            self._assert_old_tags_intact(prepare)
        targets = dict(FROZEN_MILESTONE_COMMITS)
        targets["v1.6"] = progress["new_main"]
        created = progress.get("tags") or {}
        for name in PUBLIC_MILESTONES:
            target = targets[name]
            if name == "v1.0":
                live = self.get_ref("tags/v1.0")
                if live is None or live["object"]["sha"] != target:
                    self.fail("v1.0 ref must keep original SHA", live=None if live is None else live["object"]["sha"])
                created[name] = {"action": "kept", "sha": target}
                continue
            action = self.create_tag_ref(name, target)
            created[name] = {"action": action, "sha": target}
            progress["tags"] = created
            write_json(progress_path, progress)
        releases_done = progress.get("releases") or {}
        for name in PUBLIC_MILESTONES:
            if releases_done.get(name):
                continue
            spec = self.release_spec_for_milestone(name, targets[name], make_latest=(name == "v1.6"))
            existing = self.get_release(name)
            if existing:
                if name == "v1.0":
                    patched = self.patch_release(existing["id"], spec)
                    releases_done[name] = {"action": "patched", "id": patched.get("id") or existing["id"]}
                else:
                    releases_done[name] = {"action": "exists", "id": existing["id"]}
            else:
                created_rel = self.create_release(spec)
                releases_done[name] = {"action": "created", "id": created_rel["id"]}
            progress["releases"] = releases_done
            write_json(progress_path, progress)
        for name in PUBLIC_MILESTONES:
            if self.get_ref(f"tags/{name}") is None:
                self.fail("cannot delete old v2: public tag missing", tag=name)
            if self.get_release(name) is None:
                self.fail("cannot delete old v2: public release missing", tag=name)
        old_v2 = [t for t in prepare["tags"] if t["name"].startswith("v2.")]
        deleted = list(progress.get("deleted_v2") or [])
        deleted_set = set(deleted)
        for t in old_v2:
            if t["name"] in deleted_set:
                continue
            rel = self.get_release(t["name"])
            if rel:
                self.delete_release(rel["id"])
            self.delete_ref(f"refs/tags/{t['name']}")
            deleted.append(t["name"])
            deleted_set.add(t["name"])
            progress["deleted_v2"] = deleted
            write_json(progress_path, progress)
        v16 = self.get_release("v1.6")
        if v16:
            self.patch_release(
                v16["id"],
                {**self.release_spec_for_milestone("v1.6", targets["v1.6"], True), "make_latest": True},
            )
        post = {
            "stage": "postpublish",
            "ts": utc_now(),
            "old_main": prepare["main"]["sha"],
            "old_tree": prepare["main"]["tree"],
            "new_main": progress["new_main"],
            "new_tree": progress["new_tree"],
            "public_tags": {k: targets[k] for k in PUBLIC_MILESTONES},
            "created_tags": CREATED_TAGS,
            "deleted_v2_tags": [t["name"] for t in old_v2],
            "v1_0_release_id_kept": True,
            "irreversible": IRREVERSIBLE_NOTE,
        }
        write_json(self.work_dir / "postpublish-state.json", post)
        self.push_state_to_archive("postpublish-state.json", post)
        write_json(self.work_dir / "publish-state.json", {"stage": "publish", "mode": "execute", **post})
        self.journal("stage_done", stage="publish", mode="execute", main=progress["new_main"])
        return post

    def push_state_to_archive(self, name, obj):
        archive = self.get_ref("heads/" + ARCHIVE_BRANCH)
        if archive is None:
            self.fail("archive branch missing while writing remote state")
        expected = archive["object"]["sha"]
        commit = self.get_commit(expected)
        payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        path = f"{RECOVERY_ROOT}/{name}"
        tree_sha, _ = self.create_tree(commit["tree"]["sha"], {path: payload})
        new_commit = self.create_commit(f"archive: update {name}", tree_sha, [expected])
        self.update_ref(ARCHIVE_REF, new_commit, expected, force=False)
        self.journal("archive_state_updated", name=name, commit=new_commit)

    def tree_diff_only_allowed(self, old_tree, new_tree):
        old_map = self.get_tree_map(old_tree)["blobs"]
        new_map = self.get_tree_map(new_tree)["blobs"]
        changed = []
        for path, meta in new_map.items():
            if path not in old_map:
                changed.append(path)
            elif old_map[path]["sha"] != meta["sha"]:
                changed.append(path)
        for path in old_map:
            if path not in new_map:
                changed.append(path)
        unexpected = [p for p in changed if path_not_allowed(p)]
        if unexpected:
            self.fail("main tree changed outside allowed docs", paths=unexpected)
        return sorted(set(changed))

    def verify(self):
        self.require_no_execute_for_readonly("verify")
        self.journal("stage_start", stage="verify")
        prepare = self.load_prepare()
        live_main = self.get_ref("heads/main")
        if live_main is None:
            self.fail("main missing")
        commit = self.get_commit(live_main["object"]["sha"])
        parents = [p["sha"] for p in commit.get("parents") or []]
        if prepare["main"]["sha"] not in parents:
            self.fail("main is not a child of the backed-up main", parents=parents)
        changed = self.tree_diff_only_allowed(prepare["main"]["tree"], commit["tree"]["sha"])
        tags = {}
        for name in PUBLIC_MILESTONES:
            live = self.get_ref(f"tags/{name}")
            if live is None:
                self.fail("public tag missing", tag=name)
            tags[name] = live["object"]["sha"]
        for name, sha in FROZEN_MILESTONE_COMMITS.items():
            if tags[name] != sha:
                self.fail("public tag target mismatch", tag=name, live=tags[name], expected=sha)
        if tags["v1.6"] != live_main["object"]["sha"]:
            self.fail("v1.6 must point at current main", live=tags["v1.6"], main=live_main["object"]["sha"])
        rels = {r["tag_name"]: r for r in self.snapshot_releases()}
        for name in PUBLIC_MILESTONES:
            if name not in rels:
                self.fail("public release missing", tag=name)
        leftover = [n for n in rels if n.startswith("v2.")]
        leftover_tags = []
        tag_refs = self.gh.api("GET", f"repos/{self.repo}/git/matching-refs/tags", paginate=True) or []
        for entry in tag_refs:
            n = tag_name_from_ref(entry["ref"])
            if n.startswith("v2."):
                leftover_tags.append(n)
        if leftover or leftover_tags:
            self.fail("old v2 refs still present", releases=leftover, tags=leftover_tags)
        latest = None
        for r in rels.values():
            if r.get("prerelease") or r.get("draft"):
                continue
            latest = r
            break
        # GitHub list is newest first; after make_latest, v1.6 should be first non-draft
        if "v1.6" not in rels:
            self.fail("v1.6 release missing")
        plan = self.restore_plan(persist=False)
        report = {
            "main": live_main["object"]["sha"],
            "changed_paths": changed,
            "tags": tags,
            "release_count": len(rels),
            "v1_6_latest_candidate": rels["v1.6"]["id"],
            "restore_plan_old_ref_count": len(plan["restore_refs"]),
            "restore_plan_delete_new": plan["delete_new_refs"],
        }
        write_json(self.work_dir / "verify-state.json", report)
        self.journal("stage_done", stage="verify")
        return report

    def restore_plan(self, persist=True):
        self.require_no_execute_for_readonly("restore-plan")
        self.journal("stage_start", stage="restore-plan")
        prepare = self.load_prepare()
        post_path = self.work_dir / "postpublish-state.json"
        post = read_json(post_path) if post_path.is_file() else None
        restore_refs = []
        for t in prepare["tags"]:
            restore_refs.append(
                {
                    "ref": t["ref"],
                    "object_type": t["ref_object_type"],
                    "object_sha": t["ref_object_sha"],
                    "resolved_commit": t["resolved_commit"],
                    "annotated_chain": t["annotated_chain"],
                }
            )
        restore_releases = []
        for r in prepare["releases"]:
            restore_releases.append(
                {
                    "tag_name": r["tag_name"],
                    "name": r["name"],
                    "body": r["body"],
                    "draft": r["draft"],
                    "prerelease": r["prerelease"],
                    "target_commitish": r["target_commitish"],
                    "original_id": r["id"],
                    "original_published_at": r["published_at"],
                    "keep_id_if_present": r["tag_name"] == "v1.0",
                }
            )
        plan = {
            "old_main": prepare["main"]["sha"],
            "old_tree": prepare["main"]["tree"],
            "restore_refs": restore_refs,
            "restore_releases": restore_releases,
            "delete_new_refs": [f"refs/tags/{n}" for n in CREATED_TAGS],
            "delete_new_releases": list(CREATED_TAGS),
            "main_rollback": {
                "method": "new commit with old tree, parent=current main, fast-forward only, no force",
                "tree": prepare["main"]["tree"],
                "reject_if_main_not_expected_postpublish": True,
                "reject_if_any_public_tag_changed": True,
            },
            "irreversible": IRREVERSIBLE_NOTE,
            "postpublish": post,
        }
        if persist:
            write_json(self.work_dir / "restore-plan.json", plan)
        self.journal("stage_done", stage="restore-plan", refs=len(restore_refs))
        return plan

    def restore(self):
        plan = self.restore_plan(persist=True)
        self.journal("stage_start", stage="restore", execute=self.execute)
        post = plan.get("postpublish")
        if not self.execute:
            write_json(self.work_dir / "restore-state.json", {"stage": "restore", "mode": "dry-run", "plan": plan, "guard_ready": post is not None})
            self.journal("stage_done", stage="restore", mode="dry-run", guard_ready=post is not None)
            return plan
        if post is None:
            self.fail("postpublish-state.json missing; needed to guard the 7-tag state")
        expected_main = post["new_main"]
        expected_tags = post["public_tags"]
        write_json(
            self.work_dir / "restore-guard.json",
            {"main": expected_main, "tags": expected_tags, "ts": utc_now()},
        )
        self.guard_main(expected_main)
        self.guard_tags(expected_tags)
        progress_path = self.work_dir / "restore-progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else {"restored_refs": {}, "restored_releases": {}}
        if isinstance(progress.get("restored_refs"), list):
            progress["restored_refs"] = {x.get("name"): x for x in progress["restored_refs"] if x.get("name")}
        if isinstance(progress.get("restored_releases"), list):
            progress["restored_releases"] = {x.get("tag"): x for x in progress["restored_releases"] if x.get("tag")}
        for item in plan["restore_refs"]:
            name = tag_name_from_ref(item["ref"])
            live = self.get_ref(f"tags/{name}")
            if live is not None and live["object"]["sha"] == item["object_sha"]:
                progress["restored_refs"][name] = {"name": name, "action": "already"}
                write_json(progress_path, progress)
                continue
            if live is not None and live["object"]["sha"] != item["object_sha"]:
                self.delete_ref(item["ref"])
            if item.get("object_type") == "tag" and item.get("annotated_chain"):
                self.ensure_annotated_object(item["annotated_chain"][0])
            action = self.create_tag_ref(name, item["object_sha"])
            progress["restored_refs"][name] = {"name": name, "action": action}
            write_json(progress_path, progress)
        for spec in plan["restore_releases"]:
            existing = self.get_release(spec["tag_name"])
            payload = {
                "tag_name": spec["tag_name"],
                "name": spec["name"],
                "body": spec["body"],
                "draft": spec["draft"],
                "prerelease": spec["prerelease"],
                "target_commitish": spec["target_commitish"],
            }
            if existing and spec.get("keep_id_if_present") and existing["id"] == spec["original_id"]:
                self.patch_release(existing["id"], payload)
                progress["restored_releases"][spec["tag_name"]] = {
                    "tag": spec["tag_name"],
                    "action": "patched_original_id",
                    "id": existing["id"],
                }
            elif existing:
                self.patch_release(existing["id"], payload)
                progress["restored_releases"][spec["tag_name"]] = {
                    "tag": spec["tag_name"],
                    "action": "patched_new_id",
                    "id": existing["id"],
                }
            else:
                created = self.create_release(payload)
                progress["restored_releases"][spec["tag_name"]] = {
                    "tag": spec["tag_name"],
                    "action": "created",
                    "id": created["id"],
                }
            write_json(progress_path, progress)
        # confirm 21 old tags restored before deleting 6 new
        for item in plan["restore_refs"]:
            live = self.get_ref(f"tags/{tag_name_from_ref(item['ref'])}")
            if live is None or live["object"]["sha"] != item["object_sha"]:
                self.fail("old tag not fully restored; refusing to delete new refs", tag=item["ref"])
        for spec in plan["restore_releases"]:
            if self.get_release(spec["tag_name"]) is None:
                self.fail("old release not restored; refusing to delete new releases", tag=spec["tag_name"])
        for name in CREATED_TAGS:
            rel = self.get_release(name)
            if rel:
                self.delete_release(rel["id"])
            live = self.get_ref(f"tags/{name}")
            if live:
                self.delete_ref(f"refs/tags/{name}")
        self.guard_main(expected_main)
        commit = self.get_commit(expected_main)
        if commit["tree"]["sha"] == plan["old_tree"]:
            # already old tree (unexpected); still do not force
            new_main = expected_main
        else:
            new_commit = self.create_commit(
                "revert: restore pre-milestones-20260910 main tree",
                plan["old_tree"],
                [expected_main],
            )
            self.update_ref("refs/heads/main", new_commit, expected_main, force=False)
            new_main = new_commit
        result = {
            "stage": "restore",
            "mode": "execute",
            "new_main": new_main,
            "old_tree": plan["old_tree"],
            "progress": progress,
            "irreversible": IRREVERSIBLE_NOTE,
        }
        write_json(self.work_dir / "restore-state.json", result)
        try:
            self.push_state_to_archive("restore-state.json", result)
        except Fail as exc:
            self.journal("archive_state_push_failed", message=str(exc))
            raise
        self.journal("stage_done", stage="restore", mode="execute", main=new_main)
        return result


def path_not_allowed(path):
    return path not in ALLOWED_DOC_FILES


def build_parser():
    p = argparse.ArgumentParser(description="Milestone backup/publish/restore via gh (no git.exe).")
    p.add_argument("stage", choices=["prepare", "backup", "publish", "verify", "restore-plan", "restore"])
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--docs-dir", default=None)
    p.add_argument("--local-snapshot", default=None, help="User working copy; stored separately from remote zip.")
    p.add_argument("--execute", action="store_true", help="Allow remote writes. Default is dry-run.")
    p.add_argument("--dry-run", action="store_true", help="Force dry-run even if --execute is also passed.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    execute = bool(args.execute) and not args.dry_run
    m = Migration(
        repo=args.repo,
        work_dir=args.work_dir,
        docs_dir=args.docs_dir,
        local_snapshot=args.local_snapshot,
        execute=execute,
    )
    try:
        if args.stage == "prepare":
            m.prepare()
        elif args.stage == "backup":
            m.backup()
        elif args.stage == "publish":
            m.publish()
        elif args.stage == "verify":
            m.verify()
        elif args.stage == "restore-plan":
            m.restore_plan()
        elif args.stage == "restore":
            m.restore()
    except Block as exc:
        print(f"BLOCK {exc}", file=sys.stderr)
        return 2
    except (Fail, GhError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print("OVERALL=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
