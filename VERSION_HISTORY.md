# Version history / 版本对照

Public versions are the seven measured experiment milestones. Older publication numbers that only maintained wording, layout, or filing are not public versions and are not listed here.

公开版本就是七个实测实验里程碑。旧发布号里只改措辞、排版或归档的，不再作为公开版本，也不在本表列出。

| Public / 公开 | Old publication / 旧号 | Experiment IDs / 实验 |
| --- | --- | --- |
| v1.0 | v1.0 | 9B-PD-01 |
| v1.1 | v2.0–v2.4 | 9B-PIPE-01, 27B-LONG-01 |
| v1.2 | v2.5 | 27B-PD-01 |
| v1.3 | v2.6 | 27B-KV-01 (C/D), 27B-DRAFT-AUDIT-01 |
| v1.4 | v2.11 | ORNITH-PD-01 |
| v1.5 | v2.14 | FLASH-SPLIT-01 |
| v1.6 | v2.26 | ORNITH-PD-02 |

EXT-DGX-01 is an external background record filed with the v1.1 period; it is not a public experiment milestone.

EXT-DGX-01 是 v1.1 同期归档的外部背景，不是公开实验里程碑。

## Rollback guidance / 回退指引

Backup branch `archive/pre-milestones-20260910` keeps the complete pre-milestone tree and snapshots of the 21 old tags/releases (those Releases had no assets). Rollback appends a new commit: no force-push, no rewritten history.

备份分支 `archive/pre-milestones-20260910` 保留精简前完整代码树和 21 个旧 tag/release 的快照；这些 Release 没有 assets。回退通过追加提交恢复旧 tree，不强推、不改写 commit 历史。

Check out that branch, enter `recovery/milestone-20260910`, then run:

从该分支检出后进入 `recovery/milestone-20260910`，执行：

```
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore-plan
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore --dry-run
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore --execute
```

`restore-plan` is read-only. `restore --dry-run` writes no remote refs. Only `restore --execute` writes the remote. Deleted Release IDs, original `published_at`, and download counts cannot be restored; a recreated Release is a new identity. Old tag URLs work again after those tags are restored.

`restore-plan` 只读。`restore --dry-run` 不写远端。只有 `restore --execute` 才改远端。已删除的 Release ID、原发布时间、下载统计不可恢复。旧 tag 链接在精简后暂时失效，恢复对应 tag 后可再访问。
