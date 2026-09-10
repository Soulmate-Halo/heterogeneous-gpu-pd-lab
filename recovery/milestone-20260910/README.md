# 回退包（异机可执行）

本目录在 `archive/pre-milestones-20260910` 分支的 `recovery/milestone-20260910` 下。不要依赖本机绝对路径。

## 认证

需要已登录的 GitHub CLI（`gh auth status` 成功），权限覆盖该仓库的 contents 与 releases。不要把 token 写进文件或贴到日志。

```
gh auth status
```

## 工作目录

在任意机器：

```
gh api repos/Soulmate-Halo/heterogeneous-gpu-pd-lab/contents/recovery/milestone-20260910/tools/migration.py?ref=archive/pre-milestones-20260910
```

更直接：检出备份分支后进入本目录。

```
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work prepare
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work backup --dry-run
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore-plan
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore --dry-run
python tools/migration.py --repo Soulmate-Halo/heterogeneous-gpu-pd-lab --work-dir ./work restore --execute
```

远端写必须显式 `--execute`。先 `restore-plan` 与 `restore --dry-run`，核对后再 `restore --execute`。不要 force-push，不要改写 commit 历史。

## 风险

- 并发有人改 main 或任何 tag：默认拒绝，不 force。
- 删除 Release 后，原 id / published_at / 下载计数不可恢复。
- zip 不上传为公开 Release 资产；远端基准是旧 tree 对象。
