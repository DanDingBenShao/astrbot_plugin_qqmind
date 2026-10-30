"""每日 token 预算账本（老板定案：烧完降级轻量，群里明说，最多 2 次/群/天）。

- 按自然日滚动，落盘 JSON，重启不丢当天用量
- limit=0 表示不限制（默认，避免老用户被突然掐断）
- 额度提示按群计数，防止刷屏
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class DailyBudget:
    def __init__(self, path=None, limit: int = 0):
        self.path = Path(path) if path else None
        self.limit = max(0, int(limit))
        self.day = time.strftime("%Y-%m-%d")
        self.used = 0
        self.notices: dict[int, int] = {}
        # 分群子账（v0.2.7）：gid -> 该群当日 token；0/None key = 非群聊（私聊/后台）
        self.per_group: dict[int, int] = {}
        self._load()

    # ---------- 落盘 ----------
    def _load(self) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.day = str(d.get("day") or self.day)
            self.used = max(0, int(d.get("used") or 0))
            self.notices = {int(k): int(v) for k, v in (d.get("notices") or {}).items()}
            self.per_group = {int(k): int(v)
                              for k, v in (d.get("per_group") or {}).items()}
        except Exception:
            self.day, self.used, self.notices = time.strftime("%Y-%m-%d"), 0, {}
            self.per_group = {}
        self._roll()

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"day": self.day, "used": self.used,
                           "notices": self.notices,
                           "per_group": self.per_group}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            # 路径被目录占位 / 磁盘只读等：账本写失败不能拖垮消息处理，
            # 内存态账目照常，恢复后自动落盘
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] budget save failed: {e!r}")

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if self.day != today:
            self.day = today
            self.used = 0
            self.notices = {}
            self.per_group = {}
            self._save()

    # ---------- 账本 ----------
    def add(self, tokens: int, gid: int | None = None) -> int:
        """记一笔。gid=None 进总账不分子账；gid=群号同时进该群子账；
        gid=0 表示非群聊消耗（私聊/后台任务），单独归档便于对账。"""
        self._roll()
        n = max(0, int(tokens))
        if n:
            self.used += n
            if gid is not None:
                self.per_group[int(gid)] = self.per_group.get(int(gid), 0) + n
            self._save()
        return n

    def group_used(self, gid: int) -> int:
        """该群当日子账。"""
        self._roll()
        return self.per_group.get(int(gid), 0)

    def exhausted(self) -> bool:
        self._roll()
        return self.limit > 0 and self.used >= self.limit

    def remaining(self) -> int | None:
        self._roll()
        return None if self.limit <= 0 else max(0, self.limit - self.used)

    def allow_notice(self, gid: int, cap: int = 2) -> bool:
        """每个群每天最多提示 cap 次。"""
        self._roll()
        gid = int(gid)
        cap = max(1, int(cap))
        if self.notices.get(gid, 0) >= cap:
            return False
        self.notices[gid] = self.notices.get(gid, 0) + 1
        self._save()
        return True

    def snapshot(self) -> dict:
        self._roll()
        return {"day": self.day, "used": self.used, "limit": self.limit,
                "remaining": self.remaining(),
                "per_group": dict(self.per_group)}
