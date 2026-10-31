"""群聊候选池（v0.3.0 公共池）：所有群的原文进同一个池，攒满一批才压缩。

老板 2026-10-31 定案（公共记忆）：
- 群聊记忆是**公共的**：不做群间隔离（隔离只用于私聊）。所有群共用一个池
  和一个日记库；每条消息带 gid 字段，谁在哪个群说的照样记得。
- 触发：攒够 pool_trigger_msgs 条（默认 1000）压一次；不足一批的残余
  每天凌晨由插件侧定时兜底压一次（「一天一处理残余」）。
- 每批一次模型调用，输出 JSON 数组（多篇话题日记，各带证据消息 ID）；
  上一批结尾 5 条作为「前情提要」附给模型，跨批话题不断头。
- 失败指数退避：5min → 10 → 20 → 40 → 封顶 60min，成功即清零。
- 池有硬上限（pool_max_msgs），超过丢最老（极端刷屏时保内存）。
- 旧版按群池子（pool_<gid>.json）首次启动自动合并迁移，原文件改名 .migrated。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

BATCH_COMPRESS_PROMPT = (
    "你是群聊记忆整理助手。下面是一段群聊原文（每行前缀 [id] 是消息编号，"
    "群号+说话人已标注）。把这段聊天按**话题**拆分成若干篇日记：同一个话题下"
    "先后发生的对话归为一篇，话题切换就新起一篇。\n"
    '只输出一个 JSON 数组（不要代码块、不要解释），每篇形如：\n'
    '{"title": "一句话标题，10字以内", '
    '"tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、结论/约定/情绪、涉及谁", '
    '"content": "最能代表这段对话的 1 句原话（保留说话人，没有就空串）", '
    '"evidence": [123, 125, 130]}\n'
    "要求：\n"
    "- evidence 必须是本篇话题真实涉及的消息 id（取 3~8 个，优先有信息量的）\n"
    "- 每篇 summary 只写本话题内的事，不跨话题；**最多 15 篇**，"
    "超出就合并相邻话题或跳过低信息段\n"
    "- 纯寒暄/表情刷屏/无信息段不要单独成篇（宁缺毋滥）\n"
    "- 不同群的消息一般属于不同场景，除非明显是同一话题延续，否则分开成篇\n"
    "- 标签 2~5 个中文词；时间顺序保持原样\n"
)

_LEGACY_POOL_RE = re.compile(r"^pool_(\d+)\.json$")


def _est(text: str) -> int:
    """池内 token 口径与 core.est_tokens 一致（CJK 1:1，其余 4:1）。"""
    n = 0
    for c in text or "":
        n += 1 if ord(c) > 0x2E80 else 0.25
    return int(n)


class CandidatePool:
    """公共候选池：所有群共用；消息带 gid。"""

    def __init__(self, store_dir,
                 *, trigger_msgs: int = 1000, max_msgs: int = 3000):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "pool_public.json"
        self.trigger_msgs = max(5, int(trigger_msgs))
        self.max_msgs = max(self.trigger_msgs + 100, int(max_msgs))
        self.msgs: list[dict] = []       # {"id","ts","gid","uid","nick","text"}
        self._next_id = 1
        self._pool_tokens = 0
        self._oldest_ts = 0.0
        self._fail_until = 0.0
        self._fail_streak = 0
        self._dropped_total = 0          # 满池丢弃计数（诊断/告警用）
        self._last_sweep_day = ""        # 最近一次凌晨残余兜底的日期
        self._prelude: list[dict] = []   # 上一批结尾 5 条（跨批话题不断头）
        self._taken: list[dict] = []     # take_all 的批次（confirm 时生成 prelude）
        self._migrate_legacy()
        self._load()

    # ---------- 旧数据迁移（按群池 → 公共池） ----------
    def _migrate_legacy(self) -> None:
        """首次启动：把 pool_<gid>.json / .pending.json 合并进公共池。"""
        if self.path.exists():
            return
        merged: list[dict] = []
        legacy_files: list[Path] = []
        for f in sorted(self.dir.glob("pool_*.json")):
            m = _LEGACY_POOL_RE.match(f.name)
            if not m:
                continue
            gid = int(m.group(1))
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                for msg in (d.get("msgs") or []):
                    msg["gid"] = gid
                    merged.append(msg)
            except Exception:
                pass
            legacy_files.append(f)
        for f in sorted(self.dir.glob("pool_*.pending.json")):
            m = re.match(r"^pool_(\d+)\.pending\.json$", f.name)
            if not m:
                continue
            gid = int(m.group(1))
            try:
                pending = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(pending, list):
                    for msg in pending:
                        msg["gid"] = gid
                        merged.append(msg)
            except Exception:
                pass
            legacy_files.append(f)
        if not merged:
            # 没有旧数据：也把遗留文件清走，避免下次误判
            for f in legacy_files:
                self._rename_migrated(f)
            return
        merged.sort(key=lambda x: float(x.get("ts") or 0))
        for i, msg in enumerate(merged, 1):
            msg["id"] = i
        self.msgs = merged
        self._next_id = len(merged) + 1
        self._pool_tokens = sum(_est(m.get("text", "")) + 12 for m in merged)
        self._oldest_ts = float(merged[0].get("ts") or 0)
        self.save()
        for f in legacy_files:
            self._rename_migrated(f)

    @staticmethod
    def _rename_migrated(f: Path) -> None:
        try:
            os.replace(f, f.with_name(f.name + ".migrated"))
        except OSError:
            pass

    # ---------- 持久化 ----------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.msgs = d.get("msgs") or []
            self._next_id = int(d.get("next_id") or
                                (self.msgs[-1]["id"] + 1 if self.msgs else 1))
            self._pool_tokens = int(d.get("tokens") or 0)
            self._oldest_ts = float(d.get("oldest_ts") or
                                    (self.msgs[0]["ts"] if self.msgs else 0))
            pre = d.get("prelude") or []
            self._prelude = [m for m in pre if isinstance(m, dict)][-5:]
            self._dropped_total = int(d.get("dropped_total") or 0)
            raw_sweep = d.get("last_sweep_day")
            # 旧版池文件没有补扫记账：按“今天已扫”处理，避免升级后立刻
            # 在白天压一次残余（第一次补扫从明天 04:30 开始）
            self._last_sweep_day = (time.strftime("%Y-%m-%d")
                                    if raw_sweep is None
                                    else str(raw_sweep or ""))
        except Exception:
            self.msgs, self._next_id = [], 1
            self._pool_tokens, self._oldest_ts = 0, 0.0
            self._dropped_total, self._last_sweep_day = 0, ""
        # 崩溃恢复：上次取走未交割的整批原样回池
        try:
            pp = self._pending_path()
            with open(pp, encoding="utf-8") as f:
                pending = json.load(f)
            if isinstance(pending, list) and pending:
                got = {m["id"] for m in self.msgs}
                back = [m for m in pending if m.get("id") not in got]
                if back:
                    self.msgs = back + self.msgs
                    self._pool_tokens = sum(_est(m["text"]) + 12
                                            for m in self.msgs)
                    self._oldest_ts = self.msgs[0]["ts"]
                    ids = [m["id"] for m in self.msgs if isinstance(m.get("id"), int)]
                    if ids:
                        self._next_id = max(self._next_id, max(ids) + 1)
            os.remove(pp)
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def save(self) -> None:
        try:
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"msgs": self.msgs, "next_id": self._next_id,
                           "tokens": self._pool_tokens,
                           "oldest_ts": self._oldest_ts,
                           "prelude": self._prelude,
                           "dropped_total": self._dropped_total,
                           "last_sweep_day": self._last_sweep_day},
                          f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] pool save failed: {e!r}")

    # ---------- 进出 ----------
    def add(self, ts: float, gid: int, uid: int, nick: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        mid = self._next_id
        self._next_id += 1
        self.msgs.append({"id": mid, "ts": float(ts or time.time()),
                          "gid": int(gid or 0), "uid": int(uid),
                          "nick": nick or "", "text": text})
        self._pool_tokens += _est(text) + 12   # 行开销（群号/nick/时间）
        if not self._oldest_ts:
            self._oldest_ts = self.msgs[0]["ts"]
        if len(self.msgs) > self.max_msgs:
            drop = len(self.msgs) - self.max_msgs
            dropped = self.msgs[:drop]
            for m in dropped:
                self._pool_tokens -= _est(m["text"]) + 12
            self.msgs = self.msgs[drop:]
            self._oldest_ts = self.msgs[0]["ts"] if self.msgs else 0.0
            # v0.3.1：丢记忆必须留痕（v0.3.0 是静默丢最老，故障时无从发现）。
            # 每次丢弃都打会刷屏，按“首次 + 每满 100 条”告警。
            before = self._dropped_total
            self._dropped_total += drop
            if before == 0 or before // 100 != self._dropped_total // 100:
                import logging
                gids: dict[int, int] = {}
                for m in dropped:
                    g = int(m.get("gid") or 0)
                    gids[g] = gids.get(g, 0) + 1
                logging.getLogger("astrbot").warning(
                    f"[qqmind] pool full ({self.max_msgs}): dropped "
                    f"{drop} oldest msgs (total {self._dropped_total}, "
                    f"gids={gids})")

    def should_flush(self, now: float | None = None) -> bool:
        """攒够一批才压（v0.3.0：按条数，不再按等待时长）。"""
        if not self.msgs:
            return False
        now = now or time.time()
        if now < self._fail_until:     # 失败退避期
            return False
        return len(self.msgs) >= self.trigger_msgs

    def mark_failed(self) -> None:
        """压缩失败：指数退避 5→10→20→40→60(min 封顶)。不落盘。"""
        self._fail_streak += 1
        cooldown = min(3600.0, 300.0 * (2 ** (self._fail_streak - 1)))
        self._fail_until = time.time() + cooldown

    def reset_failed(self) -> None:
        self._fail_streak = 0
        self._fail_until = 0.0

    def take_batch(self, limit: int | None = None) -> list[dict]:
        """取走最多 limit 条（None=全部），批次写 pending 防崩溃丢批。

        v0.3.1：单批上限由调用方传 trigger_msgs——停机/故障后池子可能涨到
        max_msgs，整批塞进一次请求会撑爆模型上下文；剩下的留在池里等下一批。
        """
        if limit is None:
            out = self.msgs
            self.msgs = []
        else:
            n = max(1, int(limit))
            out = self.msgs[:n]
            self.msgs = self.msgs[n:]
        self._pool_tokens = sum(_est(m["text"]) + 12 for m in self.msgs)
        self._oldest_ts = self.msgs[0]["ts"] if self.msgs else 0.0
        self._taken = out
        try:
            with open(self._pending_path(), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        except OSError as e:
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] pool pending write failed: {e!r}")
        return out

    def take_all(self) -> list[dict]:
        """取走全部（兼容旧调用/测试；生产 flush 用 take_batch 限流）。"""
        return self.take_batch(None)

    def confirm_taken(self) -> None:
        """压缩成功：删 pending；记下本批结尾 5 条做下批「前情提要」。"""
        try:
            os.remove(self._pending_path())
        except OSError:
            pass
        if self._taken:
            self._prelude = [dict(m) for m in self._taken[-5:]]
        self._taken = []
        self.reset_failed()
        self.save()      # v0.3.1：prelude 随交割落盘（崩溃后不断头）

    def _pending_path(self) -> Path:
        return self.dir / "pool_public.pending.json"

    def put_back(self, msgs: list[dict]) -> None:
        """压缩失败：原样放回池头（保持 id 与顺序），并清 pending。"""
        try:
            os.remove(self._pending_path())
        except OSError:
            pass
        self._taken = []
        if not msgs:
            return
        self.msgs = list(msgs) + self.msgs
        self._pool_tokens = sum(_est(m["text"]) + 12 for m in self.msgs)
        self._oldest_ts = self.msgs[0]["ts"]

    # ---------- 证据链回查 ----------
    def context_around(self, msg_id: int, window: int = 3) -> list[dict]:
        """原文库里某条消息 ±window 条（证据链寻回用）。"""
        for i, m in enumerate(self.msgs):
            if m["id"] == int(msg_id):
                lo = max(0, i - window)
                hi = min(len(self.msgs), i + window + 1)
                return self.msgs[lo:hi]
        return []

    def prelude(self) -> list[dict]:
        """上一批结尾 5 条（跨批话题的「前情提要」）。"""
        return list(self._prelude)

    # ---------- 每日残余兜底记账 ----------
    def sweep_done_today(self) -> bool:
        """今天（本地日期）是否已经做过残余兜底。"""
        return self._last_sweep_day == time.strftime("%Y-%m-%d")

    def mark_sweep_done(self) -> None:
        """残余兜底完成（或池本空）：记账并落盘，重启后不重复补扫。"""
        self._last_sweep_day = time.strftime("%Y-%m-%d")
        self.save()

    def tokens(self) -> int:
        return self._pool_tokens

    def __len__(self) -> int:
        return len(self.msgs)
