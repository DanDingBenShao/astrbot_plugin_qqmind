"""群聊候选池（v0.2.14）：浅层窗口溢出的消息先进池，攒满才批量压缩。

设计（老板 2026-10-31 定案）：
- 池按群持久化（json），带自增 msg_id——压缩输出的日记记录证据消息 ID，
  日后命中日记可回查原文 ±3 条（证据链）。
- 触发：累计 token ≥ pool_trigger_tokens 或最老消息等待超过 pool_max_wait_min；
  外加兜底：跨天清池（昨天的残余凌晨压掉）。
- 每批一次模型调用，输出 JSON 数组（多篇话题日记，各带证据消息 ID）；
  切话题的活交给模型，不做本地规则切段（简单且边界更准）。
- 池有硬上限（pool_max_msgs），超过丢最老（极端刷屏时保内存）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

BATCH_COMPRESS_PROMPT = (
    "你是群聊记忆整理助手。下面是一段群聊原文（每行前缀 [id] 是消息编号，"
    "说话人已标注）。把这段聊天按**话题**拆分成若干篇日记：同一个话题下"
    "先后发生的对话归为一篇，话题切换就新起一篇。\n"
    '只输出一个 JSON 数组（不要代码块、不要解释），每篇形如：\n'
    '{"title": "一句话标题，10字以内", '
    '"tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、结论/约定/情绪、涉及谁", '
    '"content": "最能代表这段对话的原话 1~3 句（保留说话人，没有就空串）", '
    '"evidence": [123, 125, 130]}\n'
    "要求：\n"
    "- evidence 必须是本篇话题真实涉及的消息 id（取 3~8 个，优先有信息量的）\n"
    "- 每篇 summary 只写本话题内的事，不跨话题\n"
    "- 纯寒暄/表情刷屏/无信息段可以并进相邻话题或跳过（宁缺毋滥）\n"
    "- 标签 2~5 个中文词\n"
    "- 时间顺序保持原样\n"
)


class CandidatePool:
    """单群候选池：原文消息 + 自增 id + 水位记账。"""

    def __init__(self, store_dir, gid: int, *,
                 trigger_tokens: int = 6000,
                 max_wait_min: float = 180.0,
                 max_msgs: int = 3000):
        self.gid = int(gid)
        self.path = Path(store_dir) / f"pool_{int(gid)}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trigger_tokens = max(500, int(trigger_tokens))
        self.max_wait_min = max(5.0, float(max_wait_min))
        self.max_msgs = max(200, int(max_msgs))
        self.msgs: list[dict] = []       # {"id", "ts", "uid", "nick", "text"}
        self._next_id = 1
        self._pool_tokens = 0
        self._oldest_ts = 0.0
        self._fail_until = 0.0
        self._load()

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
        except Exception:
            self.msgs, self._next_id = [], 1
            self._pool_tokens, self._oldest_ts = 0, 0.0
        # 崩溃恢复（v0.2.14 审查修）：上次取走未交割的整批原样回池
        try:
            pp = self.path.parent / f"pool_{int(self.gid)}.pending.json"
            with open(pp, encoding="utf-8") as f:
                pending = json.load(f)
            if isinstance(pending, list) and pending:
                got = {m["id"] for m in self.msgs}
                back = [m for m in pending if m.get("id") not in got]
                if back:
                    self.msgs = back + self.msgs
                    self._pool_tokens = sum(self._est(m["text"]) + 12
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
                           "oldest_ts": self._oldest_ts},
                          f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] pool save failed for {self.gid}: {e!r}")

    # ---------- 进出 ----------
    @staticmethod
    def _est(text: str) -> int:
        """池内 token 口径与 core.est_tokens 一致（CJK 1:1，其余 4:1）。"""
        n = 0
        for c in text or "":
            n += 1 if ord(c) > 0x2E80 else 0.25
        return int(n)

    def add(self, ts: float, uid: int, nick: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        mid = self._next_id
        self._next_id += 1
        self.msgs.append({"id": mid, "ts": float(ts or time.time()),
                          "uid": int(uid), "nick": nick or "", "text": text})
        self._pool_tokens += self._est(text) + 12   # 行开销（nick/时间）
        if not self._oldest_ts:
            self._oldest_ts = self.msgs[0]["ts"]
        if len(self.msgs) > self.max_msgs:
            drop = len(self.msgs) - self.max_msgs
            for m in self.msgs[:drop]:
                self._pool_tokens -= self._est(m["text"]) + 12
            self.msgs = self.msgs[drop:]
            self._oldest_ts = self.msgs[0]["ts"] if self.msgs else 0.0

    def should_flush(self, now: float | None = None) -> bool:
        if not self.msgs:
            return False
        now = now or time.time()
        # 失败退避（v0.2.14 审查修）：压缩刚失败（402/断网）时冷却，
        # 别让每条新消息都触发一次注定失败的调用
        if now < self._fail_until:
            return False
        if self._pool_tokens >= self.trigger_tokens:
            return True
        if (self._oldest_ts
                and now - self._oldest_ts >= self.max_wait_min * 60):
            return True
        return False

    def mark_failed(self, cooldown_s: float = 300.0) -> None:
        """压缩失败后的冷却（不落盘——重启即重试，失败态无需持久化）。"""
        self._fail_until = time.time() + max(30.0, float(cooldown_s))

    def take_all(self) -> list[dict]:
        """取走全部（压缩前调用）。

        v0.2.14 审查修：取走的同时把 batch 写进 pending 文件——压缩调用
        期间进程崩溃的话，启动时 _load 会把 pending 原样回池，不丢消息。
        成功入库后调用方必须 confirm_taken() 删 pending。
        """
        out = self.msgs
        self.msgs = []
        self._pool_tokens = 0
        self._oldest_ts = 0.0
        try:
            with open(self._pending_path(), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        except OSError as e:
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] pool pending write failed for {self.gid}: {e!r}")
        return out

    def confirm_taken(self) -> None:
        """压缩成功：删 pending（整批消息正式交割完毕）。"""
        try:
            os.remove(self._pending_path())
        except OSError:
            pass

    def _pending_path(self) -> Path:
        return self.path.parent / f"pool_{self.gid}.pending.json"

    def put_back(self, msgs: list[dict]) -> None:
        """压缩失败：原样放回池头（保持 id 与顺序），并清 pending。"""
        try:
            os.remove(self._pending_path())
        except OSError:
            pass
        if not msgs:
            return
        self.msgs = list(msgs) + self.msgs
        self._pool_tokens = sum(self._est(m["text"]) + 12 for m in self.msgs)
        self._oldest_ts = self.msgs[0]["ts"]

    # ---------- 证据链回查 ----------
    def context_around(self, msg_id: int, window: int = 3) -> list[dict]:
        """原文库里某条消息 ±window 条（证据链寻回用）。

        只在消息仍在池内时有效；已压缩的原文由原文档案（recent_archive）
        兜底——调用方合并两处结果。
        """
        for i, m in enumerate(self.msgs):
            if m["id"] == int(msg_id):
                lo = max(0, i - window)
                hi = min(len(self.msgs), i + window + 1)
                return self.msgs[lo:hi]
        return []

    def tokens(self) -> int:
        return self._pool_tokens

    def __len__(self) -> int:
        return len(self.msgs)
