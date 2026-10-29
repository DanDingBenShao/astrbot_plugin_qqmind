"""群聊记忆轴：全量平铺 / 分层群日记 / 精确记忆（老板 2026-10-28 定案）。

- 全量：最近 N 条群消息平铺进上下文，不压缩
- 分层：浅层保留最近若干条原文；被挤出的旧消息压成「群日记」（每群一个 SQLite，
  与私聊日记同机制：标题+时间+标签+摘要，BM25 寻回，零 embedding）
- 精确记忆（高级开关B）：按 (群, QQ) 每人一条独立记忆线，默认 4000 token/人；
  溢出把最老对话压成该群日记条目，标签带该用户 QQ
- 群与群之间记忆隔离：每群一个 JSON + 一个日记库

本模块不 import 平台；LLM 通过 .complete/.text_chat 两种鸭子类型调用。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .core import est_tokens
from .diary import DiaryStore
from .private import PrivateSession, complete_text

GROUP_COMPRESS_PROMPT = (
    "你是群聊记忆整理助手。把下面这段群聊压缩成一篇群日记，帮机器人长期记住这件事。\n"
    "只输出一行 JSON（不要代码块、不要解释）：\n"
    '{"title": "一句话标题，10字以内", "tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、有哪些关键事实/约定/人物关系/情绪"}\n'
    "标签 2~5 个中文词。"
)


class GroupMemory:
    """单个群的记忆：平铺/浅层窗口 + 每人精确记忆线 + 群日记库。"""

    def __init__(self, store_dir, gid: int, diary_dir=None, *,
                 flat_limit: int = 60, shallow_limit: int = 20,
                 precise_budget: int = 4000):
        self.gid = int(gid)
        self.path = Path(store_dir) / f"{self.gid}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flat_limit = max(2, int(flat_limit))
        self.shallow_limit = max(2, int(shallow_limit))
        self.precise_budget = max(200, int(precise_budget))
        self.diary = DiaryStore(diary_dir or store_dir, self.gid)
        self.flat: list[dict] = []
        self.precise: dict[int, list[dict]] = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.flat = d.get("flat") or []
            raw = d.get("precise") or {}
            self.precise = {int(k): v for k, v in raw.items() if str(k).isdigit()}
        except Exception:
            self.flat, self.precise = [], {}

    def save(self) -> None:
        try:
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"flat": self.flat, "precise": self.precise},
                          f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            # 路径被目录占位 / 磁盘故障等：落盘失败不能炸消息处理链，
            # 内存态窗口照常工作，路径恢复后下一条消息自动补盘
            import logging
            logging.getLogger("astrbot").warning(
                f"[qqmind] group memory save failed for {self.gid}: {e!r}")

    def close(self) -> None:
        self.diary.close()

    # ---------- 平铺 / 浅层 ----------
    @staticmethod
    def _msg(uid: int, nick: str, text: str, role: str = "user",
             ts: float | None = None) -> dict:
        return {"role": role, "content": text, "uid": int(uid or 0),
                "nick": nick or "", "ts": float(ts or time.time())}

    def add_group_message(self, uid: int, nick: str, text: str,
                          ts: float | None = None) -> None:
        if not (text or "").strip():
            return
        self.flat.append(self._msg(uid, nick, f"{nick}：{text}", ts=ts))

    def add_assistant(self, text: str, ts: float | None = None) -> None:
        if not (text or "").strip():
            return
        self.flat.append(self._msg(0, "AI", text, role="assistant", ts=ts))

    def take_flat_overflow(self, limit: int) -> list[dict]:
        """保留最近 limit 条，返回被挤出的旧消息（分层模式拿去压日记）。"""
        limit = max(2, int(limit))
        if len(self.flat) <= limit:
            return []
        evicted = self.flat[:-limit]
        self.flat = self.flat[-limit:]
        return evicted

    def flat_context(self, limit: int = 60, skip_last: int = 0) -> list[dict]:
        """最近群消息上下文。skip_last 用于回包时排除“刚读到的这批”，
        避免同一批消息既在 prompt 里又在上下文里被模型算两遍。"""
        rows = self.flat[-max(1, int(limit)):]
        if skip_last > 0:
            rows = rows[:-int(skip_last)] if int(skip_last) < len(rows) else []
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ---------- 精确记忆 ----------
    def add_precise_round(self, uid: int, user_text: str, assistant_text: str,
                          ts: float | None = None) -> None:
        line = self.precise.setdefault(int(uid), [])
        if (user_text or "").strip():
            line.append(self._msg(uid, "", user_text, ts=ts))
        if (assistant_text or "").strip():
            line.append(self._msg(uid, "", assistant_text,
                                  role="assistant", ts=ts))

    def precise_messages(self, uid: int) -> list[dict]:
        return self.precise.get(int(uid), [])

    def precise_context(self, uid: int) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]}
                for m in self.precise_messages(uid)]

    def precise_tokens(self, uid: int) -> int:
        return sum(est_tokens(m["content"]) for m in self.precise_messages(uid))

    def take_precise_overflow(self, uid: int,
                              max_msgs: int = 20) -> list[dict]:
        """超过 4000 token 时取出最老一段（成对取整，给窗口留 2 条尾巴）。"""
        line = self.precise.get(int(uid), [])
        total = sum(est_tokens(m["content"]) for m in line)
        if total <= self.precise_budget:
            return []
        need = total - self.precise_budget
        n, acc = 0, 0
        while n < len(line) and (acc < need or n < 4):
            acc += est_tokens(line[n]["content"])
            n += 1
        n = min(n, max(0, len(line) - 2))
        n -= n % 2
        if n < 4:
            return []
        evicted = line[:n]
        del line[:n]
        return evicted

    # ---------- 压缩进群日记 ----------
    async def compress_messages(self, llm, msgs: list[dict],
                                tags_extra: list[str] | None = None) -> int:
        """把一段群聊压成群日记。返回估算消耗 token（0=失败，调用方下轮再试）。"""
        if not msgs:
            return 0
        lines = []
        for m in msgs:
            if m.get("role") == "assistant":
                who = "AI"
            else:
                who = m.get("nick") or "群友"
            lines.append(f"{who}：{m.get('content', '')}")
        ask = (f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。\n\n"
               + "\n".join(lines))
        try:
            out = await complete_text(llm, GROUP_COMPRESS_PROMPT, ask,
                                      max_tokens=600, temperature=0.3)
        except Exception:
            return 0
        parsed = PrivateSession.parse_compress_output(out)
        if not parsed:
            return 0
        title, tags, summary = parsed
        if tags_extra:
            tags = list(dict.fromkeys([str(t) for t in tags_extra] + tags))[:8]
        tok = sum(est_tokens(m.get("content", "")) for m in msgs)
        self.diary.add(msgs[0].get("ts") or time.time(), tags, summary, tok,
                       title=title)
        self.save()
        return tok + 200

    def diary_headers(self, n: int = 10) -> str:
        return self.diary.headers_for_prompt(n)
