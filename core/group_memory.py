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
from .private import PrivateSession, complete_text, complete_text_resp
from .pool import BATCH_COMPRESS_PROMPT, _est as _pool_est


def _salvage_json_array(text: str) -> list:
    """从被截断的 JSON 数组里捞回已完整的对象（v0.3.0 截断容错）。

    扫描字符级括号配平（忽略字符串内的括号），收集每个顶层 {...} 的
    完整文本，逐个 json.loads；坏一个丢一个，不连坐。
    """
    objs: list = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        objs.append(json.loads(text[start:i + 1]))
                    except Exception:
                        pass
                    start = -1
    return objs

GROUP_COMPRESS_PROMPT = (
    "你是群聊记忆整理助手。把下面这段群聊压缩成一篇群日记，帮机器人长期记住这件事。\n"
    "只输出一行 JSON（不要代码块、不要解释）：\n"
    '{"title": "一句话标题，10字以内", "tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、有哪些关键事实/约定/人物关系/情绪", '
    '"content": "值得留原话的部分：直接引用1~3句最能代表这段对话的原话（保留说话人），'
    '没有值得留的就输出空字符串"}\n'
    "标签 2~5 个中文词。"
)


class GroupMemory:
    """单个群的记忆：平铺/浅层窗口 + 每人精确记忆线 + 群日记库。

    v0.3.0 公共群记忆：多个 GroupMemory 可共享同一个公共 DiaryStore
    （由调用方传入 diary），日记不再是某群私有。
    """

    def __init__(self, store_dir, gid: int, diary_dir=None, *,
                 flat_limit: int = 60, shallow_limit: int = 20,
                 precise_budget: int = 4000, diary: DiaryStore | None = None):
        self.gid = int(gid)
        self.path = Path(store_dir) / f"{self.gid}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flat_limit = max(2, int(flat_limit))
        self.shallow_limit = max(2, int(shallow_limit))
        self.precise_budget = max(200, int(precise_budget))
        self.diary = diary if diary is not None else \
            DiaryStore(diary_dir or store_dir, self.gid)
        self._owns_diary = diary is None    # 共享库由插件统一关
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
        if getattr(self, "_owns_diary", True):
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
        title, tags, summary, content = parsed
        if tags_extra:
            tags = list(dict.fromkeys([str(t) for t in tags_extra] + tags))[:8]
        tok = sum(est_tokens(m.get("content", "")) for m in msgs)
        self.diary.add(msgs[0].get("ts") or time.time(), tags, summary, tok,
                       title=title, content=content, gid=self.gid)
        self.save()
        return tok + 200

    # ---------- 候选池合批压缩（v0.3.0 公共池） ----------
    @staticmethod
    def parse_batch_output(out: str) -> list[dict]:
        """解析合批压缩输出 JSON 数组（容错剥围栏 + 截断 salvage）。

        截断 salvage（v0.3.0）：模型输出超预算被截断时，扫到最后一个完整
        对象为止，捞回已完整的篇目——剩下一刀切整批失败会双倍烧钱。
        """
        t = (out or "").strip()
        if t.startswith("```"):
            t = t.strip("`").lstrip("json").strip()
        try:
            v = json.loads(t)
        except Exception:
            v = _salvage_json_array(t)
        if isinstance(v, dict):     # 容错：单个对象也收
            v = [v]
        if not isinstance(v, list):
            return []
        return [e for e in v if isinstance(e, dict)]

    async def compress_pool_call(self, llm, pool_msgs: list[dict],
                                 prelude: list[dict] | None = None
                                 ) -> tuple[int, list[dict], object]:
        """一次调用把整批原文拆成多篇话题日记。

        prelude：上一批结尾 5 条原文（只供理解，不参与入库）——跨批话题
        不断头。返回 (est_tokens, entries, resp)；est=0 表示调用或解析失败。
        """
        lines = []
        for m in pool_msgs:
            who = "AI" if m.get("role") == "assistant" else \
                  (m.get("nick") or "群友")
            gid = m.get("gid") or 0
            tag = f"群{gid} " if gid else ""
            lines.append(f"[{m['id']}] {tag}{who}：{m.get('text', '')}")
        pre_txt = ""
        if prelude:
            pre_lines = [f"{'群' + str(m.get('gid') or '') + ' ' if m.get('gid') else ''}"
                         f"{m.get('nick') or '群友'}：{m.get('text', '')}"
                         for m in prelude]
            pre_txt = ("【前情提要·上一批结尾（仅供理解上下文，不要重复整理）】\n"
                       + "\n".join(pre_lines) + "\n\n")
        ask = (f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。"
               f"共 {len(pool_msgs)} 条消息。\n\n" + pre_txt + "\n".join(lines))
        try:
            out, resp = await complete_text_resp(
                llm, BATCH_COMPRESS_PROMPT, ask,
                max_tokens=8000, temperature=0.3)
        except Exception:
            # v0.2.15 审查修：调用方按 3 元组解包——少返回一项会让整个
            # flush 任务 ValueError 挂掉，整批消息搁浅在 pending 里，
            # 下批成功时 pending 被覆盖、上一批永久丢失
            return 0, [], None
        entries = self.parse_batch_output(out)
        if not entries:
            return 0, [], None
        est = sum(_pool_est(m.get("text", "")) + 12
                  for m in pool_msgs)
        est += 8000 // 2
        return est, entries, resp

    def compress_pool_batch(self, pool_msgs: list[dict],
                            entries: list[dict]) -> int:
        """把模型拆好的多篇话题日记入库（evidence 记证据消息 id）。

        v0.3.0：每篇的 gid 取证据消息里出现最多的群号（公共记忆里仍记
        来源群，检索/展示可用），写进 entries.gid。
        """
        n = 0
        id2msg = {m["id"]: m for m in pool_msgs}
        for e in entries:
            title = str(e.get("title") or "").strip()[:40]
            summary = str(e.get("summary") or "").strip()
            if not summary:
                continue
            tags = [str(t) for t in (e.get("tags") or [])][:8]
            content = str(e.get("content") or "").strip()[:600]
            ev = [int(x) for x in (e.get("evidence") or [])
                  if str(x).isdigit()][:10]
            gid_counts: dict[int, int] = {}
            for mid in ev:
                m = id2msg.get(mid)
                if m:
                    gid_counts[int(m.get("gid") or 0)] = \
                        gid_counts.get(int(m.get("gid") or 0), 0) + 1
            gid = max(gid_counts, key=gid_counts.get) if gid_counts else \
                int(pool_msgs[0].get("gid") or 0) if pool_msgs else 0
            ts = (id2msg.get(ev[0], {}).get("ts") if ev else None) or \
                 (pool_msgs[0].get("ts") if pool_msgs else time.time())
            eid = self.diary.add(ts, tags, summary, 0,
                                 title=title, content=content, gid=gid)
            if ev:
                self.diary.set_meta(f"evidence_{eid}", json.dumps(ev))
            n += 1
        self.save()
        return n

    def diary_evidence(self, entry_id: int) -> list[int]:
        """某篇日记登记的证据消息 id 列表。"""
        raw = self.diary.get_meta(f"evidence_{int(entry_id)}")
        if not raw:
            return []
        try:
            v = json.loads(raw)
            return [int(x) for x in v if str(x).isdigit()][:10]
        except Exception:
            return []

    def diary_headers(self, n: int = 10) -> str:
        return self.diary.headers_for_prompt(n)
