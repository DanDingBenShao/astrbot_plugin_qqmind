"""群聊记忆体（qqmind）—— AstrBot 插件。

三个挂点（全部走 AstrBot 官方扩展机制，不改 AstrBot 源码）：
1. @filter.event_message_type(ALL) 事件监听：浅层判读门。
   判否 event.should_call_llm(True)，只拦主模型、不终结事件链
   （其他插件与命令照常处理）；判是才交给 AstrBot 走 LLM。
2. @on_llm_request 记忆注入：替换 req.contexts 为我们自己的逐字窗口/私聊预算窗口，
   追加画像、日记标题行、群语境到 system_prompt。
3. @on_llm_response 落盘：把本轮对话写回窗口；预算溢出时后台压缩成日记。

核心逻辑（判读三态/日记库/预算窗口/回捞）在 core/ 包里，与 AstrBot 零耦合，
单测可脱离 AstrBot 跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import on_llm_request, on_llm_response
from astrbot.api.message_components import At, Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.web import request as _web_request

from .core import (JUDGE_PROMPT, MAX_JUDGED_PER_ROUND, ChapterStore,
                   DailyBudget, GroupBuffer, GroupMemory, MonitorSession,
                   PrivateSession, RateLimiter, batch_should_flush,
                   detect_wake, est_tokens, thinking_model,
                   keyword_wake_enabled, normalize_preset,
                   parse_judge_verdict, parse_time_query,
                   preset_memory_mode, resolve_read_mode, split_bubbles)

PLUGIN_VERSION = "0.1.0"
PLUGIN_NAME = "astrbot_plugin_qqmind"

_DIARY_CMD_RE = re.compile(r'<查日记"([^"]{1,60})">')
_GROUP_DIARY_CMD_RE = re.compile(r'<查群日记"([^"]{1,60})">')

# 控制台可读写的配置键（与 _conf_schema.json 对齐）
_CONFIG_KEYS = (
    "persona_text", "persona_shallow", "persona_deep", "persona_pack",
    "persona_packs_dir", "persona_headers_in_prompt", "persona_recall_top",
    "private_budget_tokens", "diary_headers_in_prompt", "diaries_dir",
    "group_preset", "wake_keywords", "group_buffer_size",
    "read_mode", "random_min_seconds", "random_max_seconds", "roam_enabled",
    "debounce_seconds", "batch_max_messages",
    "reply_judge_enabled", "judge_scope", "reply_rate_per_min",
    "max_judged_per_min", "max_judged_per_round",
    "judge_model", "think_model",
    "judge_fail_threshold", "bot_qq_ids", "allowed_groups",
    "group_flat_messages", "group_shallow_messages", "group_context_messages",
    "memory_layering", "precise_memory", "precise_budget_tokens",
    "followup_rounds", "history_rounds", "max_msg_tokens",
    "daily_token_budget", "quota_message", "quota_notices_per_day",
)


def _short_title(summary: str, limit: int = 18) -> str:
    """旧日记没有 title 时的显示兜底：摘要前 limit 字。"""
    s = (summary or "").strip().replace("\n", " ")
    return s[:limit] + ("…" if len(s) > limit else "")


class QQMindPlugin(star.Star):
    """群聊记忆体：判读门 + 深层记忆 + 私聊日记。"""

    def __init__(self, context: star.Context, config=None):
        super().__init__(context)
        self.plugin_config = config    # AstrBotConfig（dict 子类），带 save_config()
        self.sessions: dict[tuple[int, int], MonitorSession] = {}
        self.privates: dict[int, PrivateSession] = {}
        self.memories: dict[int, GroupMemory] = {}
        self.group_buffer = GroupBuffer(
            max_len=int(self._cfg("group_buffer_size", 60)))
        self.rate_limiter = RateLimiter(int(self._cfg("reply_rate_per_min", 6)))
        self.judge_limiter = RateLimiter(int(self._cfg("max_judged_per_min", 30)))
        self._runtime: dict[int, dict] = {}
        self._bot_names: dict[int, str] = {}
        self._judge_fail: dict[int, int] = {}
        self._degraded: set[int] = set()
        self._chapters: dict[str, ChapterStore] = {}
        usage_dir = self._abs_dir("usage_dir", "qqmind_usage")
        self.budget = DailyBudget(os.path.join(usage_dir, "daily.json"), 0)
        self._hist_lock = asyncio.Lock()
        self._register_page_apis()

    # ---------- 配置与存储路径 ----------
    def _cfg(self, key: str, default=None):
        if self.plugin_config is not None:
            return self.plugin_config.get(key, default)
        return default

    def _data_dir(self) -> str:
        import os
        base = getattr(self.context, "get_data_dir", None)
        try:
            d = base() if callable(base) else "data"
        except Exception:
            d = "data"
        return d if os.path.isabs(d) else os.path.join("data", "") and d

    def _abs_dir(self, key: str, default: str) -> str:
        import os
        d = str(self._cfg(key, default))
        if not os.path.isabs(d):
            d = os.path.join("data", d)
        os.makedirs(d, exist_ok=True)
        return d

    # ---------- 存储 ----------
    def _group_mem(self, gid: int) -> GroupMemory:
        gid = int(gid)
        gm = self.memories.get(gid)
        if gm is None:
            gm = GroupMemory(
                self._abs_dir("group_memory_dir", "qqmind_group_memory"), gid,
                diary_dir=self._abs_dir("group_diaries_dir", "qqmind_group_diaries"),
                flat_limit=int(self._cfg("group_flat_messages", 60)),
                shallow_limit=int(self._cfg("group_shallow_messages", 20)),
                precise_budget=int(self._cfg("precise_budget_tokens", 4000)))
            self.memories[gid] = gm
        return gm

    def _private(self, uid: int) -> PrivateSession:
        budget = int(self._cfg("private_budget_tokens", 10000))
        p = self.privates.get(uid)
        if p is None:
            p = PrivateSession(self._abs_dir("diaries_dir", "qqmind_diaries"), uid,
                               budget=budget)
            self.privates[uid] = p
        else:
            p.budget = budget      # 控制台改预算热生效
        return p

    def _note_group(self, gid: int, nick: str, text: str):
        self.group_buffer.note(gid, nick, text)

    # ---------- 成本护栏 ----------
    def _budget(self) -> DailyBudget:
        self.budget.limit = max(0, int(self._cfg("daily_token_budget", 0)))
        return self.budget

    def _budget_exhausted(self) -> bool:
        return self._budget().exhausted()

    def _count_usage(self, *texts, resp=None) -> int:
        usage = getattr(resp, "usage", None)
        n = (int(usage.total) if usage is not None
             else sum(est_tokens(t) for t in texts))
        if n:
            self._budget().add(n)
        return n

    def _maybe_quota_notice(self, gid: int, event: AstrMessageEvent) -> None:
        """额度耗尽时明说一次，每群每天最多 N 次。"""
        if not self._budget().allow_notice(
                gid, int(self._cfg("quota_notices_per_day", 2))):
            return
        msg = str(self._cfg("quota_message", "") or "").strip()
        if not msg or not event.unified_msg_origin:
            return
        asyncio.create_task(self._send_bubble(
            gid, 0, msg, False, event.unified_msg_origin))

    def _is_bot(self, uid: int, event: AstrMessageEvent) -> bool:
        """不响应其他 bot（防互怼循环）：优先看原始 sender 标记，其次配置名单。"""
        raw = getattr(event.message_obj, "raw_message", None)
        try:
            sender = raw.get("sender") if hasattr(raw, "get") else None
        except Exception:
            sender = None
        if isinstance(sender, dict) and (sender.get("is_robot")
                                         or sender.get("is_bot")):
            return True
        bots = {str(x) for x in (self._cfg("bot_qq_ids") or [])}
        return str(uid) in bots

    # ---------- 读取轴 / 回复轴运行时 ----------
    def _read_mode(self) -> str:
        return resolve_read_mode(self._cfg("group_preset", "light"),
                                 self._cfg("read_mode", "batch"))

    def _rt(self, gid: int) -> dict:
        return self._runtime.setdefault(int(gid), {
            "origin": "", "batch": [], "batch_task": None,
            "random_task": None, "last_msg": 0.0})

    def _allow_reply(self, gid: int) -> bool:
        return self.rate_limiter.allow(
            int(gid), per_min=int(self._cfg("reply_rate_per_min", 6)))

    def _allow_judge(self, gid: int) -> bool:
        return self.judge_limiter.allow(
            int(gid), per_min=int(self._cfg("max_judged_per_min", 30)))

    def _has_active_session(self, gid: int) -> bool:
        now = time.time()
        alive = False
        for key, s in list(self.sessions.items()):
            if s.is_expired(now):
                self.sessions.pop(key, None)
                continue
            if key[0] == int(gid) and s.monitoring:
                alive = True
        return alive

    def _cancel_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch"] = []

    def _schedule_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch_task"] = asyncio.create_task(self._batch_flush_later(gid))

    def _flush_batch_now(self, gid: int) -> None:
        """合并上限到了：取消等待，立刻整批判读回复。"""
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch_task"] = None
        asyncio.create_task(self._flush_batch(gid))

    async def _batch_flush_later(self, gid: int) -> None:
        try:
            await asyncio.sleep(float(self._cfg("debounce_seconds", 10.0)))
            await self._flush_batch(gid)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[qqmind] batch flush error: {e!r}")

    def _batch_read_allowed(self, gid: int) -> bool:
        """随机读默认只在监控期内；常驻神游是高级开关。"""
        if self._read_mode() != "random":
            return True
        if self._cfg("roam_enabled", False):
            return True
        return self._has_active_session(gid)

    async def _flush_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        # 先摘掉计时器引用：判读/回复期间新消息不能取消这次回包
        rt["batch_task"] = None
        batch = rt.get("batch") or []
        rt["batch"] = []
        if not batch or not self._batch_read_allowed(gid):
            return
        if self._budget_exhausted():
            # 熔断点补全：关判读时这里原来拦不住，攒批会继续烧主回复
            self._STATS["dropped_budget"] += len(batch)
            return
        if self._cfg("reply_judge_enabled", True):
            if not self._allow_judge(gid):
                self._STATS["dropped_rate"] += len(batch)
                logger.info(f"[qqmind] batch dropped (judge rate) in {gid}: "
                            f"{len(batch)} msgs")
                return
            ok, _ = await self._judge(gid, "", "", batch=batch,
                                      bot_name=self._bot_names.get(int(gid), ""))
            if not ok:
                return
        if not self._allow_reply(gid):
            self._STATS["dropped_rate"] += len(batch)
            logger.info(f"[qqmind] batch dropped (reply rate) in {gid}: "
                        f"{len(batch)} msgs")
            return
        last = batch[-1]
        prompt = "\n".join(f"{m['nick']}：{m['text']}" for m in batch)
        await self._reply_group(gid, last["uid"], prompt, rt.get("origin") or "",
                                batch=batch)

    def _ensure_random_loop(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("random_task")
        if task and not task.done():
            return
        rt["random_task"] = asyncio.create_task(self._random_loop(gid))

    async def _random_loop(self, gid: int) -> None:
        while True:
            lo = max(1, int(self._cfg("random_min_seconds", 20)))
            hi = max(lo, int(self._cfg("random_max_seconds", 300)))
            await asyncio.sleep(random.uniform(lo, hi))
            if self._read_mode() != "random":
                self._rt(gid)["random_task"] = None
                return
            try:
                await self._flush_batch(gid)
            except Exception as e:
                logger.warning(f"[qqmind] random read error: {e!r}")

    async def _send_bubble(self, gid: int, uid: int, text: str,
                           first: bool, origin: str) -> bool:
        """主动发一条群消息：优先走 OneBot（明确 self_id），没有可用通道才退回 context。

        注意：OneBot 通道存在但调用失败（风控/断连/权限）时**不再重发**——
        兜底通道只补「通道缺失」，不补「通道被拒」，否则一次故障会变成双发
        （假 bot 里的 context_send 记录会把拒绝当成功，真实环境则可能重复刷屏）。
        """
        rt = self._rt(gid)
        bot = rt.get("bot")
        self_id = str(rt.get("self_id") or "")
        if bot is None or not self_id or not hasattr(bot, "call_action"):
            # 通道缺失（OneBot 断连 / bot=None）：走 context 兜底
            try:
                parts = ([At(qq=str(uid)), Plain(" " + text)] if first
                         else [Plain(text)])
                await self.context.send_message(origin, MessageChain(parts))
                return True
            except Exception as e:
                logger.warning(f"[qqmind] send failed: {e!r}")
                return False
        segs = []
        body = (" " + text) if first else text
        if first:
            segs.append({"type": "at", "data": {"qq": str(uid)}})
        segs.append({"type": "text", "data": {"text": body}})
        try:
            await bot.call_action("send_group_msg", self_id=int(self_id),
                                  group_id=int(gid), message=segs)
            return True
        except Exception as e:
            logger.warning(f"[qqmind] onebot send failed: {e!r}")
            return False

    async def _reply_group(self, gid: int, uid: int, prompt: str,
                           origin: str, batch=None) -> None:
        """攒批/随机读的主动回复路径（没有事件可交给 AstrBot 原生管线）。

        与原生管线共用同一条后处理链：回捞（<查群日记>）→ 分段 → 发送 → 落盘。
        """
        provider = self.context.get_using_provider()
        if provider is None or not origin:
            return
        try:
            out = await provider.text_chat(
                prompt=prompt, session_id=None,
                system_prompt=self._group_system_prompt(gid, prompt),
                contexts=self._group_contexts(
                    gid, uid, skip_last=len(batch) if batch else 0),
                image_urls=[])
            reply = (out.completion_text or "").strip()
        except Exception as e:
            logger.warning(f"[qqmind] manual reply error: {e!r}")
            return
        if not reply:
            return
        self._count_usage(prompt, reply, resp=out)
        reply = await self._recall_reply(reply, "group", uid, gid)
        if not reply:
            return
        bubbles = split_bubbles(reply)
        delivered = 0
        for i, b in enumerate(bubbles):
            ok = await self._send_bubble(gid, uid, b, i == 0, origin)
            if not ok:
                break
            delivered += 1
            if i < len(bubbles) - 1:
                await asyncio.sleep(0.4 + 0.5 * random.random())
        messages = ([(m["uid"], m["nick"], m["text"]) for m in batch]
                    if batch else [(uid, "", prompt)])
        async with self._hist_lock:
            self._persist_group(gid, uid, messages, reply)
        self._note_group(gid, self._bot_names.get(gid) or "AI", reply)
        if delivered:
            # 一段都没发出去的不算成功回复（避免失败轮虚增面板 delivered 数）
            self._STATS["replied"] += 1

    def _memory_mode(self) -> str:
        preset = normalize_preset(self._cfg("group_preset", "light"))
        if preset == "advanced":
            if self._cfg("precise_memory", False):
                return "precise"
            return "layered" if self._cfg("memory_layering", True) else "flat"
        return preset_memory_mode(preset) or "flat"

    def _is_light(self) -> bool:
        return normalize_preset(self._cfg("group_preset", "light")) == "light"

    def _group_contexts(self, gid: int, uid: int,
                        skip_last: int = 0) -> list[dict]:
        mode = self._memory_mode()
        if mode == "precise":
            return self._group_mem(gid).precise_context(uid)
        if self._is_light():
            return []
        limit = (int(self._cfg("group_flat_messages", 60))
                 if mode == "flat"
                 else int(self._cfg("group_shallow_messages", 20)))
        return self._group_mem(gid).flat_context(limit, skip_last=skip_last)

    # ---------- 人设三层 + 背景故事 ----------
    def _persona_block(self) -> str:
        """浅层（当前状态）+ 深层（核心性格）；都空时兼容旧版单卡。"""
        shallow = str(self._cfg("persona_shallow", "") or "").strip()
        deep = str(self._cfg("persona_deep", "") or "").strip()
        if not shallow and not deep:
            return str(self._cfg("persona_text", "") or "").strip()
        parts = []
        if deep:
            parts.append(deep)
        if shallow:
            parts.append("【当前状态】\n" + shallow)
        return "\n".join(parts)

    def _chapter_store(self) -> ChapterStore | None:
        pack = str(self._cfg("persona_pack", "") or "").strip()
        if not pack:
            return None
        return self._chapter_store_for(pack)

    def _chapter_store_for(self, pack: str) -> ChapterStore:
        pack = str(pack or "").strip() or "default"
        cs = self._chapters.get(pack)
        if cs is None:
            cs = ChapterStore(
                self._abs_dir("persona_packs_dir", "qqmind_persona_packs"),
                pack)
            self._chapters[pack] = cs
        return cs

    def _persona_background(self, query: str = "") -> str:
        """标题行常驻；当前消息命中关键词才带摘要+正文。"""
        cs = self._chapter_store()
        if cs is None:
            return ""
        parts = []
        n_headers = int(self._cfg("persona_headers_in_prompt", 20))
        headers = cs.headers_for_prompt(n_headers)
        if headers:
            parts.append("【背景故事篇目】\n" + headers)
        if query:
            hits = cs.search(query, int(self._cfg("persona_recall_top", 2)))
            if hits:
                parts.append("【与本条消息相关的背景】\n" + hits)
        return "\n".join(parts)

    def _group_system_prompt(self, gid: int, query: str = "") -> str:
        parts = []
        persona = self._persona_block()
        if persona:
            parts.append(persona)
        background = self._persona_background(query)
        if background:
            parts.append(background)
        mode = self._memory_mode()
        if mode in ("layered", "precise"):
            headers = self._group_mem(gid).diary_headers(
                int(self._cfg("diary_headers_in_prompt", 10)))
            if headers:
                parts.append("你有本群的日记（时间 标题，想细看就输出 "
                             "<查群日记\"关键词或时间\">）：\n" + headers)
        if mode == "precise":
            n_ctx = 10
        elif self._is_light():
            n_ctx = int(self._cfg("group_context_messages", 20))
        else:
            n_ctx = 0
        if n_ctx:
            ctx = "\n".join(self.group_buffer.recent(gid, n_ctx))
            if ctx:
                parts.append(f"【最近的群聊（供理解语境，不必逐条回应）】\n{ctx}")
        return "\n".join(parts)

    def _record_incoming(self, gid: int, uid: int, nick: str, text: str) -> None:
        """全量/分层模式把每条群消息记进平铺窗口（轻量不记，精确记忆只记互动）。"""
        if not (text or "").strip() or self._is_light():
            return
        mode = self._memory_mode()
        if mode == "precise":
            return
        gm = self._group_mem(gid)
        gm.add_group_message(uid, nick, text)
        limit = (int(self._cfg("group_flat_messages", 60))
                 if mode == "flat"
                 else int(self._cfg("group_shallow_messages", 20)))
        evicted = gm.take_flat_overflow(limit)
        if evicted and mode == "layered":
            asyncio.create_task(self._compress_or_restore(gid, evicted))
        gm.save()

    def _persist_group(self, gid: int, uid: int,
                       messages: list[tuple[int, str, str]], reply: str) -> None:
        """群回复落盘：按记忆模式平铺 / 压日记 / 进精确记忆线。"""
        if self._is_light():
            return
        mode = self._memory_mode()
        gm = self._group_mem(gid)
        if mode == "precise":
            user_text = "\n".join(t for _, _, t in messages if t)
            gm.add_precise_round(uid, user_text, reply)
            over = gm.take_precise_overflow(uid)
            if over:
                asyncio.create_task(
                    self._compress_or_restore(gid, over, [str(uid)], uid))
        else:
            gm.add_assistant(reply)
            limit = (int(self._cfg("group_flat_messages", 60))
                     if mode == "flat"
                     else int(self._cfg("group_shallow_messages", 20)))
            evicted = gm.take_flat_overflow(limit)
            if evicted and mode == "layered":
                asyncio.create_task(self._compress_or_restore(gid, evicted))
        gm.save()

    async def _compress_or_restore(self, gid: int, msgs: list[dict],
                                   tags_extra=None, precise_uid=None) -> None:
        """压进群日记；失败把原文放回窗口，下轮再试（不丢记忆）。"""
        provider = self.context.get_using_provider()
        gm = self._group_mem(gid)
        used = 0
        if provider is not None:
            used = await gm.compress_messages(provider, msgs, tags_extra)
        if used:
            self._budget().add(int(used))
            logger.info(f"[qqmind] group diary entry for {gid}, "
                        f"{len(msgs)} msgs")
            return
        if precise_uid is not None:
            gm.precise.setdefault(int(precise_uid), [])[:0] = msgs
        else:
            gm.flat[:0] = msgs
        gm.save()

    # ---------- 挂点 1：事件监听（判读门） ----------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """群消息判读门 + 私聊直通。判否 should_call_llm(True)：
        只拦主模型，不终结事件链——其他插件和命令照常处理。"""
        # 判读门只管群聊；私聊直接放行给挂点 2/3 处理
        if event.is_private_chat():
            return
        gid = int(event.get_group_id() or 0)
        uid = int(event.get_sender_id() or 0)
        self_id = str(event.get_self_id() or "")
        if not gid or not uid or str(uid) == self_id:
            return
        if self._is_bot(uid, event):
            # bot 消息（防互怼）：qqmind 不理，也不许主模型理
            event.should_call_llm(True)
            return
        allowed = self._cfg("allowed_groups") or []
        if allowed and str(gid) not in [str(g) for g in allowed]:
            # 白名单外：qqmind 完全隐身，既不判读也不碰任何标志
            return
        nick = (event.get_sender_name() or "").strip() or str(uid)
        text = (event.message_str or "").strip()
        preset = normalize_preset(self._cfg("group_preset", "light"))
        keywords = (self._cfg("wake_keywords", [])
                    if keyword_wake_enabled(preset) else [])
        wake = detect_wake(text, event.get_messages() or [], self_id, keywords)
        self._note_group(gid, nick, text or "[图片]")
        rt = self._rt(gid)
        rt["origin"] = event.unified_msg_origin
        rt["bot"] = getattr(event, "bot", None)
        rt["self_id"] = str(event.get_self_id() or "")
        rt["last_msg"] = time.time()
        mode = self._read_mode()

        if wake.at_all and not wake.at_me:
            # @全体成员不唤醒（写死），但消息照样进缓冲；只拦主模型
            event.should_call_llm(True)
            return

        key = (gid, uid)
        if wake.at_me:
            # @/引用我 必回：丢掉待回复批，开监控期并直接放行给主模型
            self._degraded.discard(gid)
            self._judge_fail.pop(gid, None)
            self._record_incoming(gid, uid, nick, text)
            self._cancel_batch(gid)
            s = MonitorSession(followup_rounds=int(self._cfg("followup_rounds", 3)),
                               idle_timeout_min=5)
            self.sessions[key] = s
            logger.info(f"[qqmind] @ wake for {uid} in group {gid}")
            return
        if self._budget_exhausted():
            # 预算烧完 → 降级轻量：被动消息不读；只拦主模型
            self._maybe_quota_notice(gid, event)
            event.should_call_llm(True)
            return
        if gid in self._degraded:
            # 判读连续失败 → 自动降级轻量；只拦主模型
            event.should_call_llm(True)
            return
        self._record_incoming(gid, uid, nick, text)
        if wake.keyword and keyword_wake_enabled(preset):
            # 关键词：开门（随机读靠它进入监控期），是否回复仍由判读决定
            s = self.sessions.get(key)
            if not s or not s.monitoring:
                s = MonitorSession(
                    followup_rounds=int(self._cfg("followup_rounds", 3)),
                    idle_timeout_min=5)
                self.sessions[key] = s
                logger.info(f"[qqmind] keyword wake for {uid} in group {gid}")
            else:
                s.touch()

        if mode == "full":
            await self._read_full(event, gid, uid, nick, text)
            return
        if mode in ("batch", "random"):
            # 攒批/随机读：消息进待读批即可，主模型一律不放（判读后主动回复）
            if text:
                rt["batch"].append({"uid": uid, "nick": nick, "text": text})
                if mode == "batch":
                    max_batch = max(1, int(self._cfg("batch_max_messages", 10)))
                    if batch_should_flush(len(rt["batch"]), max_batch):
                        self._flush_batch_now(gid)
                    else:
                        self._schedule_batch(gid)
                else:
                    self._ensure_random_loop(gid)
            event.should_call_llm(True)
            return

        # ---- at_only（轻量档）：@ 后监控期内逐条判读 ----
        s = self.sessions.get(key)
        if not s or not s.monitoring:
            # 监控期外：默认不接话（judge_scope=false），只拦主模型
            if not self._cfg("judge_scope", False):
                event.should_call_llm(True)
                return
            # 全群判读模式：没会话也建一个，保证额度与超时生效
            s = MonitorSession(
                followup_rounds=int(self._cfg("followup_rounds", 3)),
                idle_timeout_min=5)
            self.sessions[key] = s
        if not text:
            event.should_call_llm(True)
            return
        if s.judged >= int(self._cfg("max_judged_per_round", MAX_JUDGED_PER_ROUND)):
            event.should_call_llm(True)
            return
        if not self._allow_judge(gid):
            event.should_call_llm(True)
            return
        s.judged += 1
        bot_name = await self._bot_group_name(event, gid)
        ok, think = await self._judge(gid, nick, text, bot_name)
        self._STATS["judged"] += 1
        s = self.sessions.get(key)
        if not s or not s.monitoring:
            event.should_call_llm(True)
            return
        if not ok:
            self._STATS["blocked"] += 1
            s.rounds_left -= 1
            if s.rounds_left <= 0:
                self.sessions.pop(key, None)
                logger.info(f"[qqmind] monitoring closed for {uid}")
            event.should_call_llm(True)
            return
        if not self._allow_reply(gid):
            event.should_call_llm(True)
            return
        # 判是：标记为已唤醒，AstrBot 后续 LLM 管线才会真正生成回复
        event.is_at_or_wake_command = True
        event.is_wake = True
        # 思考模式由判读控制（老板 2026-10-29 定案）：判「是|思考」的这一轮
        # 把思考标记挂到事件上，inject_memory 据此切换思考模型；判「直接」不开
        setattr(event, "qqmind_think", bool(think))
        s.touch()   # 额度在 on_llm_response 刷新

    async def _read_full(self, event: AstrMessageEvent, gid: int, uid: int,
                         nick: str, text: str) -> None:
        """全量读：每条消息都过判读门（判是才放行原生管线）。"""
        if not text:
            event.should_call_llm(True)
            return
        if not self._cfg("reply_judge_enabled", True):
            if not self._allow_reply(gid):
                self._STATS["dropped_rate"] += 1
                event.should_call_llm(True)
                return
            event.is_at_or_wake_command = True
            event.is_wake = True
            return
        if not self._allow_judge(gid):
            self._STATS["dropped_rate"] += 1
            event.should_call_llm(True)
            return
        bot_name = await self._bot_group_name(event, gid)
        ok, think = await self._judge(gid, nick, text, bot_name)
        self._STATS["judged"] += 1
        if not ok:
            self._STATS["blocked"] += 1
            event.should_call_llm(True)
            return
        if not self._allow_reply(gid):
            self._STATS["dropped_rate"] += 1
            event.should_call_llm(True)
            return
        event.is_at_or_wake_command = True
        event.is_wake = True
        setattr(event, "qqmind_think", bool(think))   # 思考模式由判读控制

    async def _judge(self, gid: int, nick: str, text: str,
                     bot_name: str = "", batch=None) -> tuple[bool, bool]:
        """浅层判读。batch 非空时整批判一次；出错 fail-closed。"""
        if self._budget_exhausted():
            return False, False
        provider = self.context.get_using_provider()
        if provider is None:
            return False, False
        k = len(batch) if batch else 1
        ctx_all = self.group_buffer.recent(gid, 7 + k)
        ctx_lines = ctx_all[:-k][-7:] if len(ctx_all) >= k else []
        pending = ("\n".join(f"{m['nick']}：{m['text']}" for m in batch)
                   if batch else f"{nick}：{text}")
        question = ("以上这些消息是在对AI助手说话吗？若在，需要AI深度思考吗？"
                    if batch else
                    "这条消息是在对AI助手说话吗？若在，需要AI深度思考吗？")
        who = f"AI助手在本群的名字是「{bot_name}」。\n" if bot_name else ""
        if ctx_lines:
            ask = (who + "【最近群聊记录】\n" + "\n".join(ctx_lines)
                   + f"\n【待判断消息】\n{pending}\n{question}")
        else:
            ask = who + f"【待判断消息】\n{pending}\n{question}"
        logger.debug(f"[qqmind] judge ask: {ask[:260]!r}")
        judge_model = str(self._cfg("judge_model", "") or "").strip()
        try:
            out = await provider.text_chat(
                prompt=ask, session_id=None,
                system_prompt=JUDGE_PROMPT,
                contexts=[], image_urls=[],
                model=judge_model or None)
            out_text = out.completion_text or ""
        except Exception as e:
            fails = self._judge_fail.get(gid, 0) + 1
            self._judge_fail[gid] = fails
            threshold = int(self._cfg("judge_fail_threshold", 3))
            if fails >= threshold:
                self._degraded.add(gid)
                logger.warning(f"[qqmind] judge failed {fails}x in group {gid}; "
                               f"degraded to light until next @")
            logger.warning(f"[qqmind] judge error (fail-closed): {e!r}")
            return False, False
        self._count_usage(ask, out_text, resp=out)
        verdict = parse_judge_verdict(out_text)
        if verdict is None:
            # 模型输出不合规格：按失败计，走 fail-closed（宁漏接不插嘴）
            # 注意：成功调用但输出垃圾也是判读失败——计数与降级判定必须放在
            # 这里（verdict 解析之后），不能在调用成功时就清零
            fails = self._judge_fail.get(gid, 0) + 1
            self._judge_fail[gid] = fails
            threshold = int(self._cfg("judge_fail_threshold", 3))
            if fails >= threshold:
                self._degraded.add(gid)
                logger.warning(f"[qqmind] judge malformed {fails}x in group "
                               f"{gid}; degraded to light until next @")
            logger.warning(f"[qqmind] judge malformed output, fail-closed: "
                           f"{out_text[:60]!r}")
            return False, False
        self._judge_fail[gid] = 0
        self._degraded.discard(gid)
        ok, think = verdict
        self._STATS["judge_think"] += int(think)
        logger.info(f"[qqmind] judge -> {'是|思考' if think else ('是|直接' if ok else '否')}: {pending[:40]}")
        return ok, think

    async def _bot_group_name(self, event: AstrMessageEvent, gid: int) -> str:
        """机器人在本群的群名片（判读提示用，识别「叫的是我」）。每群缓存一次。"""
        if gid in self._bot_names:
            return self._bot_names[gid]
        name = ""
        me = str(event.get_self_id() or "")
        bot = getattr(event, "bot", None)
        if bot is not None and me:
            try:
                info = await bot.call_action(
                    "get_group_member_info", self_id=int(me),
                    group_id=int(gid), user_id=int(me))
                if isinstance(info, dict):
                    name = (info.get("card") or info.get("nickname") or "").strip()
            except Exception as e:
                logger.debug(f"[qqmind] member info lookup failed: {e!r}")
        if not name:
            try:
                group = await event.get_group()
                for m in (getattr(group, "members", None) or []):
                    if str(getattr(m, "user_id", "")) == me:
                        name = (getattr(m, "nickname", "") or "").strip()
                        break
            except Exception as e:
                logger.debug(f"[qqmind] bot group name lookup failed: {e!r}")
        self._bot_names[gid] = name
        return name

    # ---------- 挂点 2：LLM 请求前（记忆注入） ----------
    @on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        uid = int(event.get_sender_id() or 0)
        if not uid:
            return
        if event.is_private_chat():
            p = self._private(uid)
            profile = self._private_system_prompt(p, event.message_str or "")
            if profile:
                req.system_prompt = (profile + "\n"
                                     + (req.system_prompt or "")).strip()
            # 用我们自己的窗口替换 AstrBot 自带历史（避免双份 token）
            req.contexts = [{"role": m["role"], "content": m["content"]}
                            for m in p.messages]
            return
        gid = int(event.get_group_id() or 0)
        profile = self._group_system_prompt(gid, event.message_str or "")
        if profile:
            req.system_prompt = (profile + "\n"
                                 + (req.system_prompt or "")).strip()
        req.contexts = self._group_contexts(gid, uid)
        self._apply_think_model(event, req)

    def _apply_think_model(self, event, req: ProviderRequest) -> None:
        """思考模式由判读控制（老板 2026-10-29 定案）。

        判读判「是|思考」的当轮：主回复切到思考模型（think_model 配置，
        缺省用 thinking_model() 从主模型名推导 nothink→thinking 别名）；
        判「是|直接」或 @ 路径未判读的：保持主模型不动。
        """
        if not getattr(event, "qqmind_think", False):
            return
        think_model = str(self._cfg("think_model", "") or "").strip()
        if not think_model:
            think_model = thinking_model(
                getattr(self.context.get_using_provider(), "get_model", lambda: "")())
        if think_model:
            req.model = think_model
            logger.debug(f"[qqmind] think round -> model={think_model}")

    def _diary_headers(self, p: PrivateSession) -> str:
        n = int(self._cfg("diary_headers_in_prompt", 10))
        headers = p.diary.headers_for_prompt(n)
        if not headers:
            return ""
        return (f"你有关于这位用户的日记（时间 标题，想细看就输出 <查日记\"关键词或时间\">）：\n"
                f"{headers}")

    def _private_system_prompt(self, p: PrivateSession,
                               query: str = "") -> str:
        parts = []
        persona = self._persona_block()
        if persona:
            parts.append(persona)
        background = self._persona_background(query)
        if background:
            parts.append(background)
        headers = self._diary_headers(p)
        if headers:
            parts.append(headers)
        return "\n".join(parts)

    async def _recall_reply(self, reply: str, scope: str, uid: int,
                            gid: int = 0) -> str:
        """统一回捞：处理 <查日记>/<查群日记>，查库后带结果重答一轮。

        scope: "private" | "group"。原生管线（on_llm_response）与手动路径
        （攒批/随机读）共用，保证两条回复路径能力对等。
        返回清洗后的最终文本（不再含任何回捞标记）。
        """
        cmd_re = _DIARY_CMD_RE if scope == "private" else _GROUP_DIARY_CMD_RE
        m = cmd_re.search(reply)
        if not m:
            return reply
        query = m.group(1)
        if scope == "private":
            p = self._private(uid)
            found = (p.diary.by_time(query) if parse_time_query(query)
                     else p.diary.search(query))
            system = self._private_system_prompt(p)
            contexts = [{"role": x["role"], "content": x["content"]}
                        for x in p.messages]
        else:
            gm = self._group_mem(gid)
            found = (gm.diary.by_time(query) if parse_time_query(query)
                     else gm.diary.search(query))
            system = self._group_system_prompt(gid)
            contexts = self._group_contexts(gid, uid)
        provider = self.context.get_using_provider()
        if provider is None:
            return reply.replace(m.group(0), "").strip() or \
                "我翻了一下日记，但没找到合适的答案。"
        try:
            out = await provider.text_chat(
                prompt=(f"（你刚才想查记忆：{query}）\n查询结果：\n{found}\n"
                        "请根据查询结果，用自然的语气回答对方。"),
                session_id=None, system_prompt=system,
                contexts=contexts, image_urls=[])
            new_reply = (out.completion_text or "").strip()
            if new_reply:
                self._count_usage(found, new_reply, resp=out)
                clean = _DIARY_CMD_RE.sub("", new_reply)
                clean = _GROUP_DIARY_CMD_RE.sub("", clean).strip()
                logger.info(f"[qqmind] diary recall '{query}' answered")
                return clean or new_reply
        except Exception as e:
            logger.warning(f"[qqmind] diary recall failed: {e!r}")
        stripped = reply.replace(m.group(0), "").strip()
        return stripped or "我翻了一下日记，但没找到合适的答案。"

    async def _resolve_recall(self, event: AstrMessageEvent, resp,
                              reply: str) -> str:
        """原生管线挂点：解析回捞标记并写回 resp.completion_text。"""
        uid = int(event.get_sender_id() or 0)
        scope = "private" if event.is_private_chat() else "group"
        gid = int(event.get_group_id() or 0)
        new_reply = await self._recall_reply(reply, scope, uid, gid)
        if new_reply != reply:
            resp.completion_text = new_reply
        return new_reply

    # ---------- 挂点 3：LLM 回复后（落盘/压缩/额度刷新） ----------
    @on_llm_response()
    async def persist_round(self, event: AstrMessageEvent, resp):
        uid = int(event.get_sender_id() or 0)
        reply = (getattr(resp, "completion_text", "") or "").strip()
        if not uid or not reply:
            return
        self._count_usage(event.message_str or "", reply, resp=resp)
        reply = await self._resolve_recall(event, resp, reply)
        if not reply:
            return
        self._STATS["replied"] += 1
        text = (event.message_str or "").strip()
        if event.is_private_chat():
            p = self._private(uid)
            cap = int(self._cfg("max_msg_tokens", 100)) * 4
            async with self._hist_lock:
                if text and est_tokens(text) <= cap:
                    p.messages.append({"role": "user", "content": text, "ts": time.time()})
                p.messages.append({"role": "assistant", "content": reply, "ts": time.time()})
                need = p.over_budget()
            if need:
                asyncio.create_task(self._compress_private(p))
        else:
            gid = int(event.get_group_id() or 0)
            key = (gid, uid)
            nick = (event.get_sender_name() or "").strip() or str(uid)
            cap = int(self._cfg("max_msg_tokens", 100))
            user_text = text if (text and est_tokens(text) <= cap) else ""
            async with self._hist_lock:
                self._persist_group(gid, uid, [(uid, nick, user_text)], reply)
            # 机器人自己的回复也进群缓冲：轻量档语境全靠它，
            # 判读提示词也承诺「含 AI 助手的发言」——缺了会重复回答/接不上话
            self._note_group(gid, self._bot_names.get(gid) or "AI", reply)
            s = self.sessions.get(key)
            if s:
                s.rounds_left = int(self._cfg("followup_rounds", 3))
                s.touch()

    async def _compress_private(self, p: PrivateSession):
        provider = self.context.get_using_provider()
        if provider is None:
            return
        for _ in range(3):
            if not p.over_budget():
                return
            ok = await p.compress_oldest(provider)
            if not ok:
                return
            self._budget().add(int(ok))
        logger.info(f"[qqmind] private compressed for {p.uid}, "
                    f"window={p.window_tokens()}/{p.budget} tok")

    async def terminate(self) -> None:
        """插件卸载/重载：停掉随机读与攒批任务，关掉日记库句柄。"""
        for rt in self._runtime.values():
            for key in ("batch_task", "random_task"):
                t = rt.get(key)
                if t and not t.done():
                    t.cancel()
        self._runtime.clear()
        for p in self.privates.values():
            p.diary.close()
        self.privates.clear()
        for gm in self.memories.values():
            gm.close()
        self.memories.clear()
        for cs in self._chapters.values():
            cs.close()
        self._chapters.clear()

    # ---------- 配置页后端 API（views/console 通过面板桥调用） ----------
    _STATS = {"judged": 0, "blocked": 0, "replied": 0, "judge_think": 0,
              "dropped_rate": 0, "dropped_budget": 0}

    def _register_page_apis(self):
        """注册 views/console 页面用的后端接口（面板同源鉴权）。"""

        async def api_stats():
            diary_dir = self._abs_dir("diaries_dir", "qqmind_diaries")
            diary_total = 0
            try:
                for fn in os.listdir(diary_dir):
                    if fn.endswith(".db") and fn[:-3].isdigit():
                        diary_total += self._private(int(fn[:-3])).diary.count()
            except OSError:
                pass
            max_win = max((p.window_tokens() for p in self.privates.values()),
                          default=0)
            return {"status": "ok", "data": {
                "judged": self._STATS["judged"],
                "blocked": self._STATS["blocked"],
                "replied": self._STATS["replied"],
                "judge_think": self._STATS["judge_think"],
                "dropped_rate": self._STATS["dropped_rate"],
                "dropped_budget": self._STATS["dropped_budget"],
                "diary_total": diary_total,
                "max_window_tokens": max_win,
                "budget": self._budget().snapshot()}}

        async def api_get_config():
            c = self.plugin_config or {}
            return {"status": "ok", "data": {
                k: c.get(k) for k in _CONFIG_KEYS}}

        async def api_set_config():
            payload = await _web_request.json(default={}) or {}
            c = self.plugin_config
            if c is None:
                return {"status": "error", "message": "config unavailable"}
            for key in _CONFIG_KEYS:
                if key in payload:
                    c[key] = payload[key]
            try:
                c.save_config()
            except Exception as e:
                logger.warning(f"[qqmind] save_config failed: {e!r}")
            return {"status": "ok"}

        async def api_diaries():
            import time as _t
            uid = str(_web_request.query.get("uid") or "").strip()
            if uid:
                p = self._private(int(uid))
            elif self.privates:
                latest_uid = max(self.privates,
                                 key=lambda u: self.privates[u].diary.count())
                p = self.privates[latest_uid]
            else:
                return {"status": "ok", "data": {"entries": []}}
            rows = p.diary.recent_rows(10)
            entries = [{
                "ts": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(r["ts"])),
                "title": (r["title"] or "").strip() or _short_title(r["summary"]),
                "tags": "、".join(json.loads(r["tags"])),
                "summary": r["summary"]} for r in rows]
            return {"status": "ok", "data": {"entries": entries}}

        async def api_sessions():
            """有私聊日记的用户列表：扫磁盘日记库（不只内存里活跃的），
            内存中有会话的补窗口 token。按日记数排序。"""
            import os
            diary_dir = self._abs_dir("diaries_dir", "qqmind_diaries")
            disk_uids = []
            try:
                for fn in os.listdir(diary_dir):
                    if fn.endswith(".db") and fn[:-3].isdigit():
                        disk_uids.append(int(fn[:-3]))
            except OSError:
                pass
            users = []
            for u in sorted(set(disk_uids) | set(self.privates.keys())):
                p = self.privates.get(u)
                n = p.diary.count() if p else \
                    self._private(u).diary.count()
                users.append({"uid": str(u), "diaries": n,
                              "window_tokens": p.window_tokens() if p else 0})
            users.sort(key=lambda x: -x["diaries"])
            return {"status": "ok", "data": {"users": users}}

        async def api_packs():
            """背景故事角色包列表。"""
            packs_dir = self._abs_dir("persona_packs_dir", "qqmind_persona_packs")
            packs = []
            try:
                for fn in sorted(os.listdir(packs_dir)):
                    if fn.endswith(".db"):
                        name = fn[:-3]
                        packs.append({
                            "name": name,
                            "chapters": self._chapter_store_for(name).count()})
            except OSError:
                pass
            active = str(self._cfg("persona_pack", "") or "").strip()
            names = {p["name"] for p in packs}
            if active and active not in names:
                packs.append({"name": active,
                              "chapters": self._chapter_store_for(active).count()})
                packs.sort(key=lambda x: x["name"])
            return {"status": "ok", "data": {"packs": packs, "active": active}}

        async def api_chapters():
            pack = str(_web_request.query.get("pack") or "").strip()
            if not pack:
                return {"status": "ok", "data": {"chapters": []}}
            rows = self._chapter_store_for(pack).list_rows(500)
            return {"status": "ok", "data": {"chapters": [
                {"id": r["id"], "title": r["title"],
                 "summary": r["summary"], "content": r["content"]}
                for r in rows]}}

        async def api_chapter_save():
            payload = await _web_request.json(default={}) or {}
            pack = str(payload.get("pack") or "").strip()
            if not pack:
                return {"status": "error", "message": "缺少角色包名"}
            title = str(payload.get("title") or "").strip()
            content = str(payload.get("content") or "").strip()
            if not title:
                return {"status": "error", "message": "标题不能为空"}
            cs = self._chapter_store_for(pack)
            cid = payload.get("id")
            if cid:
                ok = cs.update(int(cid), title,
                               str(payload.get("summary") or ""), content)
                return {"status": "ok" if ok else "error", "data": {"id": int(cid)}}
            new_id = cs.add(title, str(payload.get("summary") or ""), content)
            return {"status": "ok", "data": {"id": new_id}}

        async def api_chapter_delete():
            payload = await _web_request.json(default={}) or {}
            pack = str(payload.get("pack") or "").strip()
            cid = payload.get("id")
            if not pack or not cid:
                return {"status": "error", "message": "缺少参数"}
            ok = self._chapter_store_for(pack).delete(int(cid))
            return {"status": "ok" if ok else "error"}

        self.context.register_web_api(
            f"/{PLUGIN_NAME}/stats", api_stats, ["GET"], "qqmind 今日账本")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/config", api_get_config, ["GET"], "qqmind 读配置")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/config", api_set_config, ["POST"], "qqmind 写配置")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/diaries", api_diaries, ["GET"], "qqmind 日记浏览")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/sessions", api_sessions, ["GET"], "qqmind 用户列表")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/packs", api_packs, ["GET"], "qqmind 角色包列表")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/chapters", api_chapters, ["GET"], "qqmind 篇章列表")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/chapter", api_chapter_save, ["POST"], "qqmind 存篇章")
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/chapter/delete", api_chapter_delete, ["POST"],
            "qqmind 删篇章")

    # ---------- 指令 ----------
    @filter.command("qqmind")
    async def cmd_qqmind(self, event: AstrMessageEvent):
        """查看插件状态与记忆账本。"""
        uid = int(event.get_sender_id() or 0)
        n_mem = len(self.memories)
        n_priv = len(self.privates)
        diaries = self.privates[uid].diary.count() if uid in self.privates else 0
        b = self._budget().snapshot()
        budget_line = ("今日 token：不限" if b["limit"] <= 0
                       else f"今日 token：{b['used']}/{b['limit']}")
        yield event.plain_result(
            f"群聊记忆体 v{PLUGIN_VERSION}\n"
            f"群聊套餐：{normalize_preset(self._cfg('group_preset', 'light'))}\n"
            f"群聊记忆：{n_mem} 个群\n"
            f"私聊会话：{n_priv} 人\n"
            f"你的日记：{diaries} 篇\n"
            f"{budget_line}\n"
            f"判读门：{'开' if self._cfg('reply_judge_enabled', True) else '关'}")

    @filter.command("日记")
    async def cmd_diary(self, event: AstrMessageEvent):
        """翻日记：/日记 最近 或 /日记 关键词 或 /日记 昨天"""
        uid = int(event.get_sender_id() or 0)
        if not event.is_private_chat():
            yield event.plain_result("日记是私聊的记忆，私聊我再看哦")
            return
        p = self._private(uid)
        arg = (event.message_str or "").strip()
        if not arg or arg in ("最近", "列表"):
            out = p.diary.recent(5)
        elif parse_time_query(arg):
            out = p.diary.by_time(arg)
        else:
            out = p.diary.search(arg)
        yield event.plain_result(out)
