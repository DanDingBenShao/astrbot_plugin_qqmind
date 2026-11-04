"""判读门（C2 解耦自 main.py，v0.3.9）。

on_message / _read_full / _judge / _bot_group_name(_lazy) 五个方法的宿主，
以 mixin 形式被 main.QQMindPlugin 继承——`self` 即插件实例，状态、配置、
装饰器挂接方式与解耦前完全一致。

拆出边界即未来散装包 qqmind-token 的边界：本模块只依赖
- 插件宿主的状态：group_buffer/_runtime/sessions/_degraded/_judge_fail/
  _bot_names/_affection/_STATS
- core/ 纯内核（判读提示词、唤醒检测、分泡）
- astrbot.api 的 logger / filter
"""
from __future__ import annotations

import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter

from ..core import (MAX_JUDGED_PER_ROUND, JUDGE_PROMPT,
                    MonitorSession, batch_should_flush, detect_wake,
                    est_tokens, keyword_wake_enabled, normalize_preset,
                    parse_judge_verdict)
from ..main_helpers import _is_poke_self, _quote_reply_prefix, _safe_int_id

JUDGE_PROMPT_LEN = est_tokens(JUDGE_PROMPT)   # 判读 system 提示词的 token 数（账单口径）


class JudgeGateMixin:
    """群消息判读门：省 token 主线。判否只拦主模型，不终结事件链。

    注意：本 mixin 的 on_message 不挂装饰器——main.py 里留 6 行转发壳
    （on_message -> on_message_impl），装饰器必须在插件主模块上才能通过
    AstrBot 的 handler_module_path 过滤（见 main.py 挂点 1 注释）。"""

    async def on_message_impl(self, event: AstrMessageEvent):
        """群消息判读门 + 私聊直通。判否 should_call_llm(True)：
        只拦主模型，不终结事件链——其他插件和命令照常处理。"""
        # 判读门只管群聊；私聊直接放行给挂点 2/3 处理
        if event.is_private_chat():
            try:
                self._last_platform = event.get_platform_name() or self._last_platform
            except Exception:
                pass
            puid = _safe_int_id(event.get_sender_id())
            if puid:
                # 记住最近一次私聊 origin（主动私信的发送地址）；
                # 用户回了话 → 主动私信未回应计数清零
                self._runtime[("private", puid)] = {
                    "origin": event.unified_msg_origin,
                    "last_msg": time.time()}
                self._proactive_reply_watch(puid)
            return
        gid = _safe_int_id(event.get_group_id())
        uid = _safe_int_id(event.get_sender_id())
        try:
            self._last_platform = event.get_platform_name() or self._last_platform
        except Exception:
            pass
        self_id = str(event.get_self_id() or "")
        if not gid or not uid or str(uid) == self_id:
            return
        key = (gid, uid)
        if self._is_bot(uid, event):
            # bot 消息（防互怼）：qqmind 不理，也不许主模型理
            event.should_call_llm(True)
            return
        allowed = self._cfg("allowed_groups") or []
        if allowed and str(gid) not in [str(g) for g in allowed]:
            # 白名单外：qqmind 完全隐身，既不判读也不碰任何标志
            return
        switch = self._group_switch(gid)
        if switch == "off":
            # 群开关=关：不读不回，完全隐身（与白名单外同路径）
            return
        if switch == "at_only":
            # 仅@模式：只有 @我/引用我/戳我 才走后续；其余消息不读不判
            # （戳自己另有独立路径，在上面已分流，这里不会误伤）
            wake_probe = detect_wake((event.message_str or "").strip(),
                                     event.get_messages() or [], self_id, [])
            if not wake_probe.at_me:
                return
            # @/引用我：必回，开监控期直接放行主模型（与 wake.at_me 主路径一致）
            self._degraded.discard(gid)
            self._judge_fail.pop(gid, None)
            nick = (event.get_sender_name() or "").strip() or str(uid)
            text = await self._vision_annotate(
                event, (event.message_str or "").strip())
            self._note_group(gid, f"{nick}({uid})", text or "[图片]")
            rt = self._rt(gid)
            rt["origin"] = event.unified_msg_origin
            rt["bot"] = getattr(event, "bot", None)
            rt["self_id"] = self_id
            rt["last_msg"] = time.time()
            self._record_incoming(gid, uid, f"{nick}({uid})", text)
            self._cancel_batch(gid)
            s = MonitorSession(followup_rounds=int(self._cfg("followup_rounds", 3)),
                               idle_timeout_min=5)
            self.sessions[key] = s
            if self._cfg("affinity_enabled", False):
                self._affection.add(uid, 2)
            logger.info(f"[qqmind] at_only wake for {uid} in group {gid}")
            return
        # 戳一戳（v0.2.12）：Poke 组件 target_id=自己 → 独立反应路径
        if _is_poke_self(event.get_messages() or [], self_id):
            event.should_call_llm(True)   # 不让主模型接管
            # 先落 origin/bot/self_id（_on_poked 发送要用；notice 事件
            # 在 normal 流程里也要走这一步，这里提前补齐）
            rt = self._rt(gid)
            rt["origin"] = event.unified_msg_origin
            rt["bot"] = getattr(event, "bot", None)
            rt["self_id"] = self_id
            await self._on_poked(gid, uid, event)
            return
        nick = (event.get_sender_name() or "").strip() or str(uid)
        text = (event.message_str or "").strip()
        # 通知类事件拦截（v0.2.15 审查修）：aiocqhttp 把戳别人/进退群/撤回等
        # notice 也当 GROUP_MESSAGE 派发（message_str 恒空、无引用无图）——
        # 不拦会每条通知写一行「某人：[图片]」假消息污染群缓冲与判读语境。
        # 非戳自己的 Poke（互戳）同样静默。判据两层：OneBot post_type=notice
        # 优先（真事件），fallback 看无文本无引用无图（假事件/其他适配器）。
        raw = getattr(event.message_obj, "raw_message", None)
        is_notice = False
        try:
            is_notice = (getattr(raw, "post_type", None)
                         or (raw.get("post_type") if hasattr(raw, "get") else None)
                         or "") == "notice"
        except Exception:
            is_notice = False
        if is_notice or (
                not text
                and not _quote_reply_prefix(event.get_messages() or [])
                and not any(type(s).__name__ == "Image"
                            for s in (event.get_messages() or []))):
            event.should_call_llm(True)
            return
        preset = normalize_preset(self._cfg("group_preset", "light"))
        keywords = (self._cfg("wake_keywords", [])
                    if keyword_wake_enabled(preset) else [])
        wake = detect_wake(text, event.get_messages() or [], self_id, keywords)
        text = await self._vision_annotate(event, text)
        # 引用展开（v0.2.11）：Reply 段的「谁说了什么」拼进正文——适配器把
        # 引用内容放在 Reply 组件里，message_str 只有纯文本段，不展开的话
        # 判读和主模型都看不见被引用的那句话（2026-10-30 群内实测半盲）。
        quote = _quote_reply_prefix(event.get_messages() or [])
        if quote:
            text = f"{quote}{text}".strip()
        self._note_group(gid, f"{nick}({uid})", text or "[图片]")
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

        if wake.at_me:
            # @/引用我 必回：丢掉待回复批，开监控期并直接放行给主模型
            self._degraded.discard(gid)
            self._judge_fail.pop(gid, None)
            self._record_incoming(gid, uid, f"{nick}({uid})", text)
            self._cancel_batch(gid)
            s = MonitorSession(followup_rounds=int(self._cfg("followup_rounds", 3)),
                               idle_timeout_min=5)
            self.sessions[key] = s
            if self._cfg("affinity_enabled", False):
                self._affection.add(uid, 2)
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
        self._record_incoming(gid, uid, f"{nick}({uid})", text)
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
                rt["batch"].append({"uid": uid,
                                    "nick": f"{nick}({uid})", "text": text})
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
        s = self.sessions.get(key)
        if not s or not s.monitoring:
            event.should_call_llm(True)
            return
        if not ok:
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
        if self._cfg("affinity_enabled", False):
            self._affection.add(uid, 1)
        s.touch()   # 额度在 on_llm_response 刷新

    async def _read_full(self, event: AstrMessageEvent, gid: int, uid: int,
                         nick: str, text: str) -> None:
        """全量读：每条消息都过判读门（判是才放行原生管线）。"""
        if not text:
            event.should_call_llm(True)
            return
        if not self._cfg("reply_judge_enabled", True):
            if not self._allow_reply(gid):
                self._stat("dropped_rate", gid)
                event.should_call_llm(True)
                return
            event.is_at_or_wake_command = True
            event.is_wake = True
            return
        if not self._allow_judge(gid):
            self._stat("dropped_rate", gid)
            event.should_call_llm(True)
            return
        bot_name = await self._bot_group_name(event, gid)
        ok, think = await self._judge(gid, nick, text, bot_name)
        if not ok:
            event.should_call_llm(True)
            return
        if not self._allow_reply(gid):
            self._stat("dropped_rate", gid)
            event.should_call_llm(True)
            return
        event.is_at_or_wake_command = True
        event.is_wake = True
        setattr(event, "qqmind_think", bool(think))   # 思考模式由判读控制
        if self._cfg("affinity_enabled", False):
            self._affection.add(uid, 1)

    async def _judge(self, gid: int, nick: str, text: str,
                     bot_name: str = "", batch=None) -> tuple[bool, bool]:
        """浅层判读。batch 非空时整批判一次；出错 fail-closed。"""
        if self._budget_exhausted():
            return False, False
        provider = await self._aux_provider("judge")
        if provider is None:
            return False, False
        k = len(batch) if batch else 1
        # 判读语境条数可控（v0.3.5 账单实测：7 条语境占单次输入 ~40%，
        # 群越活跃缓存越不命中；98.6% 的判读结论是「否」，为「不用回」带
        # 7 条原文纯属烧钱）。默认 0=只判待判消息本身+群名片锚点；
        # 想恢复旧行为填 7。判读语境从 group_buffer 出（写侧在 on_message）。
        # 待判消息与缓冲同格式（nick(uid)：内容）。注意 on_message 入队时
        # nick 已含 (uid)，这里不再包一层（v0.2.15 审查修：否则变成
        # 张三(111)(111)，判读门认不出同一个人）
        # 单条超长截断（v0.3.5）：判读只需判断「是否在对 AI 说话」，
        # 4000 字原文整段进提示词纯烧钱，与群缓冲 max_msg_tokens 同口径
        _cap_n = max(16, int(self._cfg("max_msg_tokens", 100))) * 2

        def _cap(s: str) -> str:
            return s if len(s) <= _cap_n else s[:_cap_n].rstrip() + "…"

        pending = ("\n".join(f"{m['nick']}: {_cap(m['text'])}" for m in batch)
                   if batch else f"{nick}：{_cap(text)}")
        question = ("以上这些消息是在对AI助手说话吗？若在，需要AI深度思考吗？"
                    if batch else
                    "这条消息是在对AI助手说话吗？若在，需要AI深度思考吗？")
        who = f"AI助手在本群的名字是「{bot_name}」。\n" if bot_name else ""
        n_jctx = int(self._cfg("judge_context_messages", 0))
        ctx_lines = []
        if n_jctx > 0:
            ctx_all = self.group_buffer.recent(gid, n_jctx + k)
            ctx_lines = ctx_all[:-k][-n_jctx:] if len(ctx_all) >= k else []
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
        # real 在 count 之前（bucket 随 count 的落盘一起写盘）
        self._real_usage(out, "judge")
        self._count_usage(ask, out_text, resp=out, gid=int(gid))
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
        self._stat("judge_think", gid, int(think))
        # 账单唯一计账点（v0.2.3）：读/拦只在判读出口记，三条调用路径
        # （全量/监控期/攒批）自动统一——攒批此前整条路径漏计，账单「读」严重偏小。
        # 读=k 条（整批判读看了 k 条）；拦=1 次回复决策（整批判否也只省一次回复）
        self._STATS["judge_in"] += est_tokens(ask) + JUDGE_PROMPT_LEN
        self._STATS["judge_out"] += est_tokens(out_text)
        self._stat("judged", gid, k)
        if not ok:
            self._stat("blocked", gid)
            self._bill_saved(1)
        logger.info(f"[qqmind] judge -> {'是|思考' if think else ('是|直接' if ok else '否')}: {pending[:40]}")
        return ok, think

    async def _bot_group_name(self, event: AstrMessageEvent, gid: int) -> str:
        """机器人在本群的群名片（判读提示用，识别「叫的是我」）。每群缓存一次。"""
        if gid in self._bot_names:
            return self._bot_names[gid]
        name = ""
        me = str(event.get_self_id() or "")
        bot = getattr(event, "bot", None)
        me_i, gid_i = _safe_int_id(me), _safe_int_id(gid)
        if bot is not None and me_i and gid_i:
            try:
                info = await bot.call_action(
                    "get_group_member_info", self_id=me_i,
                    group_id=gid_i, user_id=me_i)
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

    async def _bot_group_name_lazy(self, gid: int) -> str:
        """攒批/随机读路径的群名片：没有事件对象，从运行时存的 bot/self_id
        懒查一次并缓存。查不到缓存空值，不会每批都打 API。"""
        if gid in self._bot_names:
            return self._bot_names[gid]
        rt = self._rt(gid)
        me = str(rt.get("self_id") or "")
        bot = rt.get("bot")
        if not me or bot is None or not hasattr(bot, "call_action"):
            return ""
        name = ""
        me_i, gid_i = _safe_int_id(me), _safe_int_id(gid)
        try:
            info = await bot.call_action(
                "get_group_member_info", self_id=me_i,
                group_id=gid_i, user_id=me_i)
            if isinstance(info, dict):
                name = (info.get("card") or info.get("nickname") or "").strip()
        except Exception as e:
            logger.debug(f"[qqmind] lazy member info lookup failed: {e!r}")
        self._bot_names[gid] = name
        return name
