"""记忆内核 · 纯逻辑层。

规则（与生产实现 qq_chat.py 一致，2026-09-21 v3 / 09-22 判读双字段）：
- 浅层（读）：判读只看「最近 7 条群聊 + 当前消息」，输出三态：是|思考 / 是|直接 / 否
- 深层（回）：每用户逐字窗口 110 轮；溢出一次弹 10 轮交 LLM 蒸馏进画像；
  窗口内始终 ≥100 轮逐字上下文；单条超 max_msg_tokens 不入历史
- 回捞：模型输出 <查历史"N"> / <查群聊"N"> / <查图>，框架执行后结果回喂（最多 2 轮）

本模块不 import 网络库、不 import 平台协议——给内核喂数据、拿判决，收发是外壳的事。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# ---------- 判读提示词（三态双字段） ----------
# v3（2026-10-29 瘦身实测）：v1 660字/585tok → v3 271字/241tok，真实 API 30 条
# 混合用例正确率 24/29 vs 20/29（v3 反超），成本省 40%。示例段是质量锚，必须保留。
JUDGE_PROMPT = (
    "你是群聊意图分类器。AI助手在群里（可能被叫名字/@）。"
    "提示里若给出「AI助手在本群的名字是X」，消息里叫X就是在叫它。\n"
    "判【否】：在和别人说话、接别人的话、自言自语、对全群喊话。\n"
    "判【是】：叫它/@它/问它请求它，或顺着AI刚说的话继续。"
    "其中要创作（写诗/讲故事/出主意/起名/画图）或要推理（数学/谜题）→是|思考；"
    "打招呼闲聊简单问答→是|直接；拿不准→是|直接。\n"
    "示例（AI本群叫「小桃」）：\n"
    "「小桃，帮我想个周末去哪玩的点子」→是|思考；「@小桃 在吗」→是|直接；"
    "「今天天气不错」→否。\n"
    "只输出一行，三选一：是|思考、是|直接、否。"
)

# ---------- 回复清洗 ----------
_DELIM_RE = re.compile(r"\]?<\]?\s*minimax\s*\[?>\[?")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2B00-\u2BFF\uFE0F"
                       "\u2600-\u2660\u2662-\u2664\u2666-\u27BF]+")

# ---------- 画图标记与回捞命令 ----------
_DRAW_RE = re.compile(r'<生图[“"]([^”"]{2,2000})[”"](?:\s*@\s*([^"”>]{1,12}))?\s*>')
_PROMISE_RE = re.compile(r"(给你画|这就画|马上画|我来画|帮你画|画一张|画一只|画一个|画好啦|画好了)")
_CMD_RE = re.compile(r'<(查历史|查群聊)"(\d{1,3})">')
_CMD_PIC_RE = re.compile(r"<查图>")

MAX_JUDGED_PER_ROUND = 12  # 监控期单轮判读上限，防刷屏烧 token
EXCERPT_MAX_CHARS = 120    # 回捞摘录每条截断长度


def sanitize_reply(text: str) -> str:
    """清理模型输出：分段符转换行；剥 emoji（人格规则禁用，模型偶尔不守）。颜文字不受影响。"""
    t = _DELIM_RE.sub("\n", text or "")
    return _EMOJI_RE.sub("", t).strip()


def thinking_model(model: str) -> str:
    """nothink 别名 → 同款开思考模型（判读判定要动脑的那一轮临时切换）"""
    m = str(model or "").strip()
    return m[:-len("-nothink")] if m.lower().endswith("-nothink") else m


def parse_draw(reply: str) -> tuple[str, str, str]:
    """提取画图标记。返回 (prompt, model, 去掉标记后的纯文本)；无标记时前两项为空。"""
    hits = _DRAW_RE.findall(reply or "")
    text = _DRAW_RE.sub("", reply or "").strip()
    if not hits:
        return "", "", text
    prompt, model = hits[0]
    return prompt.strip(), model.strip(), text


def promised_draw(reply: str) -> bool:
    """嘴上答应画图却没有标记的典型话术 → 触发一次补标记追问。"""
    return bool(_PROMISE_RE.search(reply or ""))


def est_tokens(text: str) -> int:
    """粗估 token：CJK 1:1，其余 4:1。够做入史门槛与账本，不追求精确。"""
    n = 0
    for c in text or "":
        n += 1 if ord(c) > 0x2E80 else 0.25
    return int(n)


def parse_judge_verdict(out: str) -> tuple[bool, bool] | None:
    """判读输出 → (是否对AI说, 是否需要思考)。

    严格三态：只接受 是|思考 / 是|直接 / 否（允许全角竖线、裸「是」/「否」）。
    任何其他输出（空、含解释、「我不确定」…）返回 None —— 调用方必须 fail-closed
    （宁漏接不插嘴），并把这次计为判读失败。
    """
    t = re.sub(r"[\s.。，,！!？?~～]", "", out or "")
    t = t.replace("｜", "|").strip("|")
    if t in ("是|思考", "是思考"):
        return True, True
    if t in ("是|直接", "是直接", "是"):
        return True, False
    if t == "否":
        return False, False
    return None


# ---------- 深层记忆：窗口与画像 ----------
class UserMemory:
    """单用户深层记忆：逐字窗口 + 蒸馏画像，JSON 落盘（原子写）。

    与生产实现一致：窗口上限 = history_rounds*2 条（一轮=用户一条+AI一条），
    溢出后一次弹 batch_pop 条（10 轮），弹出块交上层蒸馏。
    """

    def __init__(self, store_dir: str | os.PathLike, uid: int | str):
        self.uid = int(uid)
        self.path = Path(store_dir) / f"{self.uid}.json"
        self.profile = ""
        self.messages: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and isinstance(d.get("messages"), list):
                self.profile = d.get("profile") or ""
                self.messages = d["messages"]
                return
        except Exception:
            pass
        self.profile, self.messages = "", []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"profile": self.profile, "messages": self.messages},
                      f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---------- 窗口 ----------
    def append_round(self, user_text: str, assistant_text: str,
                     history_rounds: int = 110) -> list[dict]:
        """写入一轮（用户一条+AI一条），返回溢出的整块（交给蒸馏），无溢出返回 []。"""
        batch_pop = 20   # 溢出后一次弹 10 轮，保证窗口内始终 ≥100 轮逐字上下文
        limit = int(history_rounds) * 2
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        if assistant_text:
            self.messages.append({"role": "assistant", "content": assistant_text})
        evicted: list[dict] = []
        while len(self.messages) > limit + batch_pop:
            evicted.append(self.messages.pop(0))
        return evicted

    @staticmethod
    def distill_prompt(current_profile: str, evicted: list[dict]) -> tuple[str, str]:
        """溢出块的蒸馏提示词。返回 (system, user)。"""
        excerpt = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in evicted)
        user = (f"当前已知画像与记忆：\n{current_profile or '（空）'}\n\n"
                f"以下是被滚动出历史窗口的对话摘录：\n{excerpt}\n\n"
                "请输出更新后的画像与重要记忆（对象画像、关键事实、约定、偏好），"
                "条目化、紧凑，不要评论，不要寒暄。")
        return "你负责维护用户画像与长期记忆。", user

    # ---------- 回捞摘录 ----------
    def history_excerpt(self, n: int) -> str:
        msgs = self.messages
        if not msgs:
            return "（这位用户还没有更早的聊天历史）"
        part = msgs[-max(1, min(int(n), 50)):]
        lines = [f"{'用户' if m['role'] == 'user' else '我'}：{m['content'][:EXCERPT_MAX_CHARS]}"
                 for m in part]
        return f"（这位用户更早的聊天·最近{len(part)}条）\n" + "\n".join(lines)


class GroupBuffer:
    """群上下文滚动缓冲（判读与轻量档语境的唯一来源）。

    调用方必须写入双方：群友消息（on_message）与机器人自己的回复
    （on_llm_response / _reply_group），否则轻量档机器人看不到自己说过什么。
    重启清零（内存态，可接受）。判读取最近 7 条，轻量档语境取 20 条。
    """

    def __init__(self, max_len: int = 60):
        self.max_len = int(max_len)
        self.buf: dict[int, list[str]] = {}

    def note(self, gid: int, nick: str, text: str) -> None:
        text = (text or "").strip()
        if not gid or not text:
            return
        b = self.buf.setdefault(int(gid), [])
        b.append(f"{nick}：{text.replace(chr(10), '；')}")
        if len(b) > self.max_len:
            del b[:len(b) - self.max_len]

    def recent(self, gid: int, n: int = 7, exclude_last: bool = False) -> list[str]:
        b = list(self.buf.get(int(gid)) or [])
        if exclude_last:
            b = b[:-1]
        return b[-max(1, int(n)):]

    def excerpt(self, gid: int, n: int) -> str:
        b = (self.buf.get(int(gid)) or [])[-max(1, min(int(n), 50)):]
        if not b:
            return "（群聊缓冲里没有更早的消息：只记录最近一部分，且重启后清零）"
        return f"（群里更早的消息·最近{len(b)}条）\n" + "\n".join(b)


# ---------- 发送分段（手机友好） ----------
_PHONE_COLS = 34   # 手机一行约 16~17 个全角字符的显示宽度


def _disp_cols(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _is_kao(c: str) -> bool:
    """颜文字构件字符：符号类（排除汉字/字母数字/常用中文标点/空白）"""
    if not c or c.isspace():
        return False
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF or (c.isascii() and c.isalnum()):
        return False
    if c in "，。！？、；：…—·「」『』《》":
        return False
    return o >= 0x80 or c in "()[]{}\\/|^~=<>+-*_&%$#@!\"'`"


def merge_broken_kaomoji(bubbles: list[str]) -> list[str]:
    """换行切进颜文字中间（上一行括号未闭合）时把两行接回去"""
    if len(bubbles) < 2:
        return bubbles
    out = [bubbles[0]]
    for ln in bubbles[1:]:
        depth = 0
        for ch in out[-1]:
            if ch in "（([":
                depth += 1
            elif ch in "）)]":
                depth = max(0, depth - 1)
        if depth > 0:
            out[-1] = out[-1] + ln
        else:
            out.append(ln)
    return out


def wrap_safe(line: str, max_cols: int = _PHONE_COLS) -> list[str]:
    """超过手机一行宽度时折行：断点避开颜文字内部，优先落在标点/空格上"""
    if _disp_cols(line) <= max_cols:
        return [line]
    pieces, rest = [], line
    while _disp_cols(rest) > max_cols:
        cut, col = 0, 0
        for i, ch in enumerate(rest):
            w = 2 if ord(ch) > 0x2E80 else 1
            if col + w > max_cols:
                cut = i
                break
            col += w
        if cut <= 0:
            break
        best = 0
        for j in range(cut, 0, -1):
            if _is_kao(rest[j - 1]) and _is_kao(rest[j]):
                continue          # 两侧都是颜文字构件 → 在颜文字内部，禁断
            best = j
            if rest[j - 1] in "，。！？、；：.!?~～ ":
                break             # 标点/空格断点更优先
        if best <= 0:
            break
        pieces.append(rest[:best])
        rest = rest[best:]
    if rest:
        pieces.append(rest)
    return pieces or [line]


def split_bubbles(reply: str, max_bubbles: int = 5) -> list[str]:
    """回复 → 手机风格多条短消息：按换行拆、接回被切的颜文字、超宽折行、超长合并尾部。"""
    raw = [b.strip() for b in (reply or "").split("\n") if b.strip()]
    raw = merge_broken_kaomoji(raw)
    bubbles: list[str] = []
    for b in raw:
        bubbles.extend(wrap_safe(b))
    if not bubbles:
        bubbles = ["……"]
    if len(bubbles) > max_bubbles:
        bubbles = bubbles[:max_bubbles - 1] + ["\n".join(bubbles[max_bubbles - 1:])]
    return bubbles


# ---------- 监控期会话（浅层状态机，纯内存） ----------
class MonitorSession:
    """@ 触发的监控期：无需再 @ 的后续消息经判读门接话。

    额度规则（与生产一致）：判是回复后额度刷新；判否扣 1 格；扣满闭嘴；
    空闲超时自动关；单轮判读有预算上限（防刷屏烧 token）。
    """

    def __init__(self, followup_rounds: int = 3, idle_timeout_min: float = 5):
        self.monitoring = True
        self.rounds_left = int(followup_rounds)
        self.buf: list[tuple[str, int, bool, bool]] = []  # (text, message_id, persist, think)
        self.judged = 0
        self.answering = False
        self.last_active = time.time()
        self.idle_timeout = float(idle_timeout_min) * 60

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) - self.last_active > self.idle_timeout
