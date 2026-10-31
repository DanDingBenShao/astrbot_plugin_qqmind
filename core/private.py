"""私聊会话：token 预算窗口 + 溢出压成日记（老板 2026-10-28 定案）。

规则：
- 窗口按 token 预算管理（默认 10000，可调），不按轮数——感情陪伴型可调大
- 预算吃紧时，把最老的一段对话交给 LLM 压成日记（标题+时间+标签+摘要）入库，
  原文从窗口移除——记忆不丢，只是换了存放形态
- 压缩失败不阻塞聊天：原文留下轮再试
- 一个模型走天下：压缩也走主模型（老板定案，不设专用小模型）
"""
from __future__ import annotations

import json
import time

from .core import est_tokens
from .diary import DiaryStore

# 压缩提示词：一个模型走天下，输出严格 JSON（title 供控制台列表显示）
COMPRESS_PROMPT = (
    "你是日记整理助手。把下面这段私聊对话压缩成一篇日记，帮对方长期记住这件事。\n"
    "只输出一行 JSON（不要代码块、不要解释）：\n"
    '{"title": "一句话标题，10字以内", "tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、有什么值得记住的'
    '（关键事实/约定/情绪/偏好），用第二人称「你」称呼对方", '
    '"content": "值得留原话的部分：直接引用1~3句对方说过的重要原话，'
    '没有值得留的就输出空字符串"}\n'
    "标签 2~5 个中文词。"
)


async def complete_text(llm, system: str, user: str,
                        max_tokens: int = 600, temperature: float = 0.3) -> str:
    """统一 LLM 调用：测试桩用 .complete，AstrBot Provider 用 .text_chat。"""
    out = await complete_text_resp(llm, system, user,
                                   max_tokens=max_tokens,
                                   temperature=temperature)
    return out[0]


async def complete_text_resp(llm, system: str, user: str,
                             max_tokens: int = 600,
                             temperature: float = 0.3) -> tuple[str, object]:
    """同 complete_text，但连 LLMResponse 一起返回（真实 usage 分桶记账用）。

    测试桩 .complete 路径没有 usage 对象时第二项为 None。
    """
    if hasattr(llm, "complete"):
        text = await llm.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature)
        return text, None
    out = await llm.text_chat(prompt=user, session_id=None,
                              system_prompt=system, contexts=[], image_urls=[])
    return out.completion_text or "", out


class PrivateSession:
    """单用户私聊记忆：token 预算窗口 + 日记库。

    messages 与群聊 UserMemory 同构（role/content），但管理层完全独立。
    """

    def __init__(self, diaries_dir, uid: int, budget: int = 10000,
                 keep_ratio: float = 0.34):
        self.uid = int(uid)
        self.budget = int(budget)              # 窗口 token 预算（可调）
        self.keep_ratio = keep_ratio           # 兼容旧配置（当前算法按预算切，不使用）
        self.messages: list[dict] = []         # [{"role","content","ts"}]
        self.diary = DiaryStore(diaries_dir, uid)
        self.compressing = False
        self._load_window()                    # v0.3.2：窗口随日记库持久化

    # ---------- 窗口持久化（v0.3.2） ----------
    def _load_window(self) -> None:
        """从该用户日记库的 meta 恢复最近窗口（插件重载/重启不再失忆）。"""
        raw = self.diary.get_meta("window")
        if not raw:
            return
        try:
            data = json.loads(raw)
        except Exception:
            return
        if not isinstance(data, list):
            return
        msgs: list[dict] = []
        for m in data:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            try:
                ts = float(m.get("ts") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            msgs.append({"role": role, "content": content, "ts": ts})
        self.messages = msgs[-1000:]           # 防御性上限；超预算由压缩收

    def save_window(self) -> None:
        """窗口写回日记库 meta：和日记同库同生命周期，隐私边界不变。"""
        try:
            self.diary.set_meta(
                "window", json.dumps(self.messages, ensure_ascii=False))
        except Exception as e:
            print("[private] save window error:", repr(e))

    def append_user_message(self, text: str, cap_tokens: int = 400) -> None:
        """用户消息入窗口。超长消息截断保存（头 2/3、尾 1/3），不再整条丢。

        v0.3.2：旧版超过 max_msg_tokens 直接不入窗口，长消息（告白/方案/长文）
        在上下文里彻底消失；现在至少保留大意和结尾。
        """
        text = (text or "").strip()
        if not text:
            return
        if est_tokens(text) > cap_tokens:
            head = max(200, cap_tokens * 2)
            tail = max(100, cap_tokens)
            if len(text) > head + tail:      # 真有中间段才省略，短消息原样留
                text = (text[:head] + "\n…（中间省略）…\n" + text[-tail:])
        self.messages.append({"role": "user", "content": text,
                              "ts": time.time()})

    # ---------- 窗口 token 账 ----------
    def window_tokens(self) -> int:
        return sum(est_tokens(m["content"]) for m in self.messages)

    def over_budget(self) -> bool:
        return self.window_tokens() > self.budget

    def oldest_slice(self, max_msgs: int = 30) -> list[dict]:
        """最老的一段对话（含首尾配对，避免把一轮劈成两半）。

        至少压走 4 条（2 轮）；多于 4 条时给窗口留 2 条尾巴，压走成对的整块。
        """
        total = len(self.messages)
        if total < 4:
            return []
        n = total if total <= 5 else min(total - 2, max_msgs)
        n -= n % 2                                   # 成对取整
        return self.messages[:n] if n >= 4 else []

    # ---------- 压缩 ----------
    @staticmethod
    def excerpt_for_compress(msgs: list[dict]) -> str:
        lines = []
        for m in msgs:
            who = "用户" if m["role"] == "user" else "AI"
            lines.append(f"{who}：{m['content']}")
        return "\n".join(lines)

    @staticmethod
    def derive_title(summary: str, limit: int = 14) -> str:
        """模型没给标题时的兜底：摘要第一句，截断到 limit 字。"""
        s = (summary or "").strip().split("\n")[0]
        for sep in "。！？!?；;，,":
            i = s.find(sep)
            if i > 0:
                s = s[:i]
                break
        s = s.strip(" ，,。.！!？?")
        return s[:limit] + ("…" if len(s) > limit else "")

    @staticmethod
    def parse_compress_output(out: str) -> tuple[str, list[str], str, str] | None:
        """解析压缩输出 JSON（容错：剥代码块围栏）。失败返回 None。

        返回 (title, tags, summary, content)；content 是第三层原文选段
        （旧模型不输出该字段时为空串，三层退化为两层）。
        title 缺失时由摘要兜底生成。
        """
        t = (out or "").strip()
        if t.startswith("```"):
            t = t.strip("`").lstrip("json").strip()
        try:
            d = json.loads(t)
            title = str(d.get("title") or "").strip()
            tags = [str(x) for x in (d.get("tags") or [])]
            summary = str(d.get("summary") or "").strip()
            content = str(d.get("content") or "").strip()
            if summary:
                return ((title or PrivateSession.derive_title(summary)),
                        tags, summary, content[:600])
        except Exception:
            pass
        return None

    async def compress_oldest(self, llm) -> int:
        """把最老一段压成日记入库并移出窗口。返回估算消耗 token（0=失败）。"""
        if self.compressing:
            return 0
        msgs = self.oldest_slice()
        if not msgs:
            return 0
        self.compressing = True
        try:
            ask = (f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。\n\n"
                   + self.excerpt_for_compress(msgs))
            out = await complete_text(llm, COMPRESS_PROMPT, ask,
                                      max_tokens=600, temperature=0.3)
            parsed = self.parse_compress_output(out)
            if not parsed:
                return 0        # 压缩失败：原文留下轮再试
            title, tags, summary, content = parsed
            first_ts = msgs[0].get("ts") or time.time()
            tok = sum(est_tokens(m["content"]) for m in msgs)
            self.diary.add(first_ts, tags, summary, tok, title=title,
                           content=content)
            del self.messages[:len(msgs)]
            self.save_window()
            return tok + 200
        except Exception as e:
            print("[private] compress error:", repr(e))
            return 0
        finally:
            self.compressing = False
