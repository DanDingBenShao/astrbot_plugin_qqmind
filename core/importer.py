"""历史数据一键导入（老板 2026-10-30 定案）。

场景：从 DSH/桥接方案迁到 AstrBot 的用户，聊天记录散在
conversations 表 / DSH 导出 JSON 里，换框架记忆全丢。

设计：
- 来源无关：入口统一收敛成 [{role, content, ts?}] 会话数组，
  conversations.content / DSH 导出 / 通用 chat JSON 都能适配
- 批量压缩：按 token 预算切块，逐块走压缩管线进三层日记
  （与在线压缩同一套 prompt/解析，模型没有第三层字段就自动退两层）
- 画像提取：全量记录一遍过，提炼「这个用户是谁」写进画像
  （persona_slots 机制），喂回日后的 system prompt
- 后台任务：导入可能跑几分钟，asyncio.create_task + 进度 dict，
  控制台轮询（不阻塞 WebUI）
- 幂等护栏：同会话重复导入跳过（按会话指纹记录）
"""
from __future__ import annotations

import json
import time

from .core import est_tokens
from .private import PrivateSession, complete_text

# 压缩切块：一次喂给压缩模型的对话预算（和在线压缩 30 条量级对齐）
CHUNK_TOKENS = 2000
# 每块最多条数兜底（防止单条超长消息把块撑爆）
CHUNK_MAX_MSGS = 40

PROFILE_PROMPT = (
    "你是用户画像整理助手。根据下面这个人跟 AI 助手的历史聊天记录，"
    "提炼这个用户是谁。只输出一行 JSON（不要代码块）：\n"
    '{"nickname": "怎么称呼TA（从记录里TA的自我称呼/聊天风格推断，不确定就空串）", '
    '"traits": ["性格特点", "说话风格"], '
    '"interests": ["常聊的话题/爱好"], '
    '"facts": ["重要事实：职业/状态/约定/在意的人事"], '
    '"avoid": ["TA忌讳或反感的事（没有就空数组）"]}\n'
    "各列表 0~5 项，只写记录里有依据的，别脑补。"
)


def normalize_messages(raw: object) -> list[dict]:
    """把各来源的历史记录统一成 [{role, content, ts?}]。

    支持：
    - conversations.content 风格：role/content，content 可能是 str 或
      [{type:"text", text:"..."}] 段数组
    - DSH 导出风格：{"messages": [...]} 或直接数组
    - 无法识别的行丢弃（导入宁缺毋滥）
    """
    if isinstance(raw, dict):
        raw = raw.get("messages") or raw.get("conversation") or raw.get("data")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or "").strip()
        if role in ("human", "user"):
            role = "user"
        elif role in ("assistant", "bot", "gpt", "ai"):
            role = "assistant"
        else:
            continue
        c = m.get("content")
        if isinstance(c, list):
            texts = [seg.get("text", "") for seg in c
                     if isinstance(seg, dict) and seg.get("type") == "text"
                     and not str(seg.get("text", "")).startswith("<system_reminder")]
            content = "\n".join(t for t in texts if t)
        else:
            content = str(c or "")
        content = content.strip()
        if not content or content.startswith("<system_reminder"):
            continue
        try:
            ts = float(m.get("ts") or m.get("timestamp") or 0) or None
        except (TypeError, ValueError):
            ts = None
        if ts and ts > 1e12:   # 毫秒时间戳归一
            ts /= 1000.0
        out.append({"role": role, "content": content, "ts": ts})
    return out


def chunk_messages(msgs: list[dict],
                   chunk_tokens: int = CHUNK_TOKENS,
                   max_msgs: int = CHUNK_MAX_MSGS) -> list[list[dict]]:
    """按 token 预算把会话切块（保持时间顺序，块内不再重排）。"""
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_tok = 0
    for m in msgs:
        t = est_tokens(m["content"])
        if cur and (cur_tok + t > chunk_tokens or len(cur) >= max_msgs):
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(m)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def _excerpt(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        who = "用户" if m["role"] == "user" else "AI"
        lines.append(f"{who}：{m['content']}")
    return "\n".join(lines)


class ImportJob:
    """一次导入任务：进度可查，跑完落统计。插件持有 dict[import_id, job]。"""

    def __init__(self, uid: int, msgs: list[dict], total_chunks: int):
        self.uid = int(uid)
        self.msgs = msgs
        self.total_chunks = total_chunks
        self.done_chunks = 0
        self.entries = 0
        self.failed_chunks = 0
        self.state = "running"     # running / extracting / done / failed
        self.error = ""
        self.profile: dict | None = None   # 画像提取结果（done 后可读）
        self.started = time.time()
        self.finished = 0.0

    def progress(self) -> dict:
        return {
            "uid": self.uid, "state": self.state, "error": self.error,
            "chunks_done": self.done_chunks, "chunks_total": self.total_chunks,
            "entries": self.entries, "failed_chunks": self.failed_chunks,
            "profile": self.profile,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


async def run_import(job: ImportJob, diary, llm) -> None:
    """后台执行：逐块压缩进日记，再跑一遍画像提取。

    diary: DiaryStore（已按 uid 打开）；llm: provider（text_chat/complete）。
    失败块跳过继续（导入是尽力而为，中断的原文不丢——源数据还在）。
    """
    try:
        for chunk in chunk_messages(job.msgs):
            ask = (f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。\n\n"
                   + _excerpt(chunk))
            out = await complete_text(llm, _compress_prompt(),
                                      ask, max_tokens=600, temperature=0.3)
            parsed = PrivateSession.parse_compress_output(out)
            if parsed:
                title, tags, summary, content = parsed
                first_ts = next((m["ts"] for m in chunk if m.get("ts")),
                                time.time())
                tok = sum(est_tokens(m["content"]) for m in chunk)
                diary.add(first_ts, tags + ["导入"], summary, tok,
                          title=title, content=content)
                job.entries += 1
            else:
                job.failed_chunks += 1
            job.done_chunks += 1
        # 画像提取：全量记录一遍过（截断防超长）
        job.state = "extracting"
        profile = await extract_profile(llm, job.msgs)
        if profile:
            job.profile = profile
        job.state = "done"
    except Exception as e:     # noqa: BLE001 后台任务吞异常记状态，不炸插件
        job.state = "failed"
        job.error = repr(e)
    finally:
        job.finished = time.time()


def _compress_prompt() -> str:
    # 与 private.COMPRESS_PROMPT 保持同一契约（避免循环依赖直接 import）
    from .private import COMPRESS_PROMPT
    return COMPRESS_PROMPT


async def extract_profile(llm, msgs: list[dict]) -> dict | None:
    """全量记录 → 用户画像 JSON。失败返回 None（导入照常完成）。"""
    text = _excerpt(msgs)
    if len(text) > 24000:      # 画像提取看大意，超长截断
        text = text[:12000] + "\n…（中段截断）…\n" + text[-12000:]
    try:
        out = await complete_text(llm, PROFILE_PROMPT,
                                  f"历史聊天记录：\n{text}",
                                  max_tokens=500, temperature=0.3)
        t = (out or "").strip()
        if t.startswith("```"):
            t = t.strip("`").lstrip("json").strip()
        d = json.loads(t)
        if not isinstance(d, dict):
            return None
        return {
            "nickname": str(d.get("nickname") or "")[:40],
            "traits": [str(x)[:60] for x in (d.get("traits") or [])][:5],
            "interests": [str(x)[:60] for x in (d.get("interests") or [])][:5],
            "facts": [str(x)[:120] for x in (d.get("facts") or [])][:5],
            "avoid": [str(x)[:60] for x in (d.get("avoid") or [])][:5],
        }
    except Exception:
        return None
