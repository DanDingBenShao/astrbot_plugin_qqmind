"""诊断能力：环形日志缓冲 + 一键诊断包（纯内核，不依赖 AstrBot）。

设计约束：
- 环形 Handler 只抓本插件自己的日志（record.name 以 qqmind 前缀开头的
  logger，或消息带 [qqmind] 标记），上限 300 条，纯内存零文件。
- 诊断包四段：environment（版本/模型）/ config（配置快照）/ runtime
  （计数与状态）/ logs（脱敏摘录）。消息正文统一截断，隐私红线：
  日记正文绝不出包。
"""
from __future__ import annotations

import logging
import time
from collections import deque

PLUGIN_MARK = "[qqmind]"
MSG_CLIP = 20          # 日志摘录里消息正文保留长度
LOG_BUFFER_SIZE = 300

# 日志行里「消息正文」的截断规则：这些行带群友原话，出包前必须剪短
_CLIP_RULES = (
    ("judge ask:", 0),          # 判读输入（含上下文原话）——全剪
    ("judge ->", 24),           # 判读对象消息
    ("fail-closed:", 24),       # 垃圾判读输出
)


def clip_message(line: str) -> str:
    """按规则截断日志行里的消息正文。无命中规则的原样保留。"""
    for marker, keep in _CLIP_RULES:
        idx = line.find(marker)
        if idx >= 0:
            head = line[:idx + len(marker)]
            tail = line[idx + len(marker):]
            tail = tail.strip()
            if keep <= 0:
                return head
            return head + " " + tail[:keep] + ("…" if len(tail) > keep else "")
    return line


class RingBufferHandler(logging.Handler):
    """内存环形日志缓冲。attach 到插件的 logger 上即可。"""

    def __init__(self, capacity: int = LOG_BUFFER_SIZE):
        super().__init__(level=logging.DEBUG)
        self._buf: deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._buf.append({"ts": time.strftime("%m-%d %H:%M:%S",
                                             time.localtime(record.created)),
                          "level": record.levelname,
                          "line": clip_message(msg)})

    def snapshot(self) -> list[dict]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()


def attach(plugin_logger, capacity: int = LOG_BUFFER_SIZE) -> RingBufferHandler:
    """把环形缓冲挂到 logger 上（幂等：已有则先摘再挂）。返回 handler。"""
    for h in list(plugin_logger.handlers):
        if isinstance(h, RingBufferHandler):
            plugin_logger.removeHandler(h)
    h = RingBufferHandler(capacity)
    h.setFormatter(logging.Formatter("%(message)s"))
    plugin_logger.addHandler(h)
    return h


def detach(plugin_logger, handler: RingBufferHandler) -> None:
    plugin_logger.removeHandler(handler)


def build_diagnostics(*, plugin_version: str, astrbot_version: str,
                      platform: str, provider_model: str,
                      config: dict, stats: dict, runtime: dict,
                      log_lines: list[dict]) -> dict:
    """组装诊断包。config 会先过脱敏（只保留白名单键）。"""
    return {
        "diagnostic_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "plugin_version": plugin_version,
            "astrbot_version": astrbot_version,
            "platform": platform,
            "provider_model": provider_model,
        },
        "config": {k: config.get(k) for k in DIAG_CONFIG_KEYS},
        "runtime": runtime,
        "logs": log_lines,
    }


# 诊断包里允许出现的配置键（其余一律不带——防止用户私有内容如 persona_text 外泄）
DIAG_CONFIG_KEYS = (
    "group_preset", "read_mode", "judge_scope", "reply_judge_enabled",
    "judge_model", "think_model", "judge_fail_threshold",
    "reply_rate_per_min", "max_judged_per_min", "max_judged_per_round",
    "daily_token_budget", "batch_max_messages", "debounce_seconds",
    "followup_rounds", "roam_enabled", "memory_layering", "precise_memory",
    "precise_budget_tokens", "group_flat_messages", "group_shallow_messages",
    "group_context_messages", "group_buffer_size",
    "allowed_groups_count",
)
