"""群聊四轴 · 唤醒轴与套餐档位（与平台解耦，纯逻辑）。

套餐（老板 2026-10-28 定案）：
- 轻量 light  ：只 @ 读取、@ 必回，被动消息完全不读
- 全量 full   ：每条都读，判读门决定回不回
- 分层 layered：攒批读，记忆分浅层/深层
- 高级 advanced：四轴全部拆开由用户自选

唤醒条件（老板定案）：
- @机器人QQ号 → 必回（硬规则）
- @全体成员 → 不唤醒（忽略）
- 文本关键词 → 开门 + 过判读（中文无词边界，靠判读过滤「小桃」命中「小桃子」）
"""
from __future__ import annotations

import time
from dataclasses import dataclass

PRESETS = ("light", "full", "layered", "advanced")
DEFAULT_PRESET = "light"

READ_MODES = ("at_only", "full", "batch", "random")
DEFAULT_READ_MODE = "batch"

# 套餐隐含的读取方式（advanced 为 None：完全按用户配置）
_PRESET_READ = {
    "light": "at_only",
    "full": "full",
    "layered": "batch",
    "advanced": None,
}
# 套餐隐含的记忆方式（advanced 为 None：按开关组合）
_PRESET_MEMORY = {
    "light": "flat",
    "full": "flat",
    "layered": "layered",
    "advanced": None,
}


def normalize_preset(value) -> str:
    """配置里的套餐名容错：不认识的一律回落轻量。"""
    v = str(value or "").strip().lower()
    return v if v in PRESETS else DEFAULT_PRESET


def preset_read_mode(preset: str) -> str | None:
    return _PRESET_READ[normalize_preset(preset)]


def preset_memory_mode(preset: str) -> str | None:
    return _PRESET_MEMORY[normalize_preset(preset)]


def keyword_wake_enabled(preset: str) -> bool:
    """轻量档按定义只认 @；其余档位开启文本关键词唤醒。"""
    return normalize_preset(preset) != "light"


def resolve_read_mode(preset: str, configured=None) -> str:
    """套餐隐含的读取方式优先；高级档用用户配置（非法值回落攒批）。"""
    mode = preset_read_mode(preset)
    if mode:
        return mode
    m = str(configured or "").strip().lower()
    return m if m in READ_MODES else DEFAULT_READ_MODE


def keyword_hit(text: str, keywords=None) -> bool:
    """中文子串匹配（不区分大小写）；空关键词跳过。"""
    body = (text or "").casefold()
    if not body:
        return False
    for kw in keywords or []:
        k = str(kw or "").strip().casefold()
        if k and k in body:
            return True
    return False


@dataclass(frozen=True)
class WakeHit:
    at_me: bool = False       # 明确 @ 了机器人（含引用回复机器人）
    at_all: bool = False      # @全体成员（不唤醒）
    keyword: bool = False     # 命中文本关键词

    @property
    def woke(self) -> bool:
        return self.at_me or self.keyword


def detect_wake(text: str, segments, self_id, keywords=None) -> WakeHit:
    """识别唤醒条件。segments 只做鸭子类型，不 import 平台组件。

    引用回复（Reply 段 sender_id=机器人）与 @ 等价：引用它说话就是在跟它说话。
    """
    at_me = False
    at_all = False
    for seg in segments or []:
        name = type(seg).__name__
        qq = str(getattr(seg, "qq", "") or "")
        if name == "AtAll" or qq == "all":
            at_all = True
        elif name == "At" and self_id and qq == str(self_id):
            at_me = True
        elif (name == "Reply" and self_id
              and str(getattr(seg, "sender_id", "") or "") == str(self_id)):
            at_me = True
    return WakeHit(at_me=at_me, at_all=at_all,
                   keyword=keyword_hit(text, keywords))


def batch_should_flush(count: int, max_messages: int) -> bool:
    """攒批合并：条数达到上限就立刻回；max<=0 视为不限制。"""
    return max_messages > 0 and count >= max_messages


class RateLimiter:
    """每群滑动窗口频率上限（默认 6 条/分钟），同时兼作 QQ 风控保护。"""

    def __init__(self, per_min: int = 6):
        self.per_min = max(1, int(per_min))
        self._hits: dict[int, list[float]] = {}

    def allow(self, gid: int, now: float | None = None,
              per_min: int | None = None) -> bool:
        limit = max(1, int(per_min if per_min is not None else self.per_min))
        now = time.time() if now is None else float(now)
        gid = int(gid)
        hits = [t for t in self._hits.get(gid, []) if now - t < 60.0]
        if len(hits) >= limit:
            self._hits[gid] = hits
            return False
        hits.append(now)
        self._hits[gid] = hits
        return True
