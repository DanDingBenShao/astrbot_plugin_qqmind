"""main 层基础 helper（C2 解耦自 main.py，v0.3.9）。

模块级纯函数：平台 ID 边界转换、引用展开、戳一戳探测。
handlers/ 与 main.py 共用；main.py re-export 保持旧引用零改动。
"""
from __future__ import annotations

import re

from astrbot.api import logger

# 详细场景识别（v0.2.10）：对方在要"讲清楚"的内容时放宽群聊截断。
# 命中面=求做法/求原因/求解释/讲题/求证/纠错，不含纯闲聊。
_DETAIL_QUERY_RE = re.compile(
    r"怎么(做|弄|办|改|装|配|算|解|写|用|回|搞|部署|安装|设置|使用|配置|"
    r"运行|跑|实现|连|接|下载|更新|升级|导入|导出|迁移|删除|注册|登录|"
    r"同步|备份|恢复|调试|排查|修复|优化|开|关|调|加|去|上|传)|"
    r"为啥|为什么|什么意思|啥意思|什么区别|"
    r"详解|详细|细说|展开说|讲讲|讲一下|解释一下|原理|教程|步骤|推导|"
    r"帮我(看看|算|推|查|分析)|求证|对不对|对吗|哪里(错|出错了)|(错|问题)在(哪|哪里)|"
    r"解(一)?下|这道题|算式|证明|分析一下|[-+×÷=^]|\d+\s*[-+×÷^]\s*\d")


def _is_detail_query(query: str) -> bool:
    """这条用户消息需要详细回答吗（讲题/教程/求解释）？命中则放宽截断。"""
    q = (query or "").strip()
    if not q or len(q) < 4:
        return False
    return bool(_DETAIL_QUERY_RE.search(q))


def _quote_reply_prefix(segments) -> str:
    """把引用消息（Reply 组件）拼成可读前缀：[引用 某某：「原文」]。

    适配器已把被引用原文/引用人填进 Reply 段（sender_nickname/message_str），
    这里只做鸭子类型提取（有 sender_id+sender_nickname 属性即视为引用段，
    与 detect_wake 的宽松判定一致），取不到的字段静默降级；最多展开 1 条、
    原文截 60 字——引用是语境不是正文，别让它吃掉回复预算。
    """
    for seg in segments or []:
        if type(seg).__name__ == "Reply" or (
                hasattr(seg, "sender_id") and hasattr(seg, "sender_nickname")
                and not hasattr(seg, "qq")):
            who = str(getattr(seg, "sender_nickname", "") or "").strip()
            body = str(getattr(seg, "message_str", "")
                       or getattr(seg, "text", "") or "").strip()
            body = re.sub(r"\s+", " ", body)[:60]
            if not who and not body:
                return ""
            return f"[引用 {who or '有人'}：「{body}」] "
    return ""


def _safe_int_id(value) -> int:
    """平台 ID → int；非数字 ID（匿名/频道/第三方平台）返回 0。

    qqmind 的记忆、账本、发送链路都按数字 QQ 号设计；遇到
    32 位十六进制之类的平台 ID 时不能直接 int() 把 on_message
    整个炸掉，边界处统一转 0，让调用方静默跳过这条事件。
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if 0 <= value < 2**63 else 0
    s = str(value or "").strip()
    if not s.isdigit():
        if s:
            logger.debug(f"[qqmind] non-numeric platform id ignored: {s[:64]!r}")
        return 0
    try:
        v = int(s)
    except ValueError:
        return 0
    return v if v < 2**63 else 0


def _is_poke_self(segments, self_id: str) -> bool:
    """这条事件是不是「有人戳了机器人」。

    戳一戳以 Poke 组件到达（message_str 恒空）。目标字段随协议端版本
    漂移（id / target_id / qq），全部鸭子探测；target 取不到（空/0）时
    视为戳自己——notice 事件由会话路由，默认收件人就是被戳者。
    """
    for seg in segments or []:
        if type(seg).__name__ != "Poke":
            continue

        def _field(name: str) -> str:
            v = getattr(seg, name, "")
            # Poke.target_id 是方法（respond.stage 用 comp.target_id() 判断），
            # 不是字段——别把绑定方法/None 当目标值
            if callable(v):
                v = v()
            return str(v) if v not in ("", None, 0) else ""

        target = (_field("id") or _field("target_id")
                  or _field("qq")).strip()
        if not self_id or target in ("", "0") or target == str(self_id):
            return True
    return False
