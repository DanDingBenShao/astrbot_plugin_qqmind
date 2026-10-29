"""中文 FTS5 适配：unicode61 不切汉字，整句是一个 token → BM25 全部失效。

方案：写入时把文本切成**相邻二字组**（去重、空格分隔）存进 FTS 的 grams 列，
查询时把查询串同样切成二字组 OR 检索（BM25 排序兜底相关性）。
实测（2026-10-28）：「今天加班到很晚」的库里查「加班」「番禺」均命中并正确排序；
英文/数字串用二字组 phrase（顺序相邻）等效整词子串匹配。
trigram tokenizer 需要 ≥3 字符查询，中文两字词（加班/火锅）用不了，故不采用。
"""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s")


def bigrams(s: str) -> str:
    """文本 → 相邻二字组（空格分隔、去重保序）。单字符输入返回空串。"""
    s = _WS_RE.sub("", s or "")
    if len(s) < 2:
        return ""
    return " ".join(dict.fromkeys(s[i:i + 2] for i in range(len(s) - 1)))


def grams_query(q: str) -> str:
    """查询串 → FTS MATCH 表达式。

    中文：相邻二字组 OR（召回优先，bm25 排序）；单字查询没有二字组，返回空串
    （调用方走 LIKE 兜底）。英文/数字：二字组 phrase，顺序匹配等效整词子串。
    """
    s = _WS_RE.sub("", q or "")
    if not s:
        return ""
    if s.isascii():
        if len(s) < 2:
            return f'"{s}"'
        return '"' + " ".join(s[i:i + 2] for i in range(len(s) - 1)) + '"'
    grams = dict.fromkeys(s[i:i + 2] for i in range(len(s) - 1))
    return " OR ".join(f'"{g}"' for g in grams)
