"""人格槽位拼装器（基线版）。

设计意图：系统提示词不是一张"人设卡"，而是由若干**槽位**拼装而成。
槽位框架先立好，槽位的内容与拼装策略由老板亲自逐步设计——本模块只保证：
- 每个槽位可独立开关、独立编辑（web 控制台按槽位渲染编辑器）
- 拼装顺序固定，输出可预测、可单测
- 模型可输出 <生图> / <查历史> 等标记，这段规则文本也属于一个槽位
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 内置槽位 key（顺序即拼装顺序）
SLOT_PERSONA = "persona"          # 人设卡：身份、性格、语气
SLOT_RULES = "rules"              # 行为规则：分段风格、emoji 禁令等
SLOT_PROFILE = "profile"          # 该用户画像与长期记忆（运行时注入）
SLOT_GROUP_CTX = "group_ctx"      # 最近群聊语境（运行时注入）
SLOT_IDENTITY = "identity"        # 机器人本群身份（名片/自我介绍规则）
SLOT_MARKERS = "markers"          # 画图/回捞标记的使用说明


@dataclass
class PersonaSlots:
    """槽位集合。slots 里未显式配置的内置槽位用默认值。"""

    slots: dict[str, str] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        if self.enabled.get(key) is False:
            return ""
        return self.slots.get(key, default)

    def set(self, key: str, text: str, enabled: bool | None = None) -> None:
        self.slots[key] = text or ""
        if enabled is not None:
            self.enabled[key] = enabled

    # ---------- 内置默认文本（取自生产验证版，可直接跑） ----------
    DEFAULT_MARKERS = (
        "画图规则（严格遵守）：只有对方明确要你\"画图/生成图片\"时，"
        "才在回复末尾单独一行输出 <生图\"英文画面描述\">；"
        "对方指定画法/模型时写成 <生图\"英文画面描述\"@模型名>。\n"
        "这些情况绝不输出画图标记，用文字和颜文字回应就好：让你做表情或动作、"
        "只是聊到画画相关话题、比喻或玩梗的说法、对方没有明确提到要图或要画。\n"
        "查询命令：对方说没看到/翻不到/想回看/图没了时，输出命令让我去查——"
        "<查历史\"条数\"> 看这位用户更早的聊天、<查群聊\"条数\"> 看群里更早的消息"
        "（条数1~50）、<查图> 把最近生成的图重发一次；"
        "私聊里可以翻日记：<查日记\"关键词\"> 按关键词、<查日记\"昨天/上周/2026-10-20\"> "
        "按时间，对方说\"你还记得吗/之前聊过/我们上次\"时主动翻日记；"
        "命令单独成行，结果会回喂给你，你基于结果回答。\n"
        "对方只@了你一下什么都没说时，你先主动开口打个招呼、找点话题。"
        "不知道的事情诚实说不知道。"
    )

    def build_system_prompt(self, *, profile: str = "", group_ctx: str = "",
                            self_card: str = "") -> str:
        """按固定顺序拼装：人设 → 规则 → 画像 → 群语境 → 身份 → 标记说明。

        空槽位整段跳过（不留空标题）。
        """
        parts: list[str] = []
        persona = self.get(SLOT_PERSONA)
        if persona:
            parts.append(persona)
        rules = self.get(SLOT_RULES)
        if rules:
            parts.append(rules)
        if profile:
            parts.append("【关于这位用户的画像与记忆】\n" + profile)
        if group_ctx:
            parts.append("【刚才群里的聊天（供理解语境，不必逐条回应）】\n" + group_ctx)
        if self_card:
            parts.append(
                f"【你自己的名字】本群名片「{self_card}」；群里有人这么叫就是在叫你。"
                "别人问你是谁/叫什么时，按人设自然回答名字就好，"
                "绝对不要把QQ号等账号信息说出口或写出来")
        if self.get(SLOT_MARKERS, self.DEFAULT_MARKERS):
            parts.append(self.get(SLOT_MARKERS, self.DEFAULT_MARKERS))
        return "\n".join(p for p in parts if p.strip())
