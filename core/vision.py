"""图片/动图识别（老板 2026-10-30 定案：看懂表情包/截图，能接图话题）。

- 来源：群消息/私聊消息里的 Image 组件（url 可下载）；
  GIF 用 PIL 抽 3 帧并排（视觉模型当一张图看，省 token）
- 视觉模型：配 vision_model 时用该模型，否则用当前主模型
  （主模型不支持视觉时会失败——失败静默降级为 [图片]，不影响消息链）
- 预算：每日 vision_budget 张（默认 50），超限只记 [图片]，次日恢复
- 结果：一句话描述，进 GroupBuffer/日记的文本就是带描述的（[图片: 一只猫拍键盘]）
"""
from __future__ import annotations

import base64
import io
import time

VISION_PROMPT = (
    "你是群聊图片解说员。用一句中文（25字以内）概括这张图的内容/梗/情绪，"
    "群聊语境里让 AI 看懂这张图在表达什么。直接输出那句话，别解释。"
)

GIF_FRAME_MAX = 3      # 动图抽帧数
GIF_FIT = (448, 448)   # 抽帧后缩放上限（省 token）


def extract_image_urls(segments: list | None) -> list[str]:
    """消息组件列表 → 图片 URL/路径列表（鸭子类型，测试好模拟）。"""
    urls = []
    for seg in segments or []:
        name = type(seg).__name__
        if name == "Image":
            u = (getattr(seg, "url", "") or getattr(seg, "file", "") or "")
            if u and (u.startswith("http") or u.startswith("file:")
                      or u.startswith("base64:")):
                urls.append(u)
    return urls[:3]   # 单条消息最多看 3 张，防刷图烧预算


async def segment_to_png_bytes(seg, aiohttp_session=None) -> bytes | None:
    """Image 组件 → PNG bytes。URL 下载；本地路径直接读。失败返回 None。"""
    import asyncio
    url = (getattr(seg, "url", "") or getattr(seg, "file", "") or "")
    try:
        if url.startswith("base64:"):
            return base64.b64decode(url[len("base64://"):].removeprefix("//")
                                    if url.startswith("base64://")
                                    else url[len("base64:"):])
        if url.startswith("file:"):
            from pathlib import Path
            p = Path(url[len("file://"):].removeprefix("/")
                     if url.startswith("file://") else url[len("file:"):])
            if not p.is_absolute():
                p = Path(str(getattr(seg, "path", "")) or p)
            return p.read_bytes()
        if url.startswith("http"):
            if aiohttp_session is not None:
                async with aiohttp_session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.read()
                return None
            # 无共享 session 时用线程池跑阻塞下载
            def _dl():
                from urllib.request import urlopen, Request
                req = Request(url, headers={"User-Agent": "qqmind/0.2"})
                with urlopen(req, timeout=15) as r:
                    return r.read()
            return await asyncio.to_thread(_dl)
    except Exception:
        return None
    return None


def image_bytes_for_vision(data: bytes) -> str | None:
    """原始图片 bytes → 视觉模型可吃的 data URL。GIF 抽帧并排。失败 None。"""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None
    try:
        im = PILImage.open(io.BytesIO(data))
        frames = []
        if getattr(im, "is_animated", False):
            n = min(GIF_FRAME_MAX, im.n_frames)
            step = max(1, im.n_frames // n)
            for i in range(0, im.n_frames, step):
                im.seek(i)
                frames.append(im.convert("RGBA"))
                if len(frames) >= GIF_FRAME_MAX:
                    break
        else:
            frames = [im.convert("RGBA")]
        fitted = []
        for f in frames:
            f.thumbnail(GIF_FIT)
            fitted.append(f)
        if len(fitted) == 1:
            out = fitted[0]
        else:   # 并排合成一张
            w = sum(f.width for f in fitted)
            h = max(f.height for f in fitted)
            out = PILImage.new("RGBA", (w, h), (255, 255, 255, 255))
            x = 0
            for f in fitted:
                out.paste(f, (x, 0))
                x += f.width
        buf = io.BytesIO()
        out.convert("RGB").save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


class VisionBudget:
    """每日视觉预算（内存态，重启清零——与 _STATS 同口径）。"""

    def __init__(self, daily: int = 50):
        self.daily = max(0, int(daily))
        self.day = ""
        self.used = 0

    def allow(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.used = today, 0
        return self.used < self.daily

    def spend(self, n: int = 1) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.used = today, 0
        self.used += n


async def describe_image(provider, img_url: str, vision_model: str = "") -> str:
    """单张图 → 一句话描述。任何失败返回空串（调用方降级 [图片]）。"""
    try:
        out = await provider.text_chat(
            prompt=VISION_PROMPT, session_id=None,
            system_prompt="你是群聊图片解说员。",
            contexts=[],
            image_urls=[img_url],
            model=vision_model or None)
        return (out.completion_text or "").strip()[:60]
    except Exception:
        return ""
