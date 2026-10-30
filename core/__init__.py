"""qqmind 记忆内核（与 AstrBot 零耦合，可独立单测）。"""
from .core import (JUDGE_PROMPT, MAX_JUDGED_PER_ROUND, GroupBuffer,
                   MonitorSession, UserMemory, est_tokens, thinking_model,
                   parse_judge_verdict, split_bubbles)
from .diary import DiaryStore, parse_time_query
from .budget import DailyBudget
from .chapters import ChapterStore
from .group import (DEFAULT_PRESET, PRESETS, READ_MODES, RateLimiter, WakeHit,
                    batch_should_flush,
                    detect_wake, keyword_hit, keyword_wake_enabled,
                    normalize_preset, preset_memory_mode, preset_read_mode,
                    resolve_read_mode)
from .private import PrivateSession
from .group_memory import GroupMemory
from .importer import (ImportJob, PROFILE_PROMPT, chunk_messages,
                       extract_profile, normalize_messages, run_import)
from .vision import (VisionBudget, VISION_PROMPT, describe_image,
                     extract_image_urls, image_bytes_for_vision)
from .proactive import (ProactiveState, build_greeting, in_quiet_hours,
                        memory_digest, pick_target)
from .affection import (AffectionStore, in_schedule, parse_schedule,
                        profile_to_prompt)
