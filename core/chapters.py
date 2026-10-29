"""背景故事篇章库（老板 2026-10-28 定案：另类日记，使用者自己写）。

一篇 = 标题 + 摘要 + 正文。
- 常驻提示词的只有标题行（篇名列表）
- 命中关键词才把摘要+正文带进上下文（FTS5 BM25，unicode61 字切分）
- 每个角色/主题一个包、一个 SQLite，零 embedding、零额外模型
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from .fts_zh import bigrams, grams_query

_SAFE_RE = re.compile(r'[\\/:*?"<>|\s]+')


def pack_filename(pack: str) -> str:
    safe = _SAFE_RE.sub("_", (pack or "").strip()).strip(" .")
    return (safe or "default") + ".db"


class ChapterStore:
    """单个角色包的篇章库。"""

    _OPEN: list["ChapterStore"] = []

    def __init__(self, packs_dir, pack: str):
        self.pack = (pack or "").strip() or "default"
        self.path = Path(packs_dir) / pack_filename(self.pack)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        ChapterStore._OPEN.append(self)

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS chapters (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      REAL NOT NULL,
            title   TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS ch_fts USING fts5(
            grams, title, summary,
            tokenize='unicode61',
            content='chapters', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS ch_ai AFTER INSERT ON chapters BEGIN
            INSERT INTO ch_fts(rowid, grams, title, summary)
            VALUES (new.id, new.grams, new.title, new.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS ch_ad AFTER DELETE ON chapters BEGIN
            INSERT INTO ch_fts(ch_fts, rowid, grams, title, summary)
            VALUES ('delete', old.id, old.grams, old.title, old.summary);
        END;
    """
    _FTS_VER = "2"   # meta.fts_ver：2=grams 列已建并回填

    def _init_schema(self) -> None:
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(chapters)")}
        existed = bool(self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chapters'"
        ).fetchone())
        fts_cols = {r[1] for r in self._db.execute("PRAGMA table_info(ch_fts)")}
        rebuilding = bool(fts_cols) and "grams" not in fts_cols
        if rebuilding:
            self._db.execute("DROP TRIGGER IF EXISTS ch_ai")
            self._db.execute("DROP TRIGGER IF EXISTS ch_ad")
            self._db.execute("DROP TABLE IF EXISTS ch_fts")
        self._db.executescript(self._SCHEMA)
        if "grams" not in cols:
            self._db.execute(
                "ALTER TABLE chapters ADD COLUMN grams TEXT NOT NULL DEFAULT ''")
        row = self._db.execute("SELECT v FROM meta WHERE k='fts_ver'").fetchone()
        if rebuilding or not existed or (row and row[0] != self._FTS_VER) or not row:
            self._backfill_grams()
            self._db.execute("INSERT INTO ch_fts(ch_fts) VALUES ('rebuild')")
            self._db.execute(
                "INSERT INTO meta(k, v) VALUES ('fts_ver', ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (self._FTS_VER,))
        self._db.commit()

    def _backfill_grams(self) -> None:
        """检索面 = 标题+摘要（正文不索引，按设计只召回标题/摘要命中的篇）。"""
        for r in self._db.execute(
                "SELECT id, title, summary FROM chapters "
                "WHERE IFNULL(grams,'')='' ORDER BY id").fetchall():
            self._db.execute("UPDATE chapters SET grams=? WHERE id=?",
                             (bigrams(f"{r['title']} {r['summary']}"), r["id"]))

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
        try:
            ChapterStore._OPEN.remove(self)
        except ValueError:
            pass

    @classmethod
    def close_all(cls) -> None:
        for inst in list(cls._OPEN):
            inst.close()

    # ---------- 写 ----------
    def add(self, title: str, summary: str, content: str,
            ts: float | None = None) -> int:
        # grams 是 chapters 的真实列（external content 的 fts 触发器从这里取）
        grams = bigrams(f"{(title or '').strip()} {(summary or '').strip()}")
        cur = self._db.execute(
            "INSERT INTO chapters(ts, title, summary, content, grams) VALUES (?,?,?,?,?)",
            (float(ts or time.time()), (title or "").strip()[:80],
             (summary or "").strip()[:300], (content or "").strip(), grams))
        self._db.commit()
        return cur.lastrowid

    def update(self, chapter_id: int, title: str, summary: str,
               content: str) -> bool:
        rid = int(chapter_id)
        # 必须在 UPDATE 前取旧行：'delete' 指令的列值要和索引时完全一致
        old = self._db.execute(
            "SELECT grams, title, summary FROM chapters WHERE id=?", (rid,)).fetchone()
        if old is None:
            return False
        grams = bigrams(f"{(title or '').strip()} {(summary or '').strip()}")
        cur = self._db.execute(
            "UPDATE chapters SET title=?, summary=?, content=?, grams=? WHERE id=?",
            ((title or "").strip()[:80], (summary or "").strip()[:300],
             (content or "").strip(), grams, rid))
        if cur.rowcount:
            # 同步 FTS：先按旧行发 delete，再按新行重插（否则之后 DELETE 触发器
            # 用新 grams 对不上索引里的旧 token，报 malformed）
            self._db.execute(
                "INSERT INTO ch_fts(ch_fts, rowid, grams, title, summary) "
                "VALUES ('delete', ?, ?, ?, ?)",
                (rid, old["grams"], old["title"], old["summary"]))
            self._db.execute(
                "INSERT INTO ch_fts(rowid, grams, title, summary) VALUES (?,?,?,?)",
                (rid, grams, (title or "").strip()[:80],
                 (summary or "").strip()[:300]))
        self._db.commit()
        return cur.rowcount > 0

    def delete(self, chapter_id: int) -> bool:
        cur = self._db.execute("DELETE FROM chapters WHERE id=?",
                               (int(chapter_id),))
        self._db.commit()
        return cur.rowcount > 0

    # ---------- 查 ----------
    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]

    def list_rows(self, n: int = 200) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM chapters ORDER BY id ASC LIMIT ?",
            (max(1, min(int(n), 500)),)).fetchall()

    def headers_for_prompt(self, n: int = 20) -> str:
        """常驻提示词的篇目列表：只放标题，不放摘要/正文。"""
        rows = self._db.execute(
            "SELECT title FROM chapters ORDER BY id ASC LIMIT ?",
            (max(0, min(int(n), 50)),)).fetchall()
        return "\n".join(f"- {r['title']}" for r in rows if r["title"])

    def search(self, query: str, n: int = 2) -> str:
        """关键词命中 → 返回摘要+正文（最多 n 篇）。无命中返回空串。

        检索面 = 标题+摘要的二字组（fts_zh.grams_query，BM25 排序）；
        自然句查询的相邻二字组 OR 能召回「摘要含这些词」的篇章。
        """
        q = (query or "").strip()
        if not q:
            return ""
        match = grams_query(q)
        rows = []
        if match:
            try:
                rows = self._db.execute(
                    "SELECT c.* FROM ch_fts JOIN chapters c ON c.id = ch_fts.rowid "
                    "WHERE ch_fts MATCH ? ORDER BY bm25(ch_fts) LIMIT ?",
                    (match, max(1, min(int(n), 10)))).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            rows = self._db.execute(
                "SELECT * FROM chapters WHERE title LIKE ? OR summary LIKE ? "
                "ORDER BY id ASC LIMIT ?",
                (f"%{q}%", f"%{q}%", max(1, min(int(n), 10)))).fetchall()
        parts = []
        for r in rows:
            block = f"【{r['title']}】"
            if r["summary"]:
                block += f"\n摘要：{r['summary']}"
            if r["content"]:
                block += f"\n{r['content']}"
            parts.append(block)
        return "\n\n".join(parts)
