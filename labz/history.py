"""
Persistent chat history for img2md.

Storage layout:
    ~/.labz/history/
        <session-id>.json     one file per session

Each session file:
    {
        "id": "20260425-143022-a3f1",
        "started_at": "2026-04-25T14:30:22",
        "image_path": "/path/to/image.png",   # or null
        "image_summary": "first 120 chars of markdown",
        "messages": [
            {"role": "user",      "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "..."}
        ]
    }
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


HISTORY_DIR = Path.home() / ".labz" / "history"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str        # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class ChatSession:
    id: str
    started_at: str
    messages: List[Message] = field(default_factory=list)
    image_path: Optional[str] = None
    image_summary: Optional[str] = None   # first 120 chars of markdown

    @property
    def started_dt(self) -> datetime:
        return datetime.fromisoformat(self.started_at)

    @property
    def message_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def first_question(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.content[:80] + ("…" if len(m.content) > 80 else "")
        return "(no messages)"

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def to_ollama_history(self) -> list[dict]:
        """Return messages in the format Ollama /api/chat expects."""
        return [{"role": m.role, "content": m.content} for m in self.messages]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def new_session(
    image_path: Optional[str] = None,
    markdown: Optional[str] = None,
) -> ChatSession:
    """Create a new in-memory session (not saved yet)."""
    now = datetime.now()
    session_id = now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    summary = None
    if markdown:
        summary = markdown.strip()[:120].replace("\n", " ")
    return ChatSession(
        id=session_id,
        started_at=now.isoformat(timespec="seconds"),
        image_path=image_path,
        image_summary=summary,
    )


def save_session(session: ChatSession) -> None:
    """Persist *session* to disk."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{session.id}.json"
    data = {
        "id": session.id,
        "started_at": session.started_at,
        "image_path": session.image_path,
        "image_summary": session.image_summary,
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.messages
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_session(session_id: str) -> ChatSession:
    """Load a session by ID. Raises FileNotFoundError if not found."""
    path = HISTORY_DIR / f"{session_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = [Message(**m) for m in data.get("messages", [])]
    return ChatSession(
        id=data["id"],
        started_at=data["started_at"],
        messages=messages,
        image_path=data.get("image_path"),
        image_summary=data.get("image_summary"),
    )


def list_sessions() -> List[ChatSession]:
    """Return all saved sessions, newest first."""
    if not HISTORY_DIR.exists():
        return []
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            sessions.append(load_session(f.stem))
        except Exception:
            pass
    return sorted(sessions, key=lambda s: s.started_at, reverse=True)


def delete_session(session_id: str) -> bool:
    """Delete a session by ID. Returns True if deleted, False if not found."""
    path = HISTORY_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def delete_all_sessions() -> int:
    """Delete all sessions. Returns the number deleted."""
    if not HISTORY_DIR.exists():
        return 0
    count = 0
    for f in HISTORY_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
