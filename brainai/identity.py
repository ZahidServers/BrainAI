# BrainAI - built by Mohammed Zahid.
"""Identity, attribution and naming.

Attribution is immutable.  Every model instance, every checkpoint and every
serialised config carries ``built by Mohammed Zahid``, and loading a checkpoint
whose attribution has been stripped or altered raises :class:`AttributionError`.

The *display name* is free.  An operator may rename a running system to
anything they like -- ``Javri``, ``Atlas``, ``Sahar`` -- and only the display
name changes; the authorship travels with the weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

#: Immutable authorship.  Not a default, not a placeholder.
AUTHOR = "Mohammed Zahid"

#: The architecture's permanent codename, independent of any deployment's name.
CODENAME = "BrainAI"

DEFAULT_DISPLAY_NAME = "BrainAI"
VERSION = "0.1.0"
PURPOSE = "embodied cognition and humanoid robot control"

_VALID_NAME = re.compile(r"^[\w \-'.]{1,48}$", re.UNICODE)


class AttributionError(RuntimeError):
    """Raised when a checkpoint's attribution is missing or has been altered."""


class InvalidName(ValueError):
    """Raised when a requested display name is not usable."""


@dataclass
class Identity:
    """Who the system is, and who built it.

    ``display_name`` is mutable; ``author`` is a read-only property that always
    resolves to :data:`AUTHOR`, so it cannot be reassigned on an instance.
    """

    display_name: str = DEFAULT_DISPLAY_NAME
    purpose: str = PURPOSE
    version: str = VERSION

    def __post_init__(self) -> None:
        self.display_name = _clean_name(self.display_name)

    @property
    def author(self) -> str:
        return AUTHOR

    @property
    def codename(self) -> str:
        return CODENAME

    def rename(self, new_name: str) -> "Identity":
        """Rename this deployment in place and return self, for chaining."""
        self.display_name = _clean_name(new_name)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_name": self.display_name,
            "author": AUTHOR,
            "codename": CODENAME,
            "purpose": self.purpose,
            "version": self.version,
            "attribution": attribution_line(self.display_name),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Identity":
        verify(data)
        return cls(
            display_name=data.get("display_name", DEFAULT_DISPLAY_NAME),
            purpose=data.get("purpose", PURPOSE),
            version=data.get("version", VERSION),
        )

    def banner(self) -> str:
        name = self.display_name
        alias = "" if name == CODENAME else f"  (an instance of {CODENAME})"
        return (
            f"{name} v{self.version}{alias}\n"
            f"built by {AUTHOR}\n"
            f"purpose: {self.purpose}"
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return attribution_line(self.display_name)


def _clean_name(name: str) -> str:
    if not isinstance(name, str):
        raise InvalidName("display name must be a string")
    name = name.strip()
    if not name:
        raise InvalidName("display name must not be empty")
    if not _VALID_NAME.match(name):
        raise InvalidName(
            "display name may contain letters, digits, spaces, '-', '.', \"'\" "
            "and must be 1-48 characters"
        )
    return name


def attribution_line(display_name: str = DEFAULT_DISPLAY_NAME) -> str:
    """The one-line attribution stamped into configs, checkpoints and logs."""
    if display_name == CODENAME:
        return f"{CODENAME} - built by {AUTHOR}."
    return f"{display_name} - {CODENAME} architecture, built by {AUTHOR}."


def verify(data: Dict[str, Any]) -> None:
    """Raise :class:`AttributionError` unless ``data`` still credits the author.

    Called on every checkpoint load.  Renaming the deployment is fine; removing
    the author is not.
    """
    if not isinstance(data, dict):
        raise AttributionError("identity block missing from checkpoint")
    author = data.get("author")
    if author != AUTHOR:
        raise AttributionError(
            f"checkpoint attribution altered: expected author {AUTHOR!r}, "
            f"found {author!r}. {CODENAME} refuses to load unattributed weights."
        )
    if data.get("codename") != CODENAME:
        raise AttributionError(
            f"checkpoint codename altered: expected {CODENAME!r}, "
            f"found {data.get('codename')!r}."
        )
