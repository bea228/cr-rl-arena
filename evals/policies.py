"""Policy spec strings -> Policy objects (one parser for gauntlet, client, BC recording, training env).

    random                       bot.policy.RandomPolicy(seed)
    scripted                     bot.scripted.ScriptedPolicy()
    learned:<path>               bot.learned.LearnedPolicy(<path>)
    class:<module>.<Class>       any Policy class; gets seed= if its __init__ accepts it
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import PurePath

VALID_FORMS = ("random", "scripted", "learned:<path>", "class:<module>.<Class>")


def _bad(spec) -> ValueError:
    return ValueError(f"unknown policy spec {spec!r}; valid forms: {', '.join(VALID_FORMS)}")


def make_policy(spec: str, seed=None):
    if not isinstance(spec, str):
        raise _bad(spec)
    if spec == "random":
        from bot.policy import RandomPolicy

        return RandomPolicy(seed)
    if spec == "scripted":
        from bot.scripted import ScriptedPolicy

        return ScriptedPolicy()
    if spec.startswith("learned:"):
        path = spec[len("learned:"):]
        if not path:
            raise _bad(spec)
        try:
            from bot.learned import LearnedPolicy
        except ModuleNotFoundError as e:
            if e.name == "bot.learned":
                raise ValueError(
                    f"policy spec {spec!r} needs bot/learned.py, which doesn't exist yet"
                ) from e
            raise
        return LearnedPolicy(path)
    if spec.startswith("class:"):
        target = spec[len("class:"):]
        module_name, _, cls_name = target.rpartition(".")
        if not module_name or not cls_name:
            raise _bad(spec)
        try:
            cls = getattr(importlib.import_module(module_name), cls_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"policy spec {spec!r}: cannot load {target} ({e})") from e
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            params = {}
        return cls(seed=seed) if "seed" in params else cls()
    raise _bad(spec)


_UNSAFE = re.compile(r'[:/\\<>"|?*\s]+')


def spec_slug(spec: str) -> str:
    """Filename-safe version of a spec (Windows-safe)."""
    if spec.startswith("learned:"):
        spec = "learned:" + str(PurePath(spec[len("learned:"):]).with_suffix("").as_posix())
    slug = _UNSAFE.sub("-", spec)
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return slug or "policy"
