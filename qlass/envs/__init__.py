from .base import BaseEnv


def __getattr__(name):
    if name == "AlfWorldEnv":
        from .alfworld_env import AlfWorldEnv

        return AlfWorldEnv

    if name == "WebShopEnv":
        from .webshop_env import WebShopEnv

        return WebShopEnv

    if name == "SciWorldEnv":
        from .sciworld_env import SciWorldEnv

        return SciWorldEnv

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "BaseEnv",
    "AlfWorldEnv",
    "WebShopEnv",
    "SciWorldEnv",
]