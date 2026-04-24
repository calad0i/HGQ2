def register() -> None:
    """Entry point for the `alkaid-keras` second-level plugin.

    Importing each submodule triggers Alkaid's `HandlerRegMeta` on every
    handler class, registering it into `_registry`.
    """
    from . import activation, attn, core, ops, table  # noqa: F401
