# Requirements files

These pinned-baseline files mirror the extras declared in `pyproject.toml`
and exist for users / CI that prefer `pip install -r requirements/<extra>.txt`
over `pip install -e ".[<extra>]"`.

They are intentionally loose (lower bounds only). For fully-locked
deployments, generate a lock file with `pip-compile` or `uv pip compile`.
