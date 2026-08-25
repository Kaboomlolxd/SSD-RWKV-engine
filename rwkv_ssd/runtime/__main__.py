"""CLI entry: ``python -m rwkv_ssd.runtime.engine``."""

from __future__ import annotations


def main() -> None:
    from app.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
