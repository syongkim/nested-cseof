"""Panel letters (a), (b), … for the nested-CSEOF figures."""
from __future__ import annotations


def label_panels(axes, start: str = "a", titles=None) -> None:
    """Draw (a), (b), … on each panel, with an optional short subtitle."""
    try:
        seq = list(axes.ravel())
    except AttributeError:
        seq = list(axes)
    n = len(seq)
    if titles is None:
        titles = [""] * n
    for i, ax in enumerate(seq):
        letter = f"({chr(ord(start) + i)})"
        t = titles[i] if i < len(titles) else ""
        ax.set_title(
            letter if not t else f"{letter}  {t}",
            loc="center",
            fontsize=10,
            fontweight="bold",
            pad=5,
        )
