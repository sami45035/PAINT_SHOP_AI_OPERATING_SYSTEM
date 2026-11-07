"""Number and text formatting shared by every screen (Indian rupee grouping etc.)."""


def rupees(x: float, decimals: int = 0) -> str:
    """₹2,20,868 — Indian digit grouping."""
    neg = x < 0
    x = abs(float(x))
    whole = int(x)
    frac = f"{x - whole:.{decimals}f}"[1:] if decimals else ""
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if neg else "") + "₹" + s + frac


def rupees_short(x: float) -> str:
    """₹47.4 L, ₹1.2 Cr, ₹88,400 — for KPI cards."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.1f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    return rupees(x)


def pack(litres: float) -> str:
    return f"{float(litres):g} L"


def plural(n: int, word: str, many: str = "") -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"
