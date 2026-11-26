import time


def elapsed(start: float) -> str:
    s = int(time.perf_counter() - start)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "concluindo..."
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"~{h}h {m:02d}m"
    if m:
        return f"~{m}m {s:02d}s"
    return f"~{s}s"
