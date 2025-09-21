import sys, types, functools, resource

def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss * 1024 if sys.platform.startswith("linux") else ru.ru_maxrss

def memory_limit(limit_bytes: int, *, check_every: int = 4096, method: str = "rss"):
    if method == "trace":
        import tracemalloc
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                if isinstance(out, types.GeneratorType) or (
                    hasattr(out, "__iter__") and not isinstance(out, (str, bytes))
                ):
                    def gen():
                        tracemalloc.start()
                        try:
                            # baseline after start -> deltas since here
                            base_cur, base_peak = tracemalloc.get_traced_memory()
                            def _check():
                                cur, peak = tracemalloc.get_traced_memory()
                                if (peak - base_peak) > limit_bytes:
                                    raise MemoryError(
                                        f"Exceeded memory limit: delta_peak={peak-base_peak} > {limit_bytes}"
                                    )
                            for i, item in enumerate(out, 1):
                                if (i % check_every) == 0:
                                    _check()
                                yield item
                            _check()
                        finally:
                            tracemalloc.stop()
                    return gen()
                else:
                    tracemalloc.start()
                    try:
                        base_cur, base_peak = tracemalloc.get_traced_memory()
                        res = out
                        cur, peak = tracemalloc.get_traced_memory()
                        if (peak - base_peak) > limit_bytes:
                            raise MemoryError(
                                f"Exceeded memory limit: delta_peak={peak-base_peak} > {limit_bytes}"
                            )
                        return res
                    finally:
                        tracemalloc.stop()
            return wrapper
        return decorator
    else:  # method == "rss"
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # baseline BEFORE entering user code
                baseline = _peak_rss_bytes()
                allowed = baseline + limit_bytes

                out = func(*args, **kwargs)

                def _check():
                    now = _peak_rss_bytes()
                    if now > allowed:
                        raise MemoryError(f"Exceeded memory limit: delta={now-baseline} > {limit_bytes}")

                if isinstance(out, types.GeneratorType) or (
                    hasattr(out, "__iter__") and not isinstance(out, (str, bytes))
                ):
                    def gen():
                        for i, item in enumerate(out, 1):
                            if (i % check_every) == 0:
                                _check()
                            yield item
                        _check()
                    return gen()
                else:
                    res = out
                    _check()
                    return res
            return wrapper
        return decorator
