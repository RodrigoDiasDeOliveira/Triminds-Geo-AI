import tracemalloc


def test_memory_usage():
    tracemalloc.start()

    data = [i for i in range(1_000_000)]

    current, peak = tracemalloc.get_traced_memory()

    assert len(data) == 1_000_000
    assert current > 0
    assert peak < 200_000_000  # 200 MB threshold

    tracemalloc.stop()
