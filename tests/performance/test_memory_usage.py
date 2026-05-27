import tracemalloc


def test_memory_usage():

    tracemalloc.start()

    data = [i for i in range(1000000)]

    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    assert peak < 200_000_000  # 200MB threshold