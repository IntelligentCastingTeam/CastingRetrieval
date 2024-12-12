import sys
import time

import pytest

sys.path.append("../src")

format = "%Y-%m-%D %H:%M:%S"


@pytest.fixture(scope="session", autouse=True)
def time_session():
    start = time.time()
    print(f"\nstart: {time.strftime(format, time.localtime(start))}")

    yield

    finished = time.time()
    print(f"finished: {time.strftime(format, time.localtime(finished))}")
    print(f"Total time cost: {finished - start:.3f}s")


@pytest.fixture(scope="function", autouse=True)
def time_function():
    start = time.time()
    print(f"\nstart: {time.strftime(format, time.localtime(start))}")

    yield

    finished = time.time()
    print(f"finished: {time.strftime(format, time.localtime(finished))}")
    print(f"Time cost: {finished - start:.3f}s")
