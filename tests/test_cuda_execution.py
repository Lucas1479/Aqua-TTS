from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from aquatts.inference.cuda_execution import CudaExecution


def test_nested_work_can_capture_but_other_owner_forces_deferral():
    execution = CudaExecution()
    with execution.work(), execution.work():
        with execution.try_capture() as admitted:
            assert admitted
        with execution.reserve() as same:
            assert same in execution._owners and len(execution._owners) == 1
    with execution.reserve(), execution.reserve():
        with execution.try_capture() as admitted:
            assert not admitted


def test_new_work_waits_for_capture_and_exception_releases_gate():
    execution = CudaExecution()
    entered = threading.Event()
    def work():
        with execution.work():
            entered.set()
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="capture failed"):
            with execution.try_capture() as admitted:
                assert admitted
                pending = pool.submit(work)
                assert not entered.wait(.05)
                raise RuntimeError("capture failed")
        pending.result(timeout=5)
    assert entered.is_set() and not execution._owners and not execution._capturing


def test_yielded_streams_keep_distinct_owners_and_can_resume_or_close_on_another_thread():
    execution = CudaExecution()
    @execution.decorate
    def stream():
        for _ in range(2):
            with execution.try_capture() as admitted:
                yield_value = admitted
            yield yield_value
    first, second = stream(), stream()
    assert next(first)
    assert not next(second)
    assert len(execution._owners) == 2 and execution._current.get() is None
    with ThreadPoolExecutor(1) as pool:
        assert not pool.submit(next, first).result(timeout=5)
        pool.submit(first.close).result(timeout=5)
        assert pool.submit(next, second).result(timeout=5)
        pool.submit(second.close).result(timeout=5)
    assert not execution._owners
