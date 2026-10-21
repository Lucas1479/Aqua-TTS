"""Cooperative process-wide admission for CUDA work and graph capture.

Work may overlap work. Capture is admitted only when no other work owner is
active; otherwise it is deferred. A yielded stream retains its owner but never
retains a thread-local binding or a thread-owned lock across yield.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import threading


class CudaExecution:
    def __init__(self):
        self._condition = threading.Condition()
        self._owners = set()
        self._capturing = False
        self._current = ContextVar("aqua_cuda_owner", default=None)

    def _owner(self):
        binding = self._current.get()
        return binding[0] if binding is not None and binding[1] == threading.get_ident() else None

    @contextmanager
    def reserve(self):
        inherited = self._owner()
        with self._condition:
            if inherited in self._owners:
                owner, new = inherited, False
            else:
                while self._capturing:
                    self._condition.wait()
                owner, new = object(), True
                self._owners.add(owner)
        try:
            yield owner
        finally:
            if new:
                with self._condition:
                    self._owners.remove(owner)
                    self._condition.notify_all()

    @contextmanager
    def work(self, owner=None):
        if owner is None:
            with self.reserve() as reserved, self.work(reserved):
                yield reserved
            return
        with self._condition:
            if owner not in self._owners:
                raise RuntimeError("CUDA work owner is no longer active")
        token = self._current.set((owner, threading.get_ident()))
        try:
            yield owner
        finally:
            self._current.reset(token)

    @contextmanager
    def try_capture(self):
        owner = self._owner()
        with self._condition:
            admitted = not self._capturing and not (self._owners - {owner})
            if admitted:
                self._capturing = True
        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._capturing = False
                    self._condition.notify_all()

    def decorate(self, function):
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def stream(*args, **kwargs):
                with self.reserve() as owner:
                    iterator = function(*args, **kwargs)
                    try:
                        while True:
                            with self.work(owner):
                                try:
                                    item = next(iterator)
                                except StopIteration:
                                    return
                            yield item
                    finally:
                        with self.work(owner):
                            iterator.close()
            return stream

        @wraps(function)
        def call(*args, **kwargs):
            with self.work():
                return function(*args, **kwargs)
        return call


cuda_execution = CudaExecution()
cuda_work = cuda_execution.decorate
