"""Bounded process isolation for PyMuPDF, which does not support thread use.

Callers wait from their existing off-event-loop parser threads. Only plain
arguments/results cross the spawn boundary; Document/Page/Pixmap objects stay
inside one child process. Native workers never submit another PDF operation.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import multiprocessing
from pathlib import Path
from threading import Lock

_executor: ProcessPoolExecutor | None = None
_executor_lock = Lock()
_inside_worker = False


def _execute(operation: str, args: tuple):
    global _inside_worker
    _inside_worker = True
    if operation == "parse":
        from .document_representation import _parse_pdf_native
        path, runtime, cache_path = args
        return _parse_pdf_native(Path(path), runtime, Path(cache_path) if cache_path else None)
    if operation == "images":
        from app.api.repository import _extract_images_from_pdf_native
        images, warnings = _extract_images_from_pdf_native(*args)
        return [asdict(image) for image in images], warnings
    raise ValueError("Unknown PDF worker operation")


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
        return _executor


def _discard_executor(executor: ProcessPoolExecutor):
    global _executor
    with _executor_lock:
        if _executor is executor:
            _executor = None
    executor.shutdown(wait=False, cancel_futures=True)


def run_pdf_operation(operation: str, *args):
    """Run an isolated operation; a crashed pool is replaced on the next retry.

    Do not automatically replay an interrupted operation: it may have written
    page caches/images. The durable extraction job controls retries instead.
    """
    if _inside_worker:
        raise RuntimeError("PDF workers cannot submit nested PDF operations")
    executor = _get_executor()
    try:
        return executor.submit(_execute, operation, args).result()
    except BrokenProcessPool:
        _discard_executor(executor)
        raise


def shutdown_pdf_workers():
    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown_pdf_workers)
