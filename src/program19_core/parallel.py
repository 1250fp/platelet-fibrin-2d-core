"""Process-based execution of independent mechanical layers.

The GUI itself uses a thread only to remain responsive.  Actual CPU
parallelism is provided here with separate Python processes so the mechanical
solver is not serialized by Python's global interpreter lock.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from collections.abc import Callable, Iterable


ProgressCallback = Callable[[dict], None]
ResultCallback = Callable[[int, object], None]


def logical_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def effective_worker_count(requested: int, task_count: int) -> int:
    """Cap the requested process count by the jobs and visible logical CPUs."""
    return max(1, min(int(requested), int(task_count), logical_cpu_count()))


def numeric_thread_limit():
    """Avoid nested BLAS/OpenMP oversubscription inside each layer process."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional dependency
        return nullcontext()
    return threadpool_limits(limits=1)


def throttled_progress_relay(progress_queue, minimum_interval_s: float = 0.20):
    """Return a worker-local callback that avoids one IPC message per time step."""
    last_time = 0.0
    last_fraction = -1.0

    def relay(payload: dict) -> None:
        nonlocal last_time, last_fraction
        now = time.monotonic()
        fraction = float(payload.get("layer_fraction", payload.get("fraction", 0.0)))
        if (
            now - last_time >= minimum_interval_s
            or fraction >= 1.0 - 1e-12
            or fraction - last_fraction >= 0.02
        ):
            progress_queue.put(payload)
            last_time = now
            last_fraction = fraction

    return relay


def run_process_jobs(
    worker: Callable,
    jobs: Iterable[tuple],
    *,
    requested_workers: int,
    layer_count: int,
    stage: str,
    progress_callback: ProgressCallback | None,
    result_callback: ResultCallback,
    pause_event: threading.Event,
    cancel_event: threading.Event,
) -> int:
    """Execute layer jobs with process-safe pause, cancellation, and progress.

    ``jobs`` contains argument tuples whose first element is the zero-based
    layer index.  The worker receives three additional arguments: a shared
    pause event, a shared cancellation event, and a shared progress queue.
    Results are delivered in the parent process through ``result_callback``.
    """
    job_list = list(jobs)
    workers = effective_worker_count(requested_workers, len(job_list))
    if workers <= 1:
        raise ValueError("run_process_jobs requires at least two effective workers")

    context = mp.get_context("spawn")
    first_error: BaseException | None = None
    layer_progress = [0.0] * layer_count
    completed: set[int] = set()

    with context.Manager() as manager:
        process_pause = manager.Event()
        process_cancel = manager.Event()
        progress_queue = manager.Queue()
        if pause_event.is_set():
            process_pause.set()

        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(worker, *job, process_pause, process_cancel, progress_queue):
                int(job[0])
                for job in job_list
            }
            pending = set(futures)

            def synchronize_controls() -> None:
                if pause_event.is_set():
                    process_pause.set()
                else:
                    process_pause.clear()
                if cancel_event.is_set() or first_error is not None:
                    process_cancel.set()

            def relay_available_progress() -> None:
                while True:
                    try:
                        payload = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    layer_index = int(payload.get("layer_index", 1)) - 1
                    if 0 <= layer_index < layer_count:
                        layer_progress[layer_index] = max(
                            layer_progress[layer_index],
                            float(payload.get("layer_fraction", 0.0)),
                        )
                    if progress_callback:
                        progress_callback(
                            {
                                **payload,
                                "fraction": sum(layer_progress) / max(layer_count, 1),
                                "completed_layers": len(completed),
                                "parallel_workers": workers,
                                "active_workers": min(workers, len(pending)),
                            }
                        )

            while pending:
                synchronize_controls()
                relay_available_progress()
                done, pending = wait(pending, timeout=0.10, return_when=FIRST_COMPLETED)
                for future in done:
                    layer_index = futures[future]
                    if future.cancelled():
                        continue
                    try:
                        result = future.result()
                    except BaseException as exc:  # preserve partial completed layers
                        if first_error is None:
                            first_error = exc
                        process_cancel.set()
                        for waiting in pending:
                            waiting.cancel()
                        continue
                    result_status = (
                        result[1]
                        if isinstance(result, tuple) and len(result) > 1
                        else "completed"
                    )
                    if result_status == "completed":
                        completed.add(layer_index)
                        layer_progress[layer_index] = 1.0
                    result_callback(layer_index, result)
                    if progress_callback:
                        progress_callback(
                            {
                                "type": "progress",
                                "stage": stage,
                                "fraction": sum(layer_progress) / max(layer_count, 1),
                                "layer_index": layer_index + 1,
                                "layer_count": layer_count,
                                "layer_fraction": 1.0,
                                "completed_layers": len(completed),
                                "parallel_workers": workers,
                                "active_workers": min(workers, len(pending)),
                            }
                        )
                synchronize_controls()
            relay_available_progress()

    if first_error is not None:
        raise first_error
    return workers
