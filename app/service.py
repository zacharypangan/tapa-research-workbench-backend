"""Run API and worker together so both use the service's storage volume."""

import asyncio
import os
import signal
import sys


async def serve() -> int:
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)

    processes = []
    waits = []
    try:
        for arguments in (
            ("uvicorn", "main:app", "--host", "0.0.0.0", "--port", os.getenv("PORT", "8000")),
            ("arq", "app.progress.worker.WorkerSettings"),
        ):
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", *arguments)
            processes.append(process)
            waits.append(asyncio.create_task(process.wait()))
        stop_wait = asyncio.create_task(stopped.wait())
        waits.append(stop_wait)
        done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        return 0 if stop_wait in done else 1
    finally:
        for process in processes:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for task in waits:
            task.cancel()
        await asyncio.gather(*waits, return_exceptions=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(serve()))
