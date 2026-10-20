"""Structured cancellation for parallel model work."""

import asyncio


async def gather_owned(*operations):
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
