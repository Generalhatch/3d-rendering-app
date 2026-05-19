"""Global concurrency limits for background pipeline jobs."""
import asyncio

# Allow at most 2 pipeline jobs to run simultaneously.
# Each job consumes ~1.5 GB RAM at voxel_size=0.03m; 2 concurrent = ~3 GB peak.
# Increase this on machines with >16 GB RAM, decrease for shared/low-memory hosts.
JOB_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(2)
