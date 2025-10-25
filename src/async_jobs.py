"""Async job queue for background batch processing."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobResult:
    """Individual item result within a job."""

    s3_url: str
    video_id: int
    frame_id: int
    ok: bool
    image_id: Optional[str] = None
    num_patches: Optional[int] = None
    error: Optional[str] = None


@dataclass
class Job:
    """Background job for batch ingestion."""

    job_id: str
    status: JobStatus
    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    results: List[JobResult] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "results": [
                {
                    "s3_url": r.s3_url,
                    "video_id": r.video_id,
                    "frame_id": r.frame_id,
                    "ok": r.ok,
                    "image_id": r.image_id,
                    "num_patches": r.num_patches,
                    "error": r.error,
                }
                for r in self.results
            ],
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": (
                self.completed_at - self.started_at
                if self.completed_at and self.started_at
                else None
            ),
        }


class JobQueue:
    """
    In-memory job queue for async batch processing.

    For production, consider using:
    - Redis for job storage
    - Celery for distributed task processing
    - RabbitMQ/SQS for message queuing
    """

    def __init__(self, max_concurrent_jobs: int = 3):
        """
        Initialize job queue.

        Args:
            max_concurrent_jobs: Maximum number of concurrent background jobs
        """
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self._lock = asyncio.Lock()
        self._active_tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, total_items: int) -> str:
        """
        Create a new job.

        Args:
            total_items: Total number of items to process

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            total_items=total_items,
        )
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} with {total_items} items")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    async def update_job_status(self, job_id: str, status: JobStatus, error: Optional[str] = None):
        """Update job status."""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status

                if status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = time.time()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = time.time()

                if error:
                    job.error = error

    async def add_result(self, job_id: str, result: JobResult):
        """Add a result to a job."""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.results.append(result)
                job.processed_items += 1

                if result.ok:
                    job.successful_items += 1
                else:
                    job.failed_items += 1

    async def run_job(
        self,
        job_id: str,
        process_func,
        items: List[Any],
    ):
        """
        Run a job in the background.

        Args:
            job_id: Job identifier
            process_func: Async function to process each item (should return JobResult)
            items: List of items to process
        """
        try:
            await self.update_job_status(job_id, JobStatus.RUNNING)
            logger.info(f"Job {job_id} started processing {len(items)} items")

            # Process items
            for item in items:
                try:
                    result = await process_func(item)
                    await self.add_result(job_id, result)
                except Exception as e:
                    logger.error(f"Job {job_id} item processing failed: {e}", exc_info=True)
                    # Create error result
                    if hasattr(item, "s3_url"):
                        error_result = JobResult(
                            s3_url=item.s3_url,
                            video_id=item.video_id,
                            frame_id=item.frame_id,
                            ok=False,
                            error=str(e),
                        )
                        await self.add_result(job_id, error_result)

            await self.update_job_status(job_id, JobStatus.COMPLETED)
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            await self.update_job_status(job_id, JobStatus.FAILED, error=str(e))

        finally:
            # Clean up task reference
            async with self._lock:
                if job_id in self._active_tasks:
                    del self._active_tasks[job_id]

    def start_job(
        self,
        job_id: str,
        process_func,
        items: List[Any],
    ) -> asyncio.Task:
        """
        Start a job as a background task.

        Args:
            job_id: Job identifier
            process_func: Async function to process items
            items: Items to process

        Returns:
            asyncio.Task running the job
        """
        task = asyncio.create_task(self.run_job(job_id, process_func, items))
        self._active_tasks[job_id] = task
        return task

    def list_jobs(self, limit: int = 100, status: Optional[JobStatus] = None) -> List[Job]:
        """
        List jobs with optional status filter.

        Args:
            limit: Maximum number of jobs to return
            status: Optional status filter

        Returns:
            List of jobs sorted by creation time (newest first)
        """
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def cleanup_old_jobs(self, max_age_seconds: float = 3600):
        """
        Remove completed/failed jobs older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds (default: 1 hour)
        """
        current_time = time.time()
        to_remove = []

        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get or create the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
