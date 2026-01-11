from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class JobSummary(BaseModel):
    """Summary of job execution counts for a SQL query."""

    success_job_ids: Sequence[int] = Field(..., alias="successJobsIds")
    failed_job_ids: Sequence[int] = Field(..., alias="failedJobsIds")
    running_job_ids: Sequence[int] = Field(..., alias="runningJobsIds")

    model_config = ConfigDict(populate_by_name=True)


class SqlQuerySummary(BaseModel):
    """Simplified summary of a SQL query execution for LLM consumption."""

    id: int
    duration: Optional[int] = None  # Duration in milliseconds
    description: Optional[str] = None
    status: str
    submission_time: Optional[str] = Field(None, alias="submissionTime")
    plan_description: str = Field(..., alias="planDescription")
    job_summary: JobSummary = Field(..., alias="jobSummary")

    model_config = ConfigDict(populate_by_name=True)
