"""Release decision. Quality failures and infrastructure errors stay distinct."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from quantguard.policy.engine import PolicyCheck

GateStatus = Literal["PASS", "FAIL", "ERROR", "WARNING"]

EXIT_CODES: dict[GateStatus, int] = {
    "PASS": 0,
    "FAIL": 1,
    "ERROR": 2,
    "WARNING": 0,
}


class ReleaseDecision(BaseModel):
    status: GateStatus
    exit_code: int
    reasons: list[str] = Field(default_factory=list)
    checks: list[PolicyCheck] = Field(default_factory=list)


def decide(checks: list[PolicyCheck], *, infrastructure_error: str | None = None) -> ReleaseDecision:
    if infrastructure_error:
        return ReleaseDecision(
            status="ERROR",
            exit_code=EXIT_CODES["ERROR"],
            reasons=[infrastructure_error],
            checks=checks,
        )

    required_failures = [check for check in checks if check.required and check.status == "fail"]
    advisory_failures = [check for check in checks if not check.required and check.status == "fail"]
    missing = [check for check in checks if check.status == "missing" and check.required]

    if missing:
        reasons = [f"Required metric missing: {check.metric}" for check in missing]
        return ReleaseDecision(
            status="ERROR",
            exit_code=EXIT_CODES["ERROR"],
            reasons=reasons,
            checks=checks,
        )
    if required_failures:
        return ReleaseDecision(
            status="FAIL",
            exit_code=EXIT_CODES["FAIL"],
            reasons=[check.reason for check in required_failures],
            checks=checks,
        )
    if advisory_failures:
        return ReleaseDecision(
            status="WARNING",
            exit_code=EXIT_CODES["WARNING"],
            reasons=[check.reason for check in advisory_failures],
            checks=checks,
        )
    return ReleaseDecision(
        status="PASS",
        exit_code=EXIT_CODES["PASS"],
        reasons=["All required policies passed."],
        checks=checks,
    )
