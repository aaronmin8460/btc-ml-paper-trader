from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc_now() -> str:
    return utc_now().isoformat()
