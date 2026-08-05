import re
import uuid
from typing import Optional, Type, TypeVar
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


def format_short_id(prefix: str, index: int) -> str:
    """Format short ID e.g. prefix='j', index=1 -> 'j001'."""
    return f"{prefix}{index:03d}"


async def resolve_id(
    id_str: Optional[str],
    model: Type[T],
    prefix: str,
    db: AsyncSession,
) -> Optional[uuid.UUID]:
    """Resolve a UUID string OR a short ID string (e.g. 'j001', 'c001', 's001') to a UUID."""
    if not id_str:
        return None

    # Try parsing as standard UUID
    try:
        return uuid.UUID(str(id_str))
    except (ValueError, TypeError, AttributeError):
        pass

    # Try parsing short ID format e.g. j001, c002, s005
    cleaned = str(id_str).strip().lower()
    match = re.match(r"^[a-z]+(\d+)$", cleaned)
    if match:
        idx = int(match.group(1)) - 1  # Convert 1-based index to 0-based offset
        if idx >= 0:
            stmt = select(model.id).order_by(model.created_at.asc()).offset(idx).limit(1)
            res = await db.execute(stmt)
            found_id = res.scalar_one_or_none()
            if found_id is not None:
                return found_id

    return None


async def get_short_id(
    item_id: uuid.UUID,
    model: Type[T],
    prefix: str,
    db: AsyncSession,
) -> str:
    """Find the 1-based index of an item to generate its short ID (e.g. j001, c001, s001)."""
    try:
        stmt = select(model.id).order_by(model.created_at.asc())
        res = await db.execute(stmt)
        all_ids = res.scalars().all()
        idx = list(all_ids).index(item_id) + 1
        return format_short_id(prefix, idx)
    except Exception:
        return f"{prefix}001"
