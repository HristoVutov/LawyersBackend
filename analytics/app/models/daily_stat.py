"""
DailyStat model - Pre-aggregated daily statistics.
"""
from datetime import datetime
from sqlalchemy import String, Integer, Date
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DailyStat(Base):
    """
    Pre-aggregated daily statistics for dashboard queries.
    
    Populated by a background job or trigger.
    """
    __tablename__ = "daily_stats"
    
    date: Mapped[datetime] = mapped_column(Date, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(50), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_users: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
