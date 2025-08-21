import uuid
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, JSON, DateTime, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Manual(Base):
    __tablename__ = "manuals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False)
    vendor_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Core Fields
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    machine_name = Column(String(255), nullable=False)
    model_no = Column(String(255), nullable=True)
    machine_type = Column(String(255), nullable=True)
    serial_no = Column(String(255), nullable=True)
    year = Column(String(4), nullable=True)  # YEAR is stored as string(4) in MySQL with SQLAlchemy

    # JSON Fields
    tags = Column(JSON, nullable=True)
    videos = Column(JSON, nullable=True)
    purchase_links = Column(JSON, nullable=True)

    # File path
    path = Column(String(500), nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Manual(id={self.id}, uuid='{self.uuid}', title='{self.title}', machine='{self.machine_name}')>"
