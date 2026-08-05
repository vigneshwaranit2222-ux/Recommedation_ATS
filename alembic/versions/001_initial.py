"""Initial schema migration

Revision ID: 001_initial
Revises: 
Create Date: 2026-08-05

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='candidate'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Job Requirements table
    op.create_table(
        'job_requirements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('keywords', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('chroma_doc_id', sa.String(length=255), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Interview Questions table
    op.create_table(
        'interview_questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('job_requirements.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False, server_default='technical'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # Interview Sessions table
    op.create_table(
        'interview_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('candidate_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('job_requirements.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chat_history', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('is_complete', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('final_score', sa.Float(), nullable=True),
        sa.Column('feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )


def downgrade() -> None:
    op.drop_table('interview_sessions')
    op.drop_table('interview_questions')
    op.drop_table('job_requirements')
    op.drop_table('users')
