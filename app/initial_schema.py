"""initial schema: users, jobs, questions, interview_sessions

NOTE: jobs/questions/interview_sessions columns below are inferred from
the features you described (JD generation, question bank with 5-10
validation, interview chat + scoring, resume ranking). If your actual
models.py already defines these tables with different column names,
replace the op.create_table() bodies for those three tables with
`sa.inspect`-derived DDL matching your real models — do not run this
against a DB that already has these tables.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-04
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    user_role = sa.Enum("company", "candidate", name="user_role")
    user_role.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String, nullable=False, unique=True, index=True),
        sa.Column("hashed_password", sa.String, nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("company_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("company_overview", sa.Text),
        sa.Column("required_skills", sa.JSON),
        sa.Column("responsibilities", sa.JSON),
        sa.Column("evaluation_criteria", sa.JSON),
        sa.Column("raw_input_text", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "questions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("order_index", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "interview_sessions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("candidate_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("chat_history", sa.JSON),
        sa.Column("is_completed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("overall_score", sa.Float),
        sa.Column("technical_rating", sa.Float),
        sa.Column("communication_rating", sa.Float),
        sa.Column("recommendation", sa.String),
        sa.Column("summary_reasoning", sa.Text),
        sa.Column("meet_link", sa.String),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("interview_sessions")
    op.drop_table("questions")
    op.drop_table("jobs")
    op.drop_table("users")
    sa.Enum(name="user_role").drop(op.get_bind(), checkfirst=True)