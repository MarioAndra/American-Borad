from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "2b4d23011ec2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "questions",
        "question_type",
        existing_type=sa.Enum("SingleChoice", "MultipleSelect", name="question_type"),
        type_=sa.String(length=255),
        existing_nullable=False,
        postgresql_using="question_type::text",
    )


def downgrade() -> None:
    op.alter_column(
        "questions",
        "question_type",
        existing_type=sa.String(length=255),
        
        type_=sa.Enum("SingleChoice", "MultipleSelect", name="question_type"),
        existing_nullable=False,
        postgresql_using="question_type::question_type", 
    )
