"""expand cognitive_level enum to 5-level Bloom taxonomy

Revision ID: 015
Revises: 014
Create Date: 2026-08-05

Remaps existing ``questions.cognitive_level`` rows from the legacy 3-value
taxonomy to the model-aligned 5-value taxonomy:

- ``Knowledge``     -> ``RememberUnderstand``
- ``Application``   -> ``Apply``
- ``Analysis``      -> ``Analyze``

Newly introduced values (``Evaluate``, ``Create``) only appear in questions
created after this migration via the Bloom classifier.

The enum type is swapped safely:
1. rename the legacy type away,
2. widen the column to text,
3. remap row values,
4. create the new enum type,
5. cast the column to it,
6. drop the legacy type.

Downgrade remaps back and is lossy for rows that carry ``Evaluate`` or
``Create`` (both collapse to ``Analysis``) since the legacy type cannot
represent them.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE cognitive_level RENAME TO cognitive_level_legacy")

    op.execute(
        "ALTER TABLE questions ALTER COLUMN cognitive_level "
        "TYPE VARCHAR(64) USING cognitive_level::text"
    )

    op.execute(
        """
        UPDATE questions SET cognitive_level = CASE cognitive_level
            WHEN 'Knowledge' THEN 'RememberUnderstand'
            WHEN 'Application' THEN 'Apply'
            WHEN 'Analysis' THEN 'Analyze'
            ELSE cognitive_level
        END
        """
    )

    op.execute(
        "CREATE TYPE cognitive_level AS ENUM "
        "('RememberUnderstand','Apply','Analyze','Evaluate','Create')"
    )

    op.execute(
        "ALTER TABLE questions ALTER COLUMN cognitive_level "
        "TYPE cognitive_level USING cognitive_level::cognitive_level"
    )

    op.execute("DROP TYPE cognitive_level_legacy")


def downgrade() -> None:
    op.execute("ALTER TYPE cognitive_level RENAME TO cognitive_level_new")

    op.execute(
        "ALTER TABLE questions ALTER COLUMN cognitive_level "
        "TYPE VARCHAR(64) USING cognitive_level::text"
    )

    op.execute(
        """
        UPDATE questions SET cognitive_level = CASE cognitive_level
            WHEN 'RememberUnderstand' THEN 'Knowledge'
            WHEN 'Apply' THEN 'Application'
            WHEN 'Analyze' THEN 'Analysis'
            WHEN 'Evaluate' THEN 'Analysis'
            WHEN 'Create' THEN 'Analysis'
            ELSE cognitive_level
        END
        """
    )

    op.execute(
        "CREATE TYPE cognitive_level AS ENUM "
        "('Knowledge','Application','Analysis')"
    )

    op.execute(
        "ALTER TABLE questions ALTER COLUMN cognitive_level "
        "TYPE cognitive_level USING cognitive_level::cognitive_level"
    )

    op.execute("DROP TYPE cognitive_level_new")
