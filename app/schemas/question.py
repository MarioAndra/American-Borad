# Pydantic schemas for questions.
#
# Responsibility:
# - Represent question data with difficulty, cognitive level, type, and taxonomy references.
# - Include additional metadata fields: explanation, common_mistake, skill_gap.
#
# Planned contents:
# - Difficulty, CognitiveLevel, QuestionType enums mirroring app.models.enums.
# - QuestionBase, QuestionCreate (subtopic_id, abet criterion id, choices), QuestionRead.
# - Include nested choices for read if required.
# - Support multi-correct selection for MultipleSelect questions.
#
# Limitations (skeleton phase):
# - No Pydantic classes implemented.
