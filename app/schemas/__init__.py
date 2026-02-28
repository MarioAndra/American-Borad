# Pydantic schema package for request/response models.
#
# Responsibility:
# - Define data validation and serialization models exposed by the API layer.
# - Separate external schemas from internal ORM entities.
#
# Planned modules:
# - user.py: UserCreate, UserRead, UserUpdate, Role enum mirrors.
# - auth.py: Token, LoginRequest, RegisterRequest, LogoutResponse.
# - abet.py: ABETCriterionCreate/Read.
# - question.py: QuestionCreate/Read with difficulty enum.
# - choice.py: ChoiceCreate/Read.
# - exam.py: ExamCreate, ExamRead, ExamSubmission, ExamResult.
#
# Limitations (skeleton phase):
# - No Pydantic model implementations provided.

