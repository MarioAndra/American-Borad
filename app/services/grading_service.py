# Grading service placeholder.
#
# Responsibility:
# - Evaluate a student's submitted answers for an exam and compute score.
# - Compare selected choices with the correct answers from the DB.
# - Handle both SingleChoice and MultipleSelect question types:
#   - SingleChoice: selection must match the sole correct choice.
#   - MultipleSelect: the set of selected choices must exactly match the set of correct choices.
# - Update the exam status to Completed and set submitted_at timestamp.
#
# Planned contents:
# - grade_exam(exam_id, submissions): compute correct count and percentage.
# - persist_answers(exam_id, answers): store StudentAnswer rows with is_correct.
# - evaluate_question(question_id, selected_choice_ids): returns boolean correct flag.
# - finalize_exam(exam_id, score): update status and timestamps.
# - safeguards to prevent resubmission or grading multiple times.
#
# Formula:
# - score = (correct_answers / total_questions) * 100
#
# Limitations (skeleton phase):
# - No DB operations or calculations implemented.
