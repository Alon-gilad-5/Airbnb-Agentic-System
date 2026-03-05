# Evaluation Contracts

This folder contains reproducible evaluation inputs for `scripts/run_results_evaluation.py`.

## Case JSONL schema (shared)

Each line is one JSON object with these required fields:

- `case_id`: unique case identifier
- `split`: `dev` or `test`
- `prompt`: user-style prompt sent to evaluator/agent
- `context`: structured context dict
- `expected`: expected outcomes used for pass/fail checks
- `tags`: list of searchable labels

Evaluators may include extra per-case keys (for example `fixtures`, `mode`, `owner_action`) as long as the shared keys above are present.

## Output artifacts

`scripts/run_results_evaluation.py` writes:

- `outputs/eval/results_summary.json`
- `outputs/eval/per_agent_metrics.csv`
- `outputs/eval/results_tables.md`

Optional manual rubric files live in `eval/manual/`:

- `reviews_rubric_scores.csv`
- `mail_rubric_scores.csv`
