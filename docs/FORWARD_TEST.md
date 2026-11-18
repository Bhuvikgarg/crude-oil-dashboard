# Forward test — how it works and how to run it

The backtest and the lockbox experiment both used **historical** data. However carefully that is
done, the person designing the model has seen the data. A forward test removes that: the models
are frozen now, predictions are recorded live, and the outcomes do not exist yet when each
prediction is written down.

## The three pieces

| Piece | File | What it does |
|---|---|---|
| Pre-registration | `forward_test/PREREGISTRATION.md`, `forward_test/manifest.json` | Frozen models (with SHA-256 fingerprints), thresholds, test window, success criteria and a power analysis, all committed **before** the test starts |
| Hourly logger | `.github/workflows/forward-test.yml` → `python -m evaluation.forward_test run` | Every hour: downloads live data, records each completed hourly bar, scores every call whose 4-hour outcome is now known, commits to the `forward-test-log` branch |
| Report | `FORWARD_TEST_REPORT.md` on the `forward-test-log` branch | Running results and verdicts against the pre-registered criteria |

## What is being tested

| Track | Question |
|---|---|
| A. Engine baseline | How does the original dashboard do live? (reference) |
| B. Calibrated probability | Does the deployed `calibrated_p_up_4h` stay at least as good as the base rate? |
| C. Confident ML | Does the gradient-boosting model's top-30 % most confident 4-hour calls beat 50 %, after correcting for drift (Pesaran–Timmermann)? |
| D. Improved playbook | Do the wider-stop, ML-filtered setups beat the engine's own setups? |

## Why the evidence is trustworthy
1. **Frozen models.** The logger refuses to run if a model file no longer matches its pre-registered SHA-256.
2. **Timestamped log.** Each prediction is committed to git within the hour, before its outcome is known. The commit history of `forward-test-log` is public proof.
3. **Point-in-time features.** The logger uses the same replay function as the backtest, so features only see data available at the bar's close.
4. **Gaps are flagged.** Bars recorded more than 75 minutes late (scheduler delays, outages) are marked `backfilled`, and the report shows results with and without them.
5. **One amendment, made before the start and disclosed.** The fixed confidence threshold from the lockbox experiment would have called only ~5 % of hours with the final model, so the primary rule became "top 30 % relative to the model's own last 500 hours". Its backtest numbers are not lockbox-clean, which the pre-registration states. The original fixed rule is still logged as a secondary result.

## Setup checklist (one time)
1. **Merge this branch into the default branch** (`main`). GitHub runs scheduled workflows only from the default branch.
2. In GitHub → **Settings → Actions → General → Workflow permissions**, select **Read and write permissions**, so the job can push to `forward-test-log`.
3. Go to **Actions → Forward test logger → Run workflow** once to check it works. A `forward-test-log` branch should appear with `calls.csv`, `runs.csv` and `FORWARD_TEST_REPORT.md`.
4. Leave it running until the end date in the pre-registration.

## Things that can go wrong
| Problem | What happens | What to do |
|---|---|---|
| GitHub delays scheduled runs (common at busy times) | Late bars are caught up on the next run and flagged `backfilled` | Nothing; check the backfilled share in the report |
| Yahoo blocks or rate-limits GitHub's servers | The run fails and is logged as `download_error` in `runs.csv`; the next run catches up | If it persists, run `python -m evaluation.forward_test run` on your own computer (needs `requirements-eval.txt`) and commit `forward_test/log/` |
| GitHub disables scheduled workflows in repositories with no activity for 60 days | Logging stops | The hourly log commits normally count as activity; if GitHub shows a "workflow disabled" banner, re-enable it in the Actions tab |
| Someone edits a model file | The logger stops with a fingerprint error | Restore the file from the pre-registration commit |

## Commands
```bash
python -m evaluation.forward_test run                       # record + score (local log in forward_test/log/)
python -m evaluation.forward_test score --log-dir ftlog     # re-score an existing log
python -m evaluation.forward_test freeze                    # ONLY before the test starts; changes the pre-registration
```

## How to read the result
Read the power note in the pre-registration first. In 91 days, track C is expected to make about
90–110 non-overlapping calls, against the ~764 needed to detect a 54.5 % hit rate reliably. So:
- **REJECTED** (hit rate ≤ 50 %) is a clear, useful answer: the backtest edge did not survive.
- **INCONCLUSIVE** with a hit rate above 50 % means "consistent with the backtest, keep collecting".
- **CONFIRMED** in 91 days would need either a larger edge than the backtest showed, or luck. Be sceptical.
