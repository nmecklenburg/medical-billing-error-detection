# Deprecated Test Suite

This directory contains archival tests for the legacy script-based data
pipeline in `scripts/_deprecated/`. They are intentionally excluded from the
default test gate.

Default active suite:

```bash
uv run python -m unittest discover tests
```

Manual legacy suite:

```bash
uv run python -m unittest discover tests_deprecated
```

Migration notes:

- Assertions tied to removed or deprecated CLI/module paths were not ported to
  the active suite because the installable `claim.*` package is now the source
  of truth.
- The old outpatient-biased final-anchor selection checks were not ported as
  1:1 package tests because the package now materializes all encounters first
  and samples later with inpatient-aware eligibility rules.
- The active replacements for high-value legacy behavior now live in
  `tests/test_gold_bill_trilliant.py`, `tests/test_datagen_io.py`,
  `tests/test_datagen_encounters.py`, `tests/test_datagen_clean.py`, and
  `tests/test_datagen_pricing.py`.
