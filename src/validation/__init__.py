"""Step 4 verification: independent, read-only validation of the two rating methods.

``python -m src.validation.validate`` regenerates ``reports/validation.md`` from the
processed data (``data/processed``), the saved scalers and the test suite. Nothing in
this package writes to ``data/`` or changes any other module.
"""
