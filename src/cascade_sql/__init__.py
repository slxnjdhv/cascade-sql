"""cascade_sql: confidence-gated small-to-large cascades for Text-to-SQL.

Pipeline stages (each is a CLI subcommand, see `python -m cascade_sql --help`):

    prepare   -> split manifests (questions + db paths + gold SQL per label version)
    generate  -> run a model (local or large) on a split, cache JSONL records
    label     -> mark each record correct/incorrect under every label version
    features  -> add confidence features f1..f10
    fit-gate  -> fit logistic gate + Platt calibration
    evaluate  -> E1 signal quality, E2 cascade curves, E3 reliability, E4 risk
    demo      -> run the whole pipeline on a toy database with a mock model
"""

__version__ = "0.1.0"
